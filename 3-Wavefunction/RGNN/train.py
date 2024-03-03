#Reference:

# [1] D. Pfau, J.S. Spencer, A.G.D.G. Matthews, W.M.C. Foulkes, Ab initio solution of the many-electron Schrödinger equation with deep neural networks, Phys. Rev. Res. 2(3) (2020) 033429.

#This code is extended from https://github.com/google-deepmind/ferminet.git, which has the Apache License, Version 2.0, January 2004.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.


"""Core training loop for neural QMC in JAX."""

import functools
import importlib
import time
from typing import Optional, Sequence, Tuple, Union
from jax import profiler
from absl import logging
import chex
from RGNN import checkpoint
from RGNN import constants
from RGNN import curvature_tags_and_blocks
from RGNN import envelopes
from RGNN import hamiltonian
from RGNN import loss as qmc_loss_functions
from RGNN import mcmc
from RGNN import networks
from RGNN import psiformer
from RGNN.utils import statistics
from RGNN.utils import system
from RGNN.utils import writers
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import kfac_jax
import ml_collections
import numpy as np
import optax
from typing_extensions import Protocol
import pickle
import random

def call_counter(func):
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        print(f"{func.__name__} has been called {wrapper.count} times")
        return func(*args, **kwargs)

    wrapper.count = 0
    return wrapper

def _assign_spin_configuration(
    nalpha: int, nbeta: int, batch_size: int = 1
) -> jnp.ndarray:
  """Returns the spin configuration for a fixed spin polarisation."""
  spins = jnp.concatenate((jnp.ones(nalpha), -jnp.ones(nbeta)))
  return jnp.tile(spins[None], reps=(batch_size, 1))


def init_electrons(
    key,
    molecule: Sequence[system.Atom],
    electrons: Sequence[int],
    batch_size: int,
    init_width: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Initializes electron positions around each atom.

  Args:
    key: JAX RNG state.
    molecule: system.Atom objects making up the molecule.
    electrons: tuple of number of alpha and beta electrons.
    batch_size: total number of MCMC configurations to generate across all
      devices.
    init_width: width of (atom-centred) Gaussian used to generate initial
      electron configurations.

  Returns:
    array of (batch_size, (nalpha+nbeta)*ndim) of initial (random) electron
    positions in the initial MCMC configurations and ndim is the dimensionality
    of the space (i.e. typically 3), and array of (batch_size, (nalpha+nbeta))
    of spin configurations, where 1 and -1 indicate alpha and beta electrons
    respectively.
  """
  if sum(atom.charge for atom in molecule) != sum(electrons):
    if len(molecule) == 1:
      atomic_spin_configs = [electrons]
    else:
      raise NotImplementedError('No initialization policy yet '
                                'exists for charged molecules.')
  else:
    atomic_spin_configs = [
        (atom.element.nalpha, atom.element.nbeta) for atom in molecule
    ]
    assert sum(sum(x) for x in atomic_spin_configs) == sum(electrons)
    while tuple(sum(x) for x in zip(*atomic_spin_configs)) != electrons:
      i = np.random.randint(len(atomic_spin_configs))
      nalpha, nbeta = atomic_spin_configs[i]
      atomic_spin_configs[i] = nbeta, nalpha

  # Assign each electron to an atom initially.
  electron_positions = []
  for i in range(2):
    for j in range(len(molecule)):
      atom_position = jnp.asarray(molecule[j].coords)
      electron_positions.append(
          jnp.tile(atom_position, atomic_spin_configs[j][i]))
  electron_positions = jnp.concatenate(electron_positions)
  # Create a batch of configurations with a Gaussian distribution about each
  # atom.
  key, subkey = jax.random.split(key)
  electron_positions += (
      jax.random.normal(subkey, shape=(batch_size, electron_positions.size))
      * init_width
  )

  electron_spins = _assign_spin_configuration(
      electrons[0], electrons[1], batch_size
  )

  return electron_positions, electron_spins


# All optimizer states (KFAC and optax-based).
OptimizerState = Union[optax.OptState, kfac_jax.optimizer.OptimizerState]
OptUpdateResults = Tuple[networks.ParamTree, Optional[OptimizerState],
                         jnp.ndarray,
                         Optional[qmc_loss_functions.AuxiliaryLossData]]


class OptUpdate(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: optax.OptState,
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters accordingly.

    Args:
      params: network parameters.
      data: electron positions, spins and atomic positions.
      opt_state: optimizer internal state.
      key: RNG state.

    Returns:
      Tuple of (params, opt_state, loss, aux_data), where params and opt_state
      are the updated parameters and optimizer state, loss is the evaluated loss
      and aux_data auxiliary data (see AuxiliaryLossData docstring).
    """


StepResults = Tuple[
    networks.FermiNetData,
    networks.ParamTree,
    Optional[optax.OptState],
    jnp.ndarray,
    qmc_loss_functions.AuxiliaryLossData,
    jnp.ndarray,
]


class Step(Protocol):

  def __call__(
      self,
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: OptimizerState,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """Performs one set of MCMC moves and an optimization step.

    Args:
      data: batch of MCMC configurations, spins and atomic positions.
      params: network parameters.
      state: optimizer internal state.
      key: JAX RNG state.
      mcmc_width: width of MCMC move proposal. See mcmc.make_mcmc_step.

    Returns:
      Tuple of (data, params, state, loss, aux_data, pmove).
        data: Updated MCMC configurations drawn from the network given the
          *input* network parameters.
        params: updated network parameters after the gradient update.
        state: updated optimization state.
        loss: energy of system based on input network parameters averaged over
          the entire set of MCMC configurations.
        aux_data: AuxiliaryLossData object also returned from evaluating the
          loss of the system.
        pmove: probability that a proposed MCMC move was accepted.
    """

def null_update(
    params: networks.ParamTree,
    data: networks.FermiNetData,
    opt_state: Optional[optax.OptState],
    key: chex.PRNGKey,
    wrams = None
) -> OptUpdateResults:
  """Performs an identity operation with an OptUpdate interface."""
  del data, key
  return params, opt_state, jnp.zeros(1), None


def make_opt_update_step(evaluate_loss: qmc_loss_functions.LossFn,
                         optimizer: optax.GradientTransformation) -> OptUpdate:
  """Returns an OptUpdate function for performing a parameter update."""

  # Differentiate wrt parameters (argument 0)
  #loss_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

  def opt_update(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
      wrams = None,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters using optax."""
    (loss, aux_data), grad = loss_and_grad(params, key, data, wrams, key)
    #loss, aux_data = evaluate_loss(params, key, data,wrams = None,)
    grad = constants.pmean(grad)
    #grad 
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss,  aux_data

  return opt_update


def make_loss_step(evaluate_loss: qmc_loss_functions.LossFn) -> OptUpdate:
  """Returns an OptUpdate function for evaluating the loss."""

  def loss_eval(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
      wrams = None,
  ) -> OptUpdateResults:
    """Evaluates just the loss and gradients with an OptUpdate interface."""
    loss, aux_data = evaluate_loss(params, key, data, wrams)
    return params, opt_state, loss, aux_data

  return loss_eval


def make_training_step(
    mcmc_step,
    optimizer_step: OptUpdate,
    evaluate_loss: qmc_loss_functions.LossFn,
) -> Step:
  """Factory to create traning step for non-KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    optimizer_step: OptUpdate callable which evaluates the forward and backward
      passes and updates the parameters and optimizer state, as required.

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  @functools.partial(constants.pmap, donate_argnums=(0, 1, 2))
  def step(
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: Optional[optax.OptState],
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
      wrams = None,
      
  ) -> StepResults:
    """A full update iteration (except for KFAC): MCMC steps + optimization."""
    # MCMC loop
    mcmc_key, loss_key = jax.random.split(key, num=2)
    data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width, wrams)      # 1
    # Optimization step
    #loss, aux_data = evaluate_loss(params, key, data, wrams)
    new_params, state, loss, aux_data = optimizer_step(params, data, state,  # 2
    loss_key, wrams)
    #state = None
    return data, params, state, loss, aux_data, pmove

  return step

class CallCounter:
    def __init__(self, func):
        self.func = func
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return self.func(*args, **kwargs)

def make_kfac_training_step(mcmc_step, damping: float,
                            optimizer: kfac_jax.Optimizer) -> Step:
  """Factory to create traning step for KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    damping: value of damping to use for each KFAC update step.
    optimizer: KFAC optimizer instance.

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
  shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
  shared_damping = kfac_jax.utils.replicate_all_local_devices(
      jnp.asarray(damping))
  
  @CallCounter
  def step(
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: kfac_jax.optimizer.OptimizerState,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
      wrams = None,
  ) -> StepResults:
    """A full update iteration for KFAC: MCMC steps + optimization."""
    # KFAC requires control of the loss and gradient eval, so everything called
    # here must be already pmapped.
    # MCMC loop
    mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)
    data, pmove = mcmc_step(params, data, mcmc_keys, mcmc_width, wrams)
    # Optimization step
    new_params, state, stats = optimizer.step(
        params=params,
        state=state,
        rng=loss_keys,
        batch=data,
        momentum=shared_mom,
        damping=shared_damping,
        #wrams = None,
    )
    return data, new_params, state, stats['loss'], stats['aux'], pmove

  return step


def train(cfg: ml_collections.ConfigDict, writer_manager=None, wrams = None):
  """Runs training loop for QMC.

  Args:
    cfg: ConfigDict containing the system and training parameters to run on. See
      base_config.default for more details.
    writer_manager: context manager with a write method for logging output. If
      None, a default writer (ferminet.utils.writers.Writer) is used.

  Raises:
    ValueError: if an illegal or unsupported value in cfg is detected.
  """
  # Device logging
  num_devices = jax.local_device_count()
  num_hosts =  jax.device_count() // num_devices
  logging.info('Starting QMC with %i XLA devices per host '
               'across %i hosts.', num_devices, num_hosts)
  if cfg.batch_size % (num_devices * num_hosts) != 0:
    raise ValueError('Batch size must be divisible by number of devices, '
                     f'got batch size {cfg.batch_size} for '
                     f'{num_devices * num_hosts} devices.')
  host_batch_size = cfg.batch_size // num_hosts  # batch size per host
  device_batch_size = host_batch_size // num_devices  # batch size per device
  data_shape = (num_devices, device_batch_size)

  # Check if mol is a pyscf molecule and convert to internal representation
  if cfg.system.pyscf_mol:
    cfg.update(
        system.pyscf_mol_to_internal_representation(cfg.system.pyscf_mol))

  # Convert mol config into array of atomic positions and charges
  atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
  charges = jnp.array([atom.charge for atom in cfg.system.molecule])
  nspins = cfg.system.electrons

  # Generate atomic configurations for each walker
  batch_atoms = jnp.tile(atoms[None, ...], [device_batch_size, 1, 1])
  batch_atoms = kfac_jax.utils.replicate_all_local_devices(batch_atoms)
  batch_charges = jnp.tile(charges[None, ...], [device_batch_size, 1])
  batch_charges = kfac_jax.utils.replicate_all_local_devices(batch_charges)

  if cfg.debug.deterministic:
    seed = 23
  else:
    seed = jnp.asarray([1e6 * time.time()])
    seed = int(multihost_utils.broadcast_one_to_all(seed)[0])
  key = jax.random.PRNGKey(seed)

  # Create parameters, network, and vmaped/pmaped derivations
  
  if cfg.network.make_feature_layer_fn:
    feature_layer_module, feature_layer_fn = (
        cfg.network.make_feature_layer_fn.rsplit('.', maxsplit=1))
    feature_layer_module = importlib.import_module(feature_layer_module)
    make_feature_layer: networks.MakeFeatureLayer = getattr(
        feature_layer_module, feature_layer_fn
    )
    feature_layer = make_feature_layer(
        natoms=charges.shape[0],
        nspins=cfg.system.electrons,
        ndim=cfg.system.ndim,
        **cfg.network.make_feature_layer_kwargs)
  else:
    feature_layer = networks.make_ferminet_features(
        natoms=charges.shape[0],
        nspins=cfg.system.electrons,
        ndim=cfg.system.ndim,
        rescale_inputs=cfg.network.get('rescale_inputs', False),
    )

  if cfg.network.make_envelope_fn:
    envelope_module, envelope_fn = (
        cfg.network.make_envelope_fn.rsplit('.', maxsplit=1))
    envelope_module = importlib.import_module(envelope_module)
    make_envelope = getattr(envelope_module, envelope_fn)
    envelope = make_envelope(**cfg.network.make_envelope_kwargs)  # type: envelopes.Envelope
  else:
    envelope = envelopes.make_isotropic_envelope()
   
  if cfg.network.network_type == 'ferminet':
    network = networks.make_fermi_net(
        nspins,
        charges,
        ndim=cfg.system.ndim,
        determinants=cfg.network.determinants,
        envelope=envelope,
        feature_layer=feature_layer,
        jastrow=cfg.network.get('jastrow', 'default'),
        bias_orbitals=cfg.network.bias_orbitals,
        full_det=cfg.network.full_det,
        rescale_inputs=cfg.network.get('rescale_inputs', False),
        complex_output=cfg.network.get('complex', False),
        **cfg.network.ferminet,
    )
  elif cfg.network.network_type == 'psiformer':
    network = psiformer.make_fermi_net(
        nspins,
        charges,
        ndim=cfg.system.ndim,
        determinants=cfg.network.determinants,
        envelope=envelope,
        feature_layer=feature_layer,
        jastrow=cfg.network.get('jastrow', 'default'),
        bias_orbitals=cfg.network.bias_orbitals,
        rescale_inputs=cfg.network.get('rescale_inputs', False),
        complex_output=cfg.network.get('complex', False),
        **cfg.network.psiformer,
    )
  key, subkey = jax.random.split(key)
  params, _ = network.init(subkey)
  params = kfac_jax.utils.replicate_all_local_devices(params)
  signed_network = network.apply
  # Often just need log|psi(x)|.
   
  
  logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
  batch_network = jax.vmap(logabs_network, in_axes=(None, 0, 0, 0, 0, None,None), out_axes=0)  # batched network
  # Exclusively when computing the gradient wrt the energy for complex
  # wavefunctions, it is necessary to have log(psi) rather than log(|psi|).
  # This is unused if the wavefunction is real-valued.
  def log_network(*args, **kwargs):
    if not cfg.network.get('complex', False):
      raise ValueError('This function should never be used if the '
                       'wavefunction is real-valued.')
    phase, mag = signed_network(*args, **kwargs)
    return mag + 1.j * phase

  
  if True:
    key, subkey = jax.random.split(key)
    # make sure data on each host is initialized differently
    subkey = jax.random.fold_in(subkey, jax.process_index())
    pos, spins = init_electrons(
        subkey,
        cfg.system.molecule,
        cfg.system.electrons,
        batch_size=host_batch_size,
        init_width=cfg.mcmc.init_width,
    )
    pos = jnp.reshape(pos, data_shape + pos.shape[1:])
    pos = kfac_jax.utils.broadcast_all_local_devices(pos)
    spins = jnp.reshape(spins, data_shape + spins.shape[1:])
    spins = kfac_jax.utils.broadcast_all_local_devices(spins)
    data = networks.FermiNetData(
        positions=pos, spins=spins, atoms=batch_atoms, charges=batch_charges
    )
    t_init = 0
    opt_state_ckpt = None
    mcmc_width_ckpt = None
  if cfg.log.restore_path is not None:
     with np.load(cfg.log.restore_path,allow_pickle=True) as d:
         params = d["params"].tolist()
     t_init = 1
    #logging.info('No checkpoint found. Training new model.')
  for name, weights in params.items():
      if name == "layers":
         #print(name,weights['streams'])
         for da in weights['streams']:
             for k, value in da.items():
                 for inner_key, inner_value in value.items():
                      print(f"{k} - {inner_key}: {inner_value.shape}")
      if name == "orbital":
          for da in weights:
              for inner_key, inner_value in da.items():
                      print(f"{inner_key}: {inner_value.shape}")
  
      if name == "envelope":
          for da in weights:
              for inner_key, inner_value in da.items():
                      print(f"{inner_key}: {inner_value.shape}")
  # Set up logging
  train_schema = ['step', 'energy', 'ewmean', 'ewvar', 'pmove']
  t_init = 1
  # Initialisation done. We now want to have different PRNG streams on each
  # device. Shard the key over devices
  sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)

  # Main training

  # Construct MCMC step
  atoms_to_mcmc = atoms if cfg.mcmc.scale_by_nuclear_distance else None
  mcmc_step = mcmc.make_mcmc_step(
      batch_network,
      device_batch_size,
      steps=cfg.mcmc.steps,
      atoms=atoms_to_mcmc,
      blocks=cfg.mcmc.blocks,
  )
  # Construct loss and optimizer
  if cfg.system.make_local_energy_fn:
    local_energy_module, local_energy_fn = (
        cfg.system.make_local_energy_fn.rsplit('.', maxsplit=1))
    local_energy_module = importlib.import_module(local_energy_module)
    make_local_energy = getattr(local_energy_module, local_energy_fn)  # type: hamiltonian.MakeLocalEnergy
    local_energy = make_local_energy(
        f=signed_network,
        charges=charges,
        nspins=nspins,
        use_scan=False,
        **cfg.system.make_local_energy_kwargs)
  else:
    local_energy = hamiltonian.local_energy(
        f=signed_network,
        charges=charges,
        nspins=nspins,
        use_scan=False,
        complex_output=cfg.network.get('complex', False))
  evaluate_loss = qmc_loss_functions.make_loss(
      log_network if cfg.network.get('complex', False) else logabs_network,
      local_energy,
      clip_local_energy=cfg.optim.clip_local_energy,
      clip_from_median=cfg.optim.clip_median,
      center_at_clipped_energy=cfg.optim.center_at_clip,
      complex_output=cfg.network.get('complex', False)
  )
  
  # Compute the learning rate
  def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
    return cfg.optim.lr.rate * jnp.power(
        (1.0 / (1.0 + (t_/cfg.optim.lr.delay))), cfg.optim.lr.decay)

  # Construct and setup optimizer
  if cfg.optim.optimizer == 'none':
    optimizer = None
  elif cfg.optim.optimizer == 'adam':
    optimizer = optax.chain(
        optax.scale_by_adam(**cfg.optim.adam),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.))
  elif cfg.optim.optimizer == 'lamb':
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(eps=1e-7),
        optax.scale_by_trust_ratio(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1))
  elif cfg.optim.optimizer == 'kfac':
    # Differentiate wrt parameters (argument 0)
    val_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)
    optimizer = kfac_jax.Optimizer(
        val_and_grad,
        l2_reg=cfg.optim.kfac.l2_reg,
        norm_constraint=cfg.optim.kfac.norm_constraint,
        value_func_has_aux=True,
        value_func_has_rng=True,
        learning_rate_schedule=learning_rate_schedule,
        curvature_ema=cfg.optim.kfac.cov_ema_decay,
        inverse_update_period=cfg.optim.kfac.invert_every,
        min_damping=cfg.optim.kfac.min_damping,
        num_burnin_steps=0,
        register_only_generic=cfg.optim.kfac.register_only_generic,
        estimation_mode='fisher_exact',
        multi_device=True,
        pmap_axis_name=constants.PMAP_AXIS_NAME,
        auto_register_kwargs=dict(
            graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,
        ),
        # debug=True
    )

    wrams = kfac_jax.utils.replicate_all_local_devices(wrams)
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    opt_state = optimizer.init(params, subkeys, data, wrams)
    opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
  else:
    raise ValueError(f'Not a recognized optimizer: {cfg.optim.optimizer}')
  
  if not optimizer:
    opt_state = None
    step = make_training_step(
        mcmc_step=mcmc_step,
        optimizer_step=make_loss_step(evaluate_loss),
        evaluate_loss = evaluate_loss)    
  elif isinstance(optimizer, optax.GradientTransformation):
    # optax/optax-compatible optimizer (ADAM, LAMB, ...)
    opt_state = jax.pmap(optimizer.init)(params)
    opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
    step = make_training_step(
        mcmc_step=mcmc_step,
        optimizer_step=make_opt_update_step(evaluate_loss, optimizer))
  elif isinstance(optimizer, kfac_jax.Optimizer):
    step = make_kfac_training_step(
        mcmc_step=mcmc_step,
        damping=cfg.optim.kfac.damping,
        optimizer=optimizer)
  else:
    raise ValueError(f'Unknown optimizer: {optimizer}')
  if mcmc_width_ckpt is not None:
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(mcmc_width_ckpt[0])
  else:
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(
        jnp.asarray(cfg.mcmc.move_width))
  pmoves = np.zeros(cfg.mcmc.adapt_frequency)

  if t_init == 0:
    logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)

    burn_in_step = make_training_step(
        mcmc_step=mcmc_step, optimizer_step=null_update, evaluate_loss = evaluate_loss)

    for t in range(cfg.mcmc.burn_in):
      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
      data, params, *_ = burn_in_step(
          data,
          params,
          state=None,
          key=subkeys,
          mcmc_width=mcmc_width)
    logging.info('Completed burn-in MCMC steps')
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    ptotal_energy = constants.pmap(evaluate_loss)
    initial_energy, _ = ptotal_energy(params, subkeys, data)
    logging.info('Initial energy: %03.4f E_h', initial_energy[0])

  time_of_last_ckpt = time.time()
  weighted_stats = None

  if cfg.optim.optimizer == 'none' and opt_state_ckpt is not None:
    # If opt_state_ckpt is None, then we're restarting from a previous inference
    # run (most likely due to preemption) and so should continue from the last
    # iteration in the checkpoint. Otherwise, starting an inference run from a
    # training run.
    logging.info('No optimizer provided. Assuming inference run.')
    logging.info('Setting initial iteration to 0.')
    t_init = 0

  if writer_manager is None:
    writer_manager = writers.Writer(
        name='train_stats',
        schema=train_schema,
        directory="output",
        iteration_key=None,
   
        log=False)
  mcmcpositions = []
  with writer_manager as writer:
    # Main training loop
    ave_energy = []
    nw = 0

    for t in range(t_init, cfg.optim.iterations):
      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
                   
      wrams  = kfac_jax.utils.replicate_all_local_devices(wrams)

      data, params, opt_state, loss, unused_aux_data, pmove = step(
          data,
          params,
          opt_state,
          subkeys,
          mcmc_width,
          wrams,
          )
     
      nw = nw + 1
      pos_np = np.array(data.positions)
      mcmcpositions.append(pos_np)
      # due to pmean, loss, and pmove should be the same across
      # devices.
      loss = loss[0]
      # per batch variance isn't informative. Use weighted mean and variance
      # instead.
      if t == 100:
          weighted_stats = None
      weighted_stats = statistics.exponentialy_weighted_stats(
          alpha=0.1, observation=loss, previous_stats=weighted_stats)
      pmove = pmove[0]
       
      # Update MCMC move width
      if t > 0 and t % cfg.mcmc.adapt_frequency == 0:
        if np.mean(pmoves) > 0.55:
          mcmc_width *= 1.1
        if np.mean(pmoves) < 0.5:
          mcmc_width /= 1.1
        pmoves[:] = 0
      pmoves[t%cfg.mcmc.adapt_frequency] = pmove

      if cfg.debug.check_nan:
        tree = {'params': params, 'loss': loss}
        if cfg.optim.optimizer != 'none':
          tree['optim'] = opt_state
        chex.assert_tree_all_finite(tree)
      ave_energy.append(loss)
      # Logging
      if t % cfg.log.stats_frequency == 0:
        ave_energy.sort() # 
        num_to_remove = int(len(ave_energy) * 0.3) # 
        new_list = ave_energy[num_to_remove:-num_to_remove] # 
        mean_energy = sum(new_list) / len(new_list)
        #mean_energy = sum(ave_energy) / cfg.log.stats_frequency
        
        logging.info(
            'Step %05d: %03.4f E_h, exp. variance=%03.4f E_h^2, pmove=%0.2f', t,
            mean_energy, weighted_stats.mean, pmove)
        ave_energy = []
        writer.write(
            t,
            step=t,
            energy=np.asarray(loss),
            ewmean=np.asarray(weighted_stats.mean),
            ewvar=np.asarray(weighted_stats.variance),
            pmove=np.asarray(pmove))

  np_arr = np.array(mcmcpositions)
   
  #np.save('positions.npy', np_arr)
  
  np_arr = np.array(unused_aux_data.local_energy)
  #np.save('local_energy.npy', np_arr)
  return params,data,unused_aux_data
   
