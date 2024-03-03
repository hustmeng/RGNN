#Reference:

# [1] D. Pfau, J.S. Spencer, A.G.D.G. Matthews, W.M.C. Foulkes, Ab initio solution of the many-electron Schrödinger equation with deep neural networks, Phys. Rev. Res. 2(3) (2020) 033429.

#This code is extended from https://github.com/google-deepmind/ferminet.git, which has the Apache License, Version 2.0, January 2004.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.

"""Helper functions to create the loss and custom gradient of the loss."""

from typing import Tuple

import chex
from RGNN import constants
from RGNN import hamiltonian
from RGNN import networks
import jax
import jax.numpy as jnp
import kfac_jax
from typing_extensions import Protocol


@chex.dataclass
class AuxiliaryLossData:
  """Auxiliary data returned by total_energy.

  Attributes:
    variance: mean variance over batch, and over all devices if inside a pmap.
    local_energy: local energy for each MCMC configuration.
    clipped_energy: local energy after clipping has been applied
  """
  variance: jax.Array
  local_energy: jax.Array
  clipped_energy: jax.Array


class LossFn(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.FermiNetData,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched data elements to pass to the network.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """


def clip_local_values(
    local_values: jnp.ndarray,
    mean_local_values: jnp.ndarray,
    clip_scale: float,
    clip_from_median: bool,
    center_at_clipped_value: bool,
    complex_output: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Clips local operator estimates to remove outliers.

  Args:
    local_values: batch of local values,  Of/f, where f is the wavefunction and
      O is the operator of interest.
    mean_local_values: mean (over the global batch) of the local values.
    clip_scale: clip local quantities that are outside nD of the estimate of the
      expectation value of the operator, where n is this value and D the mean
      absolute deviation of the local quantities from the estimate of w, to the
      boundaries. The clipped local quantities should only be used to evaluate
      gradients.
    clip_from_median: If true, center the clipping window at the median rather
      than the mean. Potentially expensive in multi-host training, but more
      accurate/robust to outliers.
    center_at_clipped_value: If true, center the local energy differences passed
      back to the gradient around the clipped quantities, so the mean difference
      across the batch is guaranteed to be zero.
    complex_output: If true, the local energies will be complex valued.

  Returns:
    Tuple of the central value (estimate of the expectation value of the
    operator) and deviations from the central value for each element in the
    batch. If per_device_threshold is True, then the central value is per
    device.
  """

  batch_mean = lambda values: constants.pmean(jnp.mean(values))

  def clip_at_total_variation(values, center, scale):
    tv = batch_mean(jnp.abs(values- center))
    return jnp.clip(values, center - scale * tv, center + scale * tv)

  if clip_from_median:
    # More natural place to center the clipping, but expensive due to both
    # the median and all_gather (at least on multihost)
    clip_center = jnp.median(constants.all_gather(local_values).real)
  else:
    clip_center = mean_local_values
  # roughly, the total variation of the local energies
  if complex_output:
    clipped_local_values = (
        clip_at_total_variation(
            local_values.real, clip_center.real, clip_scale) +
        1.j * clip_at_total_variation(
            local_values.imag, clip_center.imag, clip_scale)
    )
  else:
    clipped_local_values = clip_at_total_variation(
        local_values, clip_center, clip_scale)
  if center_at_clipped_value:
    diff_center = batch_mean(clipped_local_values)
  else:
    diff_center = mean_local_values
  diff = clipped_local_values - diff_center
  return diff_center, diff



def make_loss(network: networks.LogFermiNetLike,
              local_energy: hamiltonian.LocalEnergy,
              clip_local_energy: float = 0.0,
              clip_from_median: bool = True,
              center_at_clipped_energy: bool = True,
              complex_output: bool = False) -> LossFn:
  """Creates the loss function, including custom gradients.

  Args:
    network: callable which evaluates the log of the magnitude of the
      wavefunction (square root of the log probability distribution) at a
      single MCMC configuration given the network parameters.
    local_energy: callable which evaluates the local energy.
    clip_local_energy: If greater than zero, clip local energies that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. The clipped local energies are only used to
      evaluate gradients.
    clip_from_median: If true, center the clipping window at the median rather
      than the mean. Potentially expensive in multi-host training, but more
      accurate.
    center_at_clipped_energy: If true, center the local energy differences
      passed back to the gradient around the clipped local energy, so the mean
      difference across the batch is guaranteed to be zero.
    complex_output: If true, the local energies will be complex valued.

  Returns:
    Callable with signature (params, data) and returns (loss, aux_data), where
    loss is the mean energy, and aux_data is an AuxiliaryLossDataobject. The
    loss is averaged over the batch and over all devices inside a pmap.
  """
  batch_local_energy = jax.vmap(
      local_energy,
      in_axes=(
          None,
          0,
          networks.FermiNetData(positions=0, spins=0, atoms=0, charges=0),
          None,
      ),
      out_axes=0,
  )
  batch_network = jax.vmap(network, in_axes=(None, 0, 0, 0, 0,  None,None), out_axes=0)

  @jax.custom_jvp
  def total_energy(
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.FermiNetData,
      wrams = None,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched MCMC configurations to pass to the local energy function.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """
    keys = jax.random.split(key, num=data.positions.shape[0])
    key, subkey = jax.random.split(key) 
    e_l = batch_local_energy(params, keys, data, wrams)
    loss = constants.pmean(jnp.mean(e_l))
     
    loss_diff = e_l - loss
    variance = constants.pmean(jnp.mean(loss_diff * jnp.conj(loss_diff)))
    return loss, AuxiliaryLossData(
        variance=variance.real, local_energy=e_l, clipped_energy=e_l)

  @total_energy.defjvp
  def total_energy_jvp(primals, tangents):  # pylint: disable=unused-variable
    """Custom Jacobian-vector product for unbiased local energy gradients."""
    params, key, data, wrams = primals
    #print(wrams[0])
    loss, aux_data = total_energy(params, key, data, wrams)
    key, subkey = jax.random.split(key)

    if clip_local_energy > 0.0:
      aux_data.clipped_energy, diff = clip_local_values(
          aux_data.local_energy,
          loss,
          clip_local_energy,
          clip_from_median,
          center_at_clipped_energy,
          complex_output)
    else:
      diff = aux_data.local_energy - loss

    # Due to the simultaneous requirements of KFAC (calling convention must be
    # (params, rng, data)) and Laplacian calculation (only want to take
    # Laplacian wrt electron positions) we need to change up the calling
    # convention between total_energy and batch_network
    #key = jax.numpy.zeros_like(key.shape, dtype=jax..float)
    data = primals[2]
    data_tangents = tangents[2]
    primals = (primals[0], data.positions, data.spins, data.atoms, data.charges, wrams, key.astype(float))
    tangents = (
        tangents[0],
        data_tangents.positions,
        data_tangents.spins,
        data_tangents.atoms,
        data_tangents.charges,
        wrams,
        key.astype(float),
    )
    psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)
    
    if complex_output:
      clipped_el = diff + aux_data.clipped_energy
      term1 = (jnp.dot(clipped_el, jnp.conjugate(psi_tangent)) +
               jnp.dot(jnp.conjugate(clipped_el), psi_tangent))
      term2 = jnp.sum(aux_data.clipped_energy*psi_tangent.real)
      kfac_jax.register_normal_predictive_distribution(
          psi_primal.real[:, None])
      primals_out = loss.real, aux_data
      device_batch_size = jnp.shape(aux_data.local_energy)[0]
      tangents_out = ((term1 - 2*term2).real / device_batch_size, aux_data)
    else:
      kfac_jax.register_normal_predictive_distribution(psi_primal[:, None])
      primals_out = loss, aux_data
      device_batch_size = jnp.shape(aux_data.local_energy)[0]
      tangents_out = (jnp.dot(psi_tangent, diff) / device_batch_size, aux_data)
    return primals_out, tangents_out

  return total_energy
