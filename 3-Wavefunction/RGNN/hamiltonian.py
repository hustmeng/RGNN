#Reference:

# [1] D. Pfau, J.S. Spencer, A.G.D.G. Matthews, W.M.C. Foulkes, Ab initio solution of the many-electron Schrödinger equation with deep neural networks, Phys. Rev. Res. 2(3) (2020) 033429.

#This code is extended from https://github.com/google-deepmind/ferminet.git, which has the Apache License, Version 2.0, January 2004.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.


"""Evaluating the Hamiltonian on a wavefunction."""

from typing import Any, Callable, Sequence, Union

import chex
from RGNN import networks
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from typing_extensions import Protocol
from RGNN import base_config

cfg = base_config.default()
Array = Union[jnp.ndarray, np.ndarray]


class LocalEnergy(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.FermiNetData,
  ) -> jnp.ndarray:
    """Returns the local energy of a Hamiltonian at a configuration.

    Args:
      params: network parameters.
      key: JAX PRNG state.
      data: MCMC configuration to evaluate.
    """


class MakeLocalEnergy(Protocol):

  def __call__(
      self,
      f: networks.FermiNetLike,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
      nspins: Sequence[int],
      use_scan: bool = False,
      complex_output: bool = False,
      **kwargs: Any
  ) -> LocalEnergy:
    """Builds the LocalEnergy function.

    Args:
      f: Callable which evaluates the sign and log of the magnitude of the
        wavefunction.
      atoms: atomic positions.
      charges: nuclear charges.
      nspins: Number of particles of each spin.
      use_scan: Whether to use a `lax.scan` for computing the laplacian.
      complex_output: If true, the output of f is complex-valued.
      **kwargs: additional kwargs to use for creating the specific Hamiltonian.
    """


KineticEnergy = Callable[
    [networks.ParamTree, networks.FermiNetData], jnp.ndarray
]


def select_output(f: Callable[..., Sequence[Any]],
                  argnum: int) -> Callable[..., Any]:
  """Return the argnum-th result from callable f."""

  def f_selected(*args, **kwargs):
    return f(*args, **kwargs)[argnum]

  return f_selected

def finite_diff_grad_logabs_f(logabs_f, params, x, spins, atoms, charges, wrams, key, h=0.03):
    """
    Compute the gradient of the function logabs_f with respect to the input vector x using the finite difference method.

    Args:
        logabs_f (function): The function for which the gradient is to be computed
        params (array): Parameters for the function logabs_f
        x (array): Input vector, with respect to which the gradient will be computed
        spins (array): Parameters for the function logabs_f
        atoms (array): Parameters for the function logabs_f
        charges (array): Parameters for the function logabs_f
        wrams (array): Parameters for the function logabs_f
        key (array): Parameters for the function logabs_f
        h (float, optional): Step size used in the finite difference method, default value is 0.03

    Returns:
        gradient (array): Gradient with respect to the input vector x
    """

    # Initialize the gradient vector with the same size as the input vector x
    gradient = jnp.zeros_like(x)

    # Iterate over each component of the input vector x
    for i in range(x.shape[0]):
        # Compute new input vectors based on the step size h
        x1 = x.at[i].set(x[i] + h)
        x2 = x.at[i].set(x[i] + 2*h)
        x3 = x.at[i].set(x[i] - h)
        x4 = x.at[i].set(x[i] - 2*h)

        # Compute the partial derivative using the finite difference method
        partial_derivative = (logabs_f(params, x1, spins, atoms, charges, wrams, key) -
                              logabs_f(params, x3, spins, atoms, charges, wrams, key)) / (2 * h)

        # Add the computed partial derivative to the gradient vector
        gradient = gradient.at[i].set(partial_derivative)

    # Return the gradient vector
    return gradient


def local_kinetic_energy(
    f: networks.FermiNetLike,
    use_scan: bool = False,
    complex_output: bool = False,
) -> KineticEnergy:
  r"""Creates a function to for the local kinetic energy, -1/2 \nabla^2 ln|f|.

  Args:
    f: Callable which evaluates the wavefunction as a
      (sign or phase, log magnitude) tuple.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    complex_output: If true, the output of f is complex-valued.

  Returns:
    Callable which evaluates the local kinetic energy,
    -1/2f \nabla^2 f = -1/2 (\nabla^2 log|f| + (\nabla log|f|)^2).
  """

  phase_f = select_output(f, 0)
  logabs_f = select_output(f, 1)

  def _lapl_over_f(params, data, wrams, key):
    n = data.positions.shape[0]
    eye = jnp.eye(n)
    grad_f = jax.grad(logabs_f, argnums=1)
    def grad_f_closure(x):
        return grad_f(params, x, data.spins, data.atoms, data.charges, wrams, key)
    def finite_diff_grad_f_closure(x):
        return finite_diff_grad_logabs_f(logabs_f, params, x, data.spins, data.atoms, data.charges, wrams, key)  
    
    cfg = base_config.default()
    if cfg.finite_diff:
       primal, dgrad_f = jax.linearize(finite_diff_grad_f_closure, data.positions)
    else:
       primal, dgrad_f = jax.linearize(grad_f_closure, data.positions)
    
    if complex_output:
      #grad_phase = jax.grad(phase_f, argnums=1)
      def grad_phase_closure(x):
        return grad_phase(params, x, data.spins, data.atoms, data.charges, wrams, key)
      phase_primal, dgrad_phase = jax.linearize(
          grad_phase_closure, data.positions)
      hessian_diagonal = (
          lambda i: dgrad_f(eye[i])[i] + 1.j * dgrad_phase(eye[i])[i]
      )
    else:
      hessian_diagonal = lambda i: dgrad_f(eye[i])[i]

    if use_scan:
      _, diagonal = lax.scan(
          lambda i, _: (i + 1, hessian_diagonal(i)), 0, None, length=n)
      result = -0.5 * jnp.sum(diagonal)
    else:
      result = -0.5 * lax.fori_loop(
          0, n, lambda i, val: val + hessian_diagonal(i), 0.0)
    result -= 0.5 * jnp.sum(primal ** 2)
    if complex_output:
      result += 0.5 * jnp.sum(phase_primal ** 2)
      result -= 1.j * jnp.sum(primal * phase_primal)
    return result

  return _lapl_over_f


def potential_electron_electron(r_ee: Array) -> jnp.ndarray:
  """Returns the electron-electron potential.

  Args:
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j. Other elements in the final axes are not
      required.
  """
  return jnp.sum(jnp.triu(1 / r_ee[..., 0], k=1))


def potential_electron_nuclear(charges: Array, r_ae: Array) -> jnp.ndarray:
  """Returns the electron-nuclearpotential.

  Args:
    charges: Shape (natoms). Nuclear charges of the atoms.
    r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
      electron i and atom j.
  """
  return -jnp.sum(charges / r_ae[..., 0])


def potential_nuclear_nuclear(charges: Array, atoms: Array) -> jnp.ndarray:
  """Returns the electron-nuclearpotential.

  Args:
    charges: Shape (natoms). Nuclear charges of the atoms.
    atoms: Shape (natoms, ndim). Positions of the atoms.
  """
  r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
  return jnp.sum(
      jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))


def potential_energy(r_ae: Array, r_ee: Array, atoms: Array,
                     charges: Array) -> jnp.ndarray:
  """Returns the potential energy for this electron configuration.

  Args:
    r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
      electron i and atom j.
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j. Other elements in the final axes are not
      required.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
  """
  return (potential_electron_electron(r_ee) +
          potential_electron_nuclear(charges, r_ae) +
          potential_nuclear_nuclear(charges, atoms))


def local_energy(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    complex_output: bool = False,
) -> LocalEnergy:
  """Creates the function to evaluate the local energy.

  Args:
    f: Callable which returns the sign and log of the magnitude of the
      wavefunction given the network parameters and configurations data.
    charges: Shape (natoms). Nuclear charges of the atoms.
    nspins: Number of particles of each spin.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    complex_output: If true, the output of f is complex-valued.

  Returns:
    Callable with signature e_l(params, key, data) which evaluates the local
    energy of the wavefunction given the parameters params, RNG state key,
    and a single MCMC configuration in data.
  """
  del nspins
  ke = local_kinetic_energy(f,
                            use_scan=use_scan,
                            complex_output=complex_output)

  def _e_l(
      params: networks.ParamTree, key: chex.PRNGKey, data: networks.FermiNetData, wrams
  ) -> jnp.ndarray:
    """Returns the total energy.

    Args:
      params: network parameters.
      key: RNG state.
      data: MCMC configuration.
    """
    
    key = jax.random.PRNGKey(key[0])
    key, subkey = jax.random.split(key)
    _, _, r_ae, r_ee = networks.construct_input_features(
        data.positions, data.atoms)
    potential = potential_energy(r_ae, r_ee, data.atoms, charges)
    kinetic = ke(params, data, wrams, key)
     
    return potential + kinetic

  return _e_l
