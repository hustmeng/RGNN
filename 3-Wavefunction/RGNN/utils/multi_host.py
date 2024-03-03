#Reference:

# [1] D. Pfau, J.S. Spencer, A.G.D.G. Matthews, W.M.C. Foulkes, Ab initio solution of the many-electron Schrödinger equation with deep neural networks, Phys. Rev. Res. 2(3) (2020) 033429.

#This code is extended from https://github.com/google-deepmind/ferminet.git, which has the Apache License, Version 2.0, January 2004.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.


"""Generic utilities."""

from absl import logging
import jax
import jax.numpy as jnp


def check_synced(obj, name):
  """Checks whether the object is synced across local devices.

  Args:
    obj: PyTree with leaf nodes mapped over local devices.
    name: the name of the object (for logging only).

  Returns:
    True if object is in sync across all devices and False otherwise.
  """
  for i in range(1, jax.local_device_count()):
    norms = jax.tree_map(lambda x: jnp.linalg.norm(x[0] - x[i]), obj)  # pylint: disable=cell-var-from-loop
    total_norms = sum(jax.tree_leaves(norms))
    if total_norms != 0.0:
      logging.info(
          '%s object is not synced across device 0 and %d. The total norm'
          ' of the difference is %.5e. For specific detail inspect '
          'the individual differences norms:\n %s.',
          name, i, total_norms, str(norms)
      )
      return False
  logging.info('%s objects are synced.', name)
  return True
