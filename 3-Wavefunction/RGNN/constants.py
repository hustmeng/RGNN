#Reference:
  
# D. Pfau, J.S. Spencer, A.G.D.G. Matthews, W.M.C. Foulkes, Ab initio solution of the many-electron Schrödinger equation with deep neural networks, Phys. Rev. Res. 2(3) (2020) 033429.
# https://github.com/google-deepmind/ferminet.git

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.


"""Constants for FermiNet."""

import functools
import jax
import kfac_jax


# Axis name we pmap over.
PMAP_AXIS_NAME = 'qmc_pmap_axis'

# Shortcut for jax.pmap over PMAP_AXIS_NAME. Prefer this if pmapping any
# function which does communications or reductions.
pmap = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)

# Shortcut for kfac utils
psum = functools.partial(kfac_jax.utils.psum_if_pmap, axis_name=PMAP_AXIS_NAME)
pmean = functools.partial(
    kfac_jax.utils.pmean_if_pmap, axis_name=PMAP_AXIS_NAME)
all_gather = functools.partial(kfac_jax.utils.wrap_if_pmap(jax.lax.all_gather),
                               axis_name=PMAP_AXIS_NAME)
