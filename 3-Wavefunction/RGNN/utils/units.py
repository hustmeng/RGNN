#Reference:

# [1] D. Pfau, J.S. Spencer, A.G.D.G. Matthews, W.M.C. Foulkes, Ab initio solution of the many-electron Schrödinger equation with deep neural networks, Phys. Rev. Res. 2(3) (2020) 033429.

#This code is extended from https://github.com/google-deepmind/ferminet.git, which has the Apache License, Version 2.0, January 2004.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.


"""Basic definition of units and converters useful for chemistry."""

from typing import TypeVar
import numpy as np

# 1 Bohr = 0.52917721067 (12) x 10^{-10} m
# https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
# Note: pyscf uses a slightly older definition of 0.52917721092 angstrom.
ANGSTROM_BOHR = 0.52917721067
BOHR_ANGSTROM = 1. / ANGSTROM_BOHR

# 1 Hartree = 627.509474 kcal/mol
# https://en.wikipedia.org/wiki/Hartree
KCAL_HARTREE = 627.509474
HARTREE_KCAL = 1. / KCAL_HARTREE

NumericalLike = TypeVar('NumericalLike', float, np.ndarray)


def bohr2angstrom(x_b: NumericalLike) -> NumericalLike:
  return x_b * ANGSTROM_BOHR


def angstrom2bohr(x_a: NumericalLike) -> NumericalLike:
  return x_a * BOHR_ANGSTROM


def hartree2kcal(x_b: NumericalLike) -> NumericalLike:
  return x_b * KCAL_HARTREE


def kcal2hartree(x_a: NumericalLike) -> NumericalLike:
  return x_a * HARTREE_KCAL
