#Reference:

# [1] D. Pfau, J.S. Spencer, A.G.D.G. Matthews, W.M.C. Foulkes, Ab initio solution of the many-electron Schrödinger equation with deep neural networks, Phys. Rev. Res. 2(3) (2020) 033429.

#This code is extended from https://github.com/google-deepmind/ferminet.git, which has the Apache License, Version 2.0, January 2004.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.


"""Functions to create different kinds of systems."""

from typing import Sequence
import attr
from RGNN.utils import elements
from RGNN.utils import units as unit_conversion
import ml_collections
import numpy as np
import pyscf


@attr.s
class Atom:
  """Atom information for Hamiltonians.

  The nuclear charge is inferred from the symbol if not given, in which case the
  symbol must be the IUPAC symbol of the desired element.

  Attributes:
    symbol: Element symbol.
    coords: An iterable of atomic coordinates. Always a list of floats and in
      bohr after initialisation. Default: place atom at origin.
    charge: Nuclear charge. Default: nuclear charge (atomic number) of atom of
      the given name.
    atomic_number: Atomic number associated with element. Default: atomic number
      of element of the given symbol. Should match charge unless fractional
      nuclear charges are being used.
    units: String giving units of coords. Either bohr or angstrom. Default:
      bohr. If angstrom, coords are converted to be in bohr and units to the
      string 'bohr'.
    coords_angstrom: list of atomic coordinates in angstrom.
    coords_array: Numpy array of atomic coordinates in bohr.
    element: elements.Element corresponding to the symbol.
  """
  symbol = attr.ib(type=str)
  coords = attr.ib(
      type=Sequence[float],
      converter=lambda xs: tuple(float(x) for x in xs),
      default=(0.0, 0.0, 0.0))
  charge = attr.ib(type=float, converter=float)
  atomic_number = attr.ib(type=int, converter=int)
  units = attr.ib(
      type=str,
      default='bohr',
      validator=attr.validators.in_(['bohr', 'angstrom']))

  @charge.default
  def _set_default_charge(self):
    return self.element.atomic_number

  @atomic_number.default
  def _set_default_atomic_number(self):
    return self.element.atomic_number

  def __attrs_post_init__(self):
    if self.units == 'angstrom':
      self.coords = [unit_conversion.angstrom2bohr(x) for x in self.coords]
      self.units = 'bohr'

  @property
  def coords_angstrom(self):
    return [unit_conversion.bohr2angstrom(x) for x in self.coords]

  @property
  def coords_array(self):
    if not hasattr(self, '_coords_arr'):
      self._coords_arr = np.array(self.coords)
    return self._coords_arr

  @property
  def element(self):
    return elements.SYMBOLS[self.symbol]


def pyscf_mol_to_internal_representation(
    mol: pyscf.gto.Mole) -> ml_collections.ConfigDict:
  """Converts a PySCF Mole object to an internal representation.

  Args:
    mol: Mole object describing the system of interest.

  Returns:
    A ConfigDict with the fields required to describe the system set.
  """
  # Ensure Mole is built so all attributes are appropriately set.
  mol.build()
  atoms = [
      Atom(mol.atom_symbol(i), mol.atom_coord(i))
      for i in range(mol.natm)
  ]
  return ml_collections.ConfigDict({
      'system': {
          'molecule': atoms,
          'electrons': mol.nelec,
      },
      'pretrain': {
          # If mol.basis isn't a string, assume that mol is passed into
          # pretraining as well and pretraining uses the basis already set in
          # mol, rather than complicating the configuration here.
          'basis': mol.basis if isinstance(mol.basis, str) else None,
      },
  })
