import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import ase
from ase.db import connect
from ase.neighborlist import neighbor_list

from data.keys import Keys

def _get_nbr_info(atoms: ase.Atoms, cutoff: float):
    """
    Helper function to obtain neighbors information from ase.Atoms object.

    At  :  Total number of atoms in the batch
    Nbr :  Total number of neighbors of each atom

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object of molecule.
    cutoff : float
        cutoff radius.

    Returns
    -------
    nbr_idx : np.ndarray
        neighborhood atoms indices. (At x Nbr) of shape.
    nbr_mask : np.ndrray
        array with a value of 0 if there are no neighboring atoms, and 1 if there are.
        (At x Nbr) of shape.
    distances : np.ndarray
        inter atomic distances of neighborhood atoms. (At x Nbr) of shape.
    unit_vec : np.ndarray
        unit vectors in the direction from the central atom to the neighboring atoms.
        (At x Nbr x 3) of shape.
    offset : np.ndarray
        cell offset values of neighboring atoms. (At x Nbr x 3)
"""
    # get neighbor info form ase interface
    n_atoms = atoms.get_global_number_of_atoms()
    idx_i, idx_j, dist, dist_vecs, idx_S = neighbor_list(
        "ijdDS", atoms, cutoff=cutoff, self_interaction=False
    )
    if idx_i.shape[0] > 0:
        uidx, n_nbh = np.unique(idx_i, return_counts=True)
        n_max_nbh = np.max(n_nbh)

        n_nbh = np.tile(n_nbh[:, np.newaxis], (1, n_max_nbh))
        nbh_range = np.tile(
            np.arange(n_max_nbh, dtype=np.int32)[np.newaxis], (n_nbh.shape[0], 1)
        )
        mask = np.zeros((n_atoms, n_max_nbh), dtype=np.bool_)
        mask[uidx, :] = nbh_range < n_nbh
        nbr_idx = -np.ones((n_atoms, n_max_nbh), dtype=np.float32)
        nbr_idx[mask] = idx_j
        # reshape idx_S to offset value with (At x Nbr x 3) shape.
        offset = np.zeros((n_atoms, n_max_nbh, 3), dtype=np.float32)
        offset[mask] = idx_S
        # reshape distances to (At x Nbr) of shape
        distances = np.zeros((n_atoms, n_max_nbh), dtype=np.float32)
        distances[mask] = dist
        # reshape dist_vecs to (At x Nbr x 3) of shape, and normalize.
        unit_vecs = np.zeros((n_atoms, n_max_nbh, 3), dtype=np.float32)
        unit_vecs[mask] = dist_vecs
        tmp_dist = distances[mask, np.newaxis]
        unit_vecs[mask] = unit_vecs[mask] / tmp_dist

    else:
        nbr_idx = np.zeros((n_atoms, 1), dtype=np.int32)
        distances = np.zeros((n_atoms, 1), dtype=np.float32)
        unit_vecs = np.zeros((n_atoms, 1, 3), dtype=np.float32)
        offset = np.zeros((n_atoms, 1, 3), dtype=np.float32)

    return nbr_idx, distances, unit_vecs, offset

def _convert_atoms(atoms: ase.Atoms, cutoff: float, output: dict = None) -> dict:
    """
    Helper function to convert ASE atoms object to GNNFF inputs format.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object of molecule.
    cutoff : flaot
        cutoff radius.
    output : dict, default=None
        Destination for converted atoms, if not None.

    Returns
    -------
    outputs : dict of torch.Tensor
        Properties including neighbor lists and masks reformated into GNNFF input format.

    References
    ----------
    .. [1] https://github.com/ken2403/schnetpack/blob/6617dbf4edd1fc4d4aae0c984bc7a747a4fe9c0c/src/schnetpack/data/atoms.py
    """
    if output is None:
        outputs = {}
    else:
        outputs = output

    # Elemental composition
    outputs[Keys.Z] = atoms.numbers.astype(np.int32)
    # positions = atoms.positions.astype(np.float32)
    # positions -= _get_center_of_gravity(atoms)
    # outputs[Keys.R] = positions
    outputs[Keys.n_atoms] = np.array([atoms.get_global_number_of_atoms()]).astype(
        np.int32
    )

    # get atom neighbors
    nbh_idx, distances, unit_vecs, offsets = _get_nbr_info(atoms, cutoff=cutoff)

    # Get neighbors
    outputs[Keys.neighbors] = nbh_idx.astype(np.int32)

    # Get cells
    # outputs[Keys.cell] = np.array(atoms.cell.array, dtype=np.float32)
    outputs[Keys.cell_offset] = offsets.astype(np.float32)

    # Get distances
    outputs[Keys.distances] = distances.astype(np.float32)
    outputs[Keys.unit_vecs] = unit_vecs.astype(np.float32)

    return outputs


class DatagraphError(Exception):
    pass

class Datagraph:      
      def __init__(
        self, db_path: str, cutoff: float, available_properties: list = None
    ) -> None:
        # checks
        if not db_path.endswith(".db"):
            raise DatagraphError(
                "Invalid dbpath! Please make sure to add the file extension '.db' to your dbpath."
            )
        self.db_path = db_path
        self.cutoff = cutoff
        self.available_properties = self._get_available_properties(
            available_properties
        )      

      def __call__(self):

          data_list = []          
          with connect(self.db_path) as conn:
               conn_length = conn.count()
          for idx in range(conn_length):
              _, properties = self.get_properties(idx=idx,available_properties = self.available_properties)
              
              properties["_idx"] = np.array([idx], dtype=np.int32)
              data_list.append(properties)
              #if idx%1000==0:
              #   print(idx)
          return data_list
          
      def _get_available_properties(self, properties):
          """
          Get available properties from argument or database.
          Returns
          -------
          available_properties : list
              all properties of the dataset
          """
          # read database properties
          if os.path.exists(self.db_path):# and len(self) != 0:
              with connect(self.db_path) as conn:
                db_properties = ['forces']
                #atmsrw = conn.get(1)
                #db_properties = list(atmsrw.data.keys())
          else:
            db_properties = None

          # use the provided list
          if properties is not None:
              if db_properties is None or set(db_properties) == set(properties):
                return properties

            # raise error if available properties do not match database
              raise DatagraphError(
                "The available_properties {} do not match the "
                "properties in the database {}!".format(properties, db_properties)
              )

          # return database properties
          if db_properties is not None:
              return db_properties

          raise DatagraphError(
              "Please define available_properties or set db_path to an existing database!"
          )

      def get_properties(self,idx: int,available_properties):        
          """
          Return atoms object and properties dictionary at given index.

          Parameters
          ----------
          idx : int
              a data index.

          Returns
          -------
          at : ase.Atoms
              atoms object.
          properties : dict
              dictionary with molecular properties.

          References
          ----------
          .. [1] https://github.com/ken2403/schnetpack/blob/6617dbf4edd1fc4d4aae0c984bc7a747a4fe9c0c/src/schnetpack/data/atoms.py
          """
          # read from ase-database
          with connect(self.db_path) as conn:
             row = conn.get(idx + 1)
          at = row.toatoms()

          # extract properties
          properties = {}
          for pname in available_properties:
              properties[pname] = row[pname]
          
          # extract/calculate structure
          properties = _convert_atoms(at, cutoff=self.cutoff, output=properties)

          return at, properties

