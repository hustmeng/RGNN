#Reference:

#1. C.W. Park, M. Kornbluth, J. Vandermause, C. Wolverton, B. Kozinsky, J.P. Mailoa, Accurate and scalable graph neural network force field and molecular dynamics with direct force architecture, npj Comput. Mater. 7(1) (2021) 73. 
#This code is extended from https://github.com/ken2403/gnnff.git, which has the MIT License.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.

import os

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import ase
from ase.db import connect
from ase.visualize import view
from ase.neighborlist import neighbor_list

from RGNN.data.keys import Keys


__all__ = ["CellData"]


class CellDataError(Exception):
    pass


class CellData(Dataset):
    """
    From the unit cell database, compose the dataset that can be used for inputs of GNNFF.

    Attributes
    ----------
    db_path : str
        path to directory containing database.
    cutoff : float
        cutoff radius.
    available_properties : list, default=None
        complete set of physical properties that are contained in the database.
    """

    def __init__(
        self, db_path: str, data_list: list, cutoff: float, available_properties: list = None
    ) -> None:
        # checks
        if not db_path.endswith(".db"):
            raise CellDataError(
                "Invalid dbpath! Please make sure to add the file extension '.db' to your dbpath."
            )
        self.db_path = db_path
        self.data_list = data_list


    def get_metadata(self, key=None):
        """
        Returns an entry from the metadata dictionary of the ASE db.

        Parameters
        ----------
        key : str
            Name of metadata entry. Return full dict if `None`.

        Returns
        -------
        value
            Value of metadata entry or full metadata dict, if key is `None`.
        """
        with connect(self.dbpath) as conn:
            if key is None:
                return conn.metadata
            if key in conn.metadata.keys():
                return conn.metadata[key]
        return None

    # Dataset function
    def __getitem__(self, idx: int):
        #idx = idx - 1
        properties = self.data_list[idx] 
        return torchify_dict(properties)

    def __len__(self):
        with connect(self.db_path) as conn:
            return conn.count()

    def __add__(self, other: Dataset) -> ConcatDataset:
        return super().__add__(other)


def torchify_dict(data: dict):
    """
    Transform np.ndarrays to torch.tensors.

    Parameters
    ----------
    data : dict
        property data of np.ndarrays.

    References
    ----------
    .. [1] https://github.com/ken2403/schnetpack/blob/6617dbf4edd1fc4d4aae0c984bc7a747a4fe9c0c/src/schnetpack/data/atoms.py
    """
    torch_properties = {}
    for pname, prop in data.items():
        if prop.dtype in [np.int32, np.int64]:
            torch_properties[pname] = torch.LongTensor(prop)
        elif prop.dtype in [np.float32, np.float64]:
            torch_properties[pname] = torch.FloatTensor(prop.copy())
        else:
            raise CellDataError(
                "Invalid datatype {} for property {}!".format(type(prop), pname)
            )
    return torch_properties


