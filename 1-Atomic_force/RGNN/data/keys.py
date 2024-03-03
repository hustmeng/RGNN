#Reference:

#1. C.W. Park, M. Kornbluth, J. Vandermause, C. Wolverton, B. Kozinsky, J.P. Mailoa, Accurate and scalable graph neural network force field and molecular dynamics with direct force architecture, npj Comput. Mater. 7(1) (2021) 73. 
#This code is extended from https://github.com/ken2403/gnnff.git, which has the MIT License.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.


__all__ = ["Keys"]


class Keys:
    """
    Keys to access structure properties in `CrystalData` object.

    References
    ----------
    .. [1] https://github.com/atomistic-machine-learning/schnetpack/blob/67226795af/src/schnetpack/__init__.py
    """

    # geometry
    Z = "_atomic_numbers"
    charge = "_charge"
    atom_mask = "_atom_mask"
    R = "_positions"
    cell = "_cell"
    pbc = "_pbc"
    neighbors = "_neighbors"
    neighbor_mask = "_neighbor_mask"
    cell_offset = "_cell_offset"
    distances = "_distances"
    unit_vecs = "_unit_vecs"
    n_atoms = "_n_atoms"

    # chemical properties
    energy = "energy"
    forces = "forces"
