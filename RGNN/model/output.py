#Reference:

#1. C.W. Park, M. Kornbluth, J. Vandermause, C. Wolverton, B. Kozinsky, J.P. Mailoa, Accurate and scalable graph neural network force field and molecular dynamics with direct force architecture, npj Comput. Mater. 7(1) (2021) 73. 
#This code is extended from https://github.com/ken2403/gnnff.git, which has the MIT License.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.


from torch import Tensor
import torch.nn as nn

from RGNN.model.functional import shifted_softplus
from RGNN.model.linears import Linear


__all__ = ["OutputModuleError", "ForceMapping",]


class OutputModuleError(Exception):
    pass


class ForceMapping(nn.Module):
    """
    From edge embedding tensor, calculating the force magnitude of all inter atomic forces.
    And then, calculate the inter atomic forces by multiplying unit vectors.

    Attributes
    ----------
    n_edge_feature : int
        dimension of the embedded edge features.
    n_layers : int, default=2
        number of output layers.
    activation : collable or None, default=gnnff.nn.activation.shifted_softplus
        activation function. All hidden layers would the same activation function
        except the output layer that does not apply any activation function.
    """

    def __init__(
        self,
        n_edge_feature: int,
        n_layers: int = 2,
        activation=shifted_softplus,
    ) -> None:
        super().__init__()
        n_neurons_list = []
        c_neurons = n_edge_feature 
        
        for ii in range(n_layers):
            n_neurons_list.append(c_neurons)
            if ii < 0:
              c_neurons = max(1, c_neurons)
            else:
              c_neurons = max(1, c_neurons // 2)
        # The output layer has 1 neurons.
        n_neurons_list.append(1)
        layers = [
            Linear(n_neurons_list[i], n_neurons_list[i + 1], activation=activation)
            for i in range(n_layers - 1)
        ]

        layers.append(Linear(n_neurons_list[-2], n_neurons_list[-1], activation=None))
        self.out_net = nn.Sequential(*layers)

    def forward(self, last_edge_embedding: Tensor, unit_vecs: Tensor) -> Tensor:
        """
        Calculates the inter atomic forces.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr :  Total number of neighbors of each atom

        Parameters
        ----------
        last_edge_embedding : torch.Tensor
            calculated edge embedding tensor of (B x At x Nbr x n_edge_features) shape.
        unit_vecs : torch.Tensor
            unit vecs of each edge.

        Returns
        -------
        predicted_forces : torch.Tensor
            predicting inter atomic forces for each atoms. (B x At x 3) shape.
        """
        # calculate force_magnitude from last edge_embedding
        force_magnitude = self.out_net(last_edge_embedding)
        force_magnitude = force_magnitude.expand(-1, -1, -1, 3)
        # predict inter atpmic forces by multiplying the unit vectors
        preditcted_forces = force_magnitude * unit_vecs
        # summation of all neighbors effection
        preditcted_forces = preditcted_forces.sum(dim=2)
        return preditcted_forces

