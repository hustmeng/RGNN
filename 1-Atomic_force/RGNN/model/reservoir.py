#Reference:

#1. C.W. Park, M. Kornbluth, J. Vandermause, C. Wolverton, B. Kozinsky, J.P. Mailoa, Accurate and scalable graph neural network force field and molecular dynamics with direct force architecture, npj Comput. Mater. 7(1) (2021) 73. 
#This code is extended from https://github.com/ken2403/gnnff.git, which has the MIT License.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.


import torch
from torch import Tensor
import torch.nn as nn
import random
from RGNN.model.linears import Linear, reservoir_Linear
from RGNN.model.neighbors import GetNodeK, GetEdgeJK

__all__ = [
    "NodeUpdate",
    "EdgeUpdate",
    "MessagePassing",
]

def quantization(fc, embedding, conductance=None, nbit=8):
    """
    Perform multibit vector-matrix multiplication to transform an analog input vector into an m-bit binary vector.

    Args:
        fc (torch.nn.Module): Fully connected layer used for vector-matrix multiplication
        embedding (torch.Tensor): Input tensor to be transformed into an m-bit binary vector
        conductance (torch.Tensor, optional): Conductance tensor, used to simulate conductance fluctuation of resistive memory
        nbit (int, optional): Number of bits for the binary vector representation, default value is 8

    Returns:
        z (torch.Tensor): Result of the multibit vector-matrix multiplication
        min_val (float): Minimum value of the input tensor 'embedding'
        max_val (float): Maximum value of the input tensor 'embedding'
    """

    # Compute the quantization base using the number of bits
    quant_base = 2 ** torch.arange(nbit).to(embedding.device)
    nsplit = 2 ** nbit - 1

    # Normalize the input tensor 'embedding' and convert it to an unsigned 8-bit integer tensor
    min_val = torch.min(embedding)
    max_val = torch.max(embedding)
    new_embedding = ((embedding - min_val) / (max_val - min_val) * nsplit).type(torch.uint8)

    # Convert the input tensor 'new_embedding' into a binary tensor
    binary = new_embedding.unsqueeze(-1).bitwise_and(quant_base).ne(0).byte()
    binary = binary.to(torch.float)

    # Initialize the output tensor 'z' to zero
    z = 0

    # Perform multibit vector-matrix multiplication
    for i in range(nbit):
        w = conductance[0]

        # Simulate the conductance fluctuation of the resistive memory
        noise = torch.randn_like(w) * w.abs() * 0.1

        # Compute the input tensor 'x' for the current bit 'i'
        if len(embedding.shape) == 4:
            x = (binary[:, :, :, :, i] * (2 ** i)).to(torch.float)
        else:
            x = (binary[:, :, :, :, :, i] * (2 ** i)).to(torch.float)

        # Update the weight of the fully connected layer 'fc' with the simulated noise
        fc.weight = torch.nn.Parameter(conductance[0] + noise, requires_grad=False)

        # Perform the vector-matrix multiplication for the current bit 'i'
        x = fc(x)

        # Accumulate the result in the output tensor 'z'
        z = z + x

    return z, min_val, max_val


def re_quantization(embedding, min_val, max_val, weight, nbit=8):
    """
    Revert the quantized tensor 'embedding' back to its original scale.

    Args:
        embedding (torch.Tensor): Quantized tensor
        min_val (float): Minimum value of the original tensor
        max_val (float): Maximum value of the original tensor
        weight (torch.Tensor): Weight tensor used in the quantization process
        nbit (int, optional): Number of bits for the binary vector representation, default value is 8

    Returns:
        embedding (torch.Tensor): Tensor reverted back to its original scale
    """

    # Compute the number of splits based on the number of bits
    nsplit = 2 ** nbit - 1

    # Compute the bias term based on the minimum value and weight tensor
    bias = torch.sum(min_val * weight, dim=1)

    # Revert the quantized tensor 'embedding' back to its original scale
    embedding = embedding / nsplit * (max_val - min_val) + bias

    return embedding

class NodeUpdate(nn.Module):
    """
    Updated the node embedding tensor from the previous node and edge embedding.

    Attributes
    ----------
    n_node_feature : int
        dimension of the embedded node features.
    n_edge_feature : int
        dimension of the embedded edge features.
    """

    def __init__(
        self,
        n_node_feature: int,
        n_edge_feature: int,
    ) -> None:
        super().__init__()
        self.fc = reservoir_Linear(
            n_node_feature + n_edge_feature, 4*n_node_feature + 4*n_edge_feature, activation=None, 
        )
        self.fc_2 = reservoir_Linear(
            4*n_node_feature + 4*n_edge_feature, 2 * n_node_feature, activation=None,
        )
        self.fc.weight.requires_grad=False
        #self.fc.bias.requires_grad=False
        self.fc_2.weight.requires_grad=False
        #self.fc_2.bias.requires_grad=False
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.bn = nn.BatchNorm1d(n_node_feature)
        self.relu = nn.ReLU()

    def forward(
        self,
        node_embedding: Tensor,
        edge_embedding: Tensor,
        nbr_mask: Tensor,
        conductance
    ) -> Tensor:
        """
        Update the node embedding.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr :  Total number of neighbors of each atom

        Parameters
        ----------
        node_embedding : torch.Tensor
            batch of node embedding tensor of (B x At x n_node_feature) shape.
        edge_embedding : torch.Tensor
            batch of edge embedding tensor of (B x At x Nbr x n_edge_feuture) shape.
        nbr_mask : torch.Tensor
            boolean mask for neighbor positions.(B x At x Nbr) of shape.

        Returns
        -------
        node_embedding : torch.Tensor
            updated node embedding tensor of (B x At x n_node_feature) shape.

        References
        ----------
        .. [1] https://github.com/ken2403/cgcnn/blob/master/cgcnn/model.py
        """
        B, At, Nbr, _ = edge_embedding.size()
        _, _, n_node_feature = node_embedding.size()

        # make c1-ij tensor
        c1 = torch.cat(
            [
                node_embedding.unsqueeze(2).expand(B, At, Nbr, n_node_feature),
                edge_embedding,
            ],
            dim=3,
        )
        # fully connected layter
        c1, min_val, max_val = quantization(self.fc,c1,conductance[0])
        c1 = re_quantization(c1, min_val, max_val, self.fc.weight)
        c1 = self.relu(c1)

        c1, min_val, max_val = quantization(self.fc_2,c1,conductance[1])
        c1 = re_quantization(c1, min_val, max_val,conductance[1][-1])

        # calculate the gate and extract features
        nbr_gate, nbr_extract = c1.chunk(2, dim=3)
        nbr_gate = self.sigmoid(nbr_gate)
        nbr_extract = self.tanh(nbr_extract)
        # elemet-wise multiplication with gate
        nbr_sumed = nbr_gate * nbr_extract
        # apply neighbor mask, if there are no neighbor, padding with 0
        nbr_sumed = nbr_sumed * nbr_mask[..., None]

        nbr_sumed = torch.sum(nbr_sumed, dim=2)
        nbr_sumed = self.bn(nbr_sumed.view(-1, n_node_feature)).view(
            B, At, n_node_feature
        )
        # last activation layer and Residual Network
        node_embedding = self.relu(node_embedding + nbr_sumed)
        return node_embedding


class EdgeUpdate(nn.Module):
    """
    Updated the edge embedding tensor from the new node embedding
    and the previous edge embedding.

    Attributes
    ----------
    n_node_feature : int
        dimension of the embedded node features.
    n_edge_feature : int
        dimension of the embedded edge features.
    """

    def __init__(
        self,
        n_node_feature: int,
        n_edge_feature: int,
    ) -> None:
        super().__init__()
        self.fc_two_body = reservoir_Linear(2 * n_node_feature , 8 * n_node_feature, activation=None,)
        self.fc_two_body_2 = reservoir_Linear(8 * n_node_feature , 2 * n_edge_feature, activation=None,)
        self.bn_two_body = nn.BatchNorm1d(n_edge_feature)
        
        self.fc_two_body.weight.requires_grad=False
        #self.fc_two_body.bias.requires_grad=False
        self.fc_two_body_2.weight.requires_grad=False
        #self.fc_two_body_2.bias.requires_grad=False
        
        self.get_node_k = GetNodeK()
        self.get_edge_jk = GetEdgeJK()
        self.fc_three_body = reservoir_Linear(
            3 * n_node_feature + 2 * n_edge_feature,
            6 * n_node_feature + 4 * n_edge_feature,
            activation=None,
        )

        self.fc_three_body_2 = reservoir_Linear(
            6 * n_node_feature + 4 * n_edge_feature,
            2 * n_edge_feature,
            activation=None,
        )

        self.fc_three_body.weight.requires_grad=False
        #self.fc_three_body.bias.requires_grad=False
        self.fc_three_body_2.weight.requires_grad=False
        #self.fc_three_body_2.bias.requires_grad=False

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.bn_three_body = nn.BatchNorm1d(n_edge_feature)

    def forward(
        self,
        node_embedding: Tensor,
        edge_embedding: Tensor,
        nbr_idx: Tensor,
        nbr_mask: Tensor,
        conductance,
    ) -> Tensor:
        """
        Calculate the updated edge embedding.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr :  Total number of neighbors of each atom

        Parameters
        ----------
        node_embedding : torch.Tensor
            batch of node embedding tensor of (B x At x n_node_feature) shape.
        edge_embedding : torch.Tensor
            batch of edge embedding tensor of (B x At x Nbr x n_edge_feuture) shape.
        nbr_idx : torch.Tensor
            Indices of neighbors of each atom. (B x At x Nbr) of shape.
        nbr_mask : torch.Tensor
            boolean mask for neighbor positions.(B x At x Nbr) of shape.

        Returns
        -------
        edge_embedding : torch.Tensor
            updated edge embedding tensor of (B x At x Nbr x n_edge_feuture) shape.
        """
        B, At, Nbr, n_edge_feature = edge_embedding.size()
        _, _, n_node_feature = node_embedding.size()

        # make c2_ij tensor. (B x At x Nbr x n_node_feature) of shape.
        nbh = nbr_idx.reshape(-1, At * Nbr, 1)
        nbh = nbh.expand(-1, -1, n_node_feature)
        # element-wise multiplication of node_i and node_j
        node_i = node_embedding.unsqueeze(2).expand(B, At, Nbr, n_node_feature)
        node_j = torch.gather(node_embedding, dim=1, index=nbh).view(B, At, Nbr, -1)
        node_j = node_j * nbr_mask[..., None]
        #c2 = node_i * node_j
        c2 = torch.cat([node_i , node_j],dim=3)
        edge_ij = edge_embedding.unsqueeze(3).expand(B, At, Nbr, Nbr, n_edge_feature)

        # reservoir computating with c2
        c2, min_val, max_val = quantization(self.fc_two_body,c2,conductance[0])
        c2 = re_quantization(c2, min_val, max_val, self.fc_two_body.weight)
        c2 = self.relu(c2)
        c2, min_val, max_val = quantization(self.fc_two_body_2,c2,conductance[1])
        c2 = re_quantization(c2, min_val, max_val, self.fc_two_body_2.weight)
        two_body_gate, two_body_extract = c2.chunk(2, dim=3)
        two_body_gate = self.sigmoid(two_body_gate)
        two_body_extract = self.tanh(two_body_extract)
        two_body_embedding = two_body_gate * two_body_extract
        two_body_embedding = self.bn_two_body(
             two_body_embedding.view(-1, n_edge_feature)
         ).view(B, At, Nbr, n_edge_feature)

        # make c3_ijk tensor. (B x At x Nbr x Nbr x 3*n_node_feature + 2*n_edge_feature) of shape.
        c3 = torch.cat(
            [  # node_i
                node_i.unsqueeze(3).expand(B, At, Nbr, Nbr, n_node_feature),
                # node_j
                node_j.unsqueeze(3).expand(B, At, Nbr, Nbr, n_node_feature),
                # node_k
                self.get_node_k(node_embedding, nbr_idx),
                # edge_ij
                edge_ij,
                # edge_jk
                self.get_edge_jk(edge_embedding, nbr_idx),
            ],
            dim=4,
        )

        # reservoir computating with c3
        c3, min_val, max_val = quantization(self.fc_three_body,c3,conductance[2])
        c3 = re_quantization(c3, min_val, max_val, self.fc_three_body.weight) 
        c3 = self.relu(c3)
        c3, min_val, max_val = quantization(self.fc_three_body_2,c3,conductance[3])
        c3 = re_quantization(c3, min_val, max_val, self.fc_three_body_2.weight)
        
        # calculate the gate and extract features with three-body interaction
        three_body_gate, three_body_extract = c3.chunk(2, dim=4)
        three_body_gate = self.sigmoid(three_body_gate)
        three_body_extract = self.tanh(three_body_extract)
        # elemet-wise multiplication with gate on three-body interaction
        three_body_embedding = three_body_gate * three_body_extract
        # apply neighbor mask
        # get j's neighbor masks. (B x At x Nbr x Nbr) of shape.
        nbr_idx_expand = (
            nbr_idx.unsqueeze(3).expand(B, At, Nbr, Nbr).reshape(B, At * Nbr, Nbr)
        )
        nbr_mask_expand = torch.gather(nbr_mask, 1, nbr_idx_expand).view(
            B, At, Nbr, Nbr
        )
        three_body_embedding = three_body_embedding * nbr_mask_expand[..., None]

        three_body_embedding = torch.sum(three_body_embedding, dim=3)
        three_body_embedding = self.bn_three_body(
            three_body_embedding.view(-1, n_edge_feature)
        ).view(B, At, Nbr, n_edge_feature)
        # last activation layer and Residual Network
        edge_embedding = self.relu(edge_embedding + two_body_embedding + three_body_embedding)
        return edge_embedding

class reservoir(nn.Module):
    """
    Automated feature extraction layer in GNNFF.

    Attributes
    ----------
    n_node_feature : int
        dimension of the embedded node features.
    n_edge_feature : int
        dimension of the embedded edge features.
    """

    def __init__(
        self,
        n_node_feature: int,
        n_edge_feature: int,
    ) -> None:
        super().__init__()
        self.update_node = NodeUpdate(n_node_feature, n_edge_feature)
        self.update_edge = EdgeUpdate(n_node_feature, n_edge_feature)

    def forward(
        self,
        node_embedding: Tensor,
        edge_embeding: Tensor,
        nbr_idx: Tensor,
        nbr_mask: Tensor,
        conductance
    ) -> Tensor:
        """
        Calculate the updated node and edge embedding by message passing layer.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr :  Total number of neighbors of each atom

        Parameters
        ----------
        node_embedding : torch.Tensor
            batch of node embedding tensor of (B x At x n_node_feature) shape.
        edge_embedding : torch.Tensor
            batch of edge embedding tensor of (B x At x Nbr x n_edge_feuture) shape.
        nbr_idx : torch.Tensor
            Indices of neighbors of each atom. (B x At x Nbr) of shape.
        nbr_mask : torch.Tensor
            boolean mask for neighbor positions.(B x At x Nbr) of shape.
        cell_offset : torch.Tensor or None, default=None
            offset of atom in cell coordinates with (B x At x Nbr x 3) shape.

        Returns
        -------
        node_embedding : torch.Tensor
            updated node embedding tensor of (B x At x n_node_feature) shape.
        edge_embedding : torch.Tensor
            updated edge embedding tensor of (B x At x Nbr x n_edge_feuture) shape.
        """
        
        node_embedding = self.update_node(
            node_embedding,
            edge_embeding,
            nbr_mask,
            conductance[:2]
        )
        edge_embedding = self.update_edge(
            node_embedding,
            edge_embeding,
            nbr_idx,
            nbr_mask,
            conductance[2:]
        )
        return node_embedding, edge_embedding
