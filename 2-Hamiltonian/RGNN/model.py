#References：
#1. H. Li, Z. Wang, N. Zou, M. Ye, R. Xu, X. Gong, W. Duan, Y. Xu, Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation, Nat. Comput. Sci. 2(6) (2022) 367-377.

#This code is extended from https://github.com/mzjb/DeepH-pack.git,which has the GNU LESSER GENERAL PUBLIC LICENSE. 

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.

import os
from typing import Union, Tuple
from math import ceil, sqrt

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.norm import LayerNorm, PairNorm, InstanceNorm
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
from torch_geometric.nn.models.dimenet import BesselBasisLayer
from torch_scatter import scatter_add, scatter
import numpy as np
from scipy.special import comb

from .from_se3_transformer import SphericalHarmonics
from .from_schnetpack import GaussianBasis
from .from_PyG_future import GraphNorm, DiffGroupNorm
from .from_HermNet import RBF, cosine_cutoff, ShiftedSoftplus, _eps
import matplotlib.pyplot as plt
import random



count = 0

def quantization(fc, embedding, gweights=None, nbit=8):
    """
    This function performs quantization of the input embedding and simulates the
    hardware noise during the vector-matrix multiplication using PyTorch.
    This part can be realized by analog circuit based on resistive memory.

    Args:
        fc: The fully connected layer to be applied after quantization.
        embedding: The input embedding matrix to be quantized.
        gweights: The random weights from the resistive memory array. If not provided,
                  no hardware noise will be simulated.
        nbit: The number of bits used in the quantization process. Default is 8.

    Returns:
        z: The quantized embedding matrix after simulating hardware noise (if applicable)
           and applying the fully connected layer.
    """

    # Compute the quantization base using the specified number of bits
    quant_base = 2 ** torch.arange(nbit).to(embedding.device)

    # Compute the number of quantization levels based on nbit
    nsplit = 2**nbit - 1

    # Find the minimum and maximum values of the input embedding
    min_val = torch.min(embedding)
    max_val = torch.max(embedding)

    # Normalize the input embedding to the range [0, nsplit]
    new_embedding = ((embedding - min_val) / (max_val - min_val) * nsplit).type(torch.uint8)

    # Convert the normalized input embedding to binary format
    binary = new_embedding.unsqueeze(-1).bitwise_and(quant_base).ne(0).byte()
    binary = binary.to(torch.float)

    # If gweights are provided, simulate the hardware noise
    if gweights is not None:
        x = new_embedding.to(torch.float)
        w = gweights[0] 
        noise = w + torch.randn_like(w) * w.abs() * 0.1
        fc.weight = torch.nn.Parameter(noise, requires_grad=False)

    # Perform the quantized vector-matrix multiplication using the fully connected layer
    z = fc(x)

    # Compute the bias term
    bias = torch.sum(min_val * noise, dim=1)

    # De-normalize the result and add the bias term
    z = z / nsplit * (max_val - min_val) + bias

    return z


class ExpBernsteinBasis(nn.Module):
    def __init__(self, K, gamma, cutoff, trainable=True):
        super(ExpBernsteinBasis, self).__init__()
        self.K = K
        if trainable:
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.gamma = torch.tensor(gamma)
        self.register_buffer('cutoff', torch.tensor(cutoff))
        self.register_buffer('comb_k', torch.Tensor(comb(K - 1, np.arange(K))))

    def forward(self, distances):
        f_zero = torch.zeros_like(distances)
        f_cut = torch.where(distances < self.cutoff, torch.exp(
            -(distances ** 2) / (self.cutoff ** 2 - distances ** 2)), f_zero)
        x = torch.exp(-self.gamma * distances)
        out = []
        for k in range(self.K):
            out.append((x ** k) * ((1 - x) ** (self.K - 1 - k)))
        out = torch.stack(out, dim=-1)
        out = out * self.comb_k[None, :] * f_cut[:, None]
        return out


def get_spherical_from_cartesian(cartesian, cartesian_x=1, cartesian_y=2, cartesian_z=0):
    spherical = torch.zeros_like(cartesian[..., 0:2])
    r_xy = cartesian[..., cartesian_x] ** 2 + cartesian[..., cartesian_y] ** 2
    spherical[..., 0] = torch.atan2(torch.sqrt(r_xy), cartesian[..., cartesian_z])
    spherical[..., 1] = torch.atan2(cartesian[..., cartesian_y], cartesian[..., cartesian_x])
    return spherical


class SphericalHarmonicsBasis(nn.Module):
    def __init__(self, num_l=5):
        super(SphericalHarmonicsBasis, self).__init__()
        self.num_l = num_l

    def forward(self, edge_attr):
        r_vec = edge_attr[:, 1:4] - edge_attr[:, 4:7]
        r_vec_sp = get_spherical_from_cartesian(r_vec)
        sph_harm_func = SphericalHarmonics()

        angular_expansion = []
        for l in range(self.num_l):
            angular_expansion.append(sph_harm_func.get(l, r_vec_sp[:, 0], r_vec_sp[:, 1]))
        angular_expansion = torch.cat(angular_expansion, dim=-1)

        return angular_expansion


"""
The class CGConv below is extended from "https://github.com/rusty1s/pytorch_geometric", which has the MIT License below

---------------------------------------------------------------------------
Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
class CGConv(MessagePassing):
    def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 0,
                 aggr: str = 'add', normalization: str = None,
                 bias: bool = True, if_exp: bool = False,no_grad: bool = False, **kwargs):
        super(CGConv, self).__init__(aggr=aggr, flow="source_to_target", **kwargs)
        self.channels = channels
        self.dim = dim
        self.normalization = normalization
        self.if_exp = if_exp
        self.no_grad = no_grad
        
        
        if isinstance(channels, int):
            channels = (channels, channels)
        
        
        if self.no_grad == True: 
           self.lin_f = nn.Linear(sum(channels) + dim, channels[1], bias=False)
           self.lin_s = nn.Linear(sum(channels) + dim, channels[1], bias=False)
           self.lin_f.weight.requires_grad = False
           weig_f = torch.randn_like(self.lin_f.weight)
           self.lin_f.weight = torch.nn.Parameter(weig_f,requires_grad=False)
        
           weig_s = torch.randn_like(self.lin_s.weight)
           self.lin_s.weight.requires_grad = False
           self.lin_s.weight = torch.nn.Parameter(weig_s,requires_grad=False)
        else :
           self.lin_f = nn.Linear(sum(channels) + dim, channels[1], bias=True)
           self.lin_s = nn.Linear(sum(channels) + dim, channels[1], bias=True)
        
        if self.normalization == 'BatchNorm':
            self.bn = nn.BatchNorm1d(channels[1], track_running_stats=True)
            self.bn.reset_parameters()

        elif self.normalization == 'LayerNorm':
            self.ln = LayerNorm(channels[1])
        elif self.normalization == 'PairNorm':
            self.pn = PairNorm(channels[1])
        elif self.normalization == 'InstanceNorm':
            self.instance_norm = InstanceNorm(channels[1])
        elif self.normalization == 'GraphNorm':
            self.gn = GraphNorm(channels[1])
        elif self.normalization == 'DiffGroupNorm':
            self.group_norm = DiffGroupNorm(channels[1], 128)
        elif self.normalization is None:
            pass
        else:
            raise ValueError('Unknown normalization function: {}'.format(normalization))
        

    def forward(self, x: Union[torch.Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor, batch, distance, g_weights = None,size: Size = None) -> torch.Tensor:
        """"""
        if isinstance(x, torch.Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, distance=distance,g_weights = g_weights, size=size)
        if self.normalization == 'BatchNorm':
            out = self.bn(out)
        elif self.normalization == 'LayerNorm':
            out = self.ln(out, batch)
        elif self.normalization == 'PairNorm':
            out = self.pn(out, batch)
        elif self.normalization == 'InstanceNorm':
            out = self.instance_norm(out, batch)
        elif self.normalization == 'GraphNorm':
            out = self.gn(out, batch)
        elif self.normalization == 'DiffGroupNorm':
            out = self.group_norm(out)
        out += x[1]
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor, distance, g_weights = None) -> torch.Tensor:
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        if self.no_grad == True:
           zf = quantization(self.lin_f,z,g_weights[0])
           zs = quantization(self.lin_s,z,g_weights[1])
           out = zf.sigmoid() * F.softplus(zs)
        else:
           out = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))
        
        if self.if_exp:
            sigma = 3
            n = 2
            out = out * torch.exp(-distance ** n / sigma ** n / 2).view(-1, 1)
        
        return out

    def __repr__(self):
        return '{}({}, dim={})'.format(self.__class__.__name__, self.channels, self.dim)


class MPLayer(nn.Module):
    def __init__(self, in_atom_fea_len, in_edge_fea_len, out_edge_fea_len, if_exp, if_edge_update, normalization,
                 atom_update_net, gauss_stop, output_layer=False, no_grad = False):
        super(MPLayer, self).__init__()
        if atom_update_net == 'CGConv':
            self.cgconv = CGConv(channels=in_atom_fea_len,
                                 dim=in_edge_fea_len,
                                 aggr='add',
                                 normalization=normalization,
                                 if_exp=if_exp,
                                 no_grad = no_grad,)

        self.if_edge_update = if_edge_update
        self.atom_update_net = atom_update_net
        self.no_grad = no_grad
        self.silu = nn.SiLU()

        if self.no_grad == True:
           self.fc1 = nn.Linear(in_edge_fea_len + in_atom_fea_len * 2, 320,bias=False)
           self.fc2 = nn.Linear(320, out_edge_fea_len,bias=False)
           weig_1 = torch.randn_like(self.fc1.weight)
           self.fc1.weight.requires_grad = False
           self.fc1.weight = torch.nn.Parameter(weig_1,requires_grad=False)
        
           weig_2 = torch.randn_like(self.fc2.weight)
           self.fc2.weight.requires_grad = False
           self.fc2.weight = torch.nn.Parameter(weig_2,requires_grad=False)
           self.silu = nn.SiLU() 
        else :
           self.fc1 = nn.Linear(in_edge_fea_len + in_atom_fea_len * 2, 14,bias=True)
           self.fc2 = nn.Linear(14, out_edge_fea_len,bias=True)
          

    def forward(self, atom_fea, edge_idx, edge_fea, batch, distance, edge_vec, g_weights = None):
        if self.no_grad == True:
          atom_fea = self.cgconv(atom_fea, edge_idx, edge_fea, batch, distance, g_weights[:2])
        else:
          atom_fea = self.cgconv(atom_fea, edge_idx, edge_fea, batch, distance)

        atom_fea_s = atom_fea
        #if self.if_edge_update:
        row, col = edge_idx
        z = torch.cat([atom_fea_s[row], atom_fea_s[col], edge_fea], dim=-1)
        if self.no_grad == True:
           z = quantization(self.fc1,z,g_weights[2])   
           z = self.silu(z)

           edge_fea = quantization(self.fc2,z,g_weights[3])
           edge_fea = self.silu(edge_fea)
        else:
           z = self.fc1(z)
           z = self.silu(z)
           edge_fea = self.fc2(z)
           edge_fea = self.silu(edge_fea)
        return atom_fea, edge_fea
        


class OutputLayer(nn.Module):
    def __init__(self, in_atom_fea_len, in_edge_fea_len, out_edge_fea_len, num_l,
                 normalization: str = None, bias: bool = True, if_exp: bool = False):
        super(OutputLayer, self).__init__()
        self.in_atom_fea_len = in_atom_fea_len
        self.normalization = normalization
        self.if_exp = if_exp

        self.lin_f = nn.Linear(in_atom_fea_len * 2 + in_edge_fea_len, in_atom_fea_len, bias=bias)
        self.lin_s = nn.Linear(in_atom_fea_len * 2 + in_edge_fea_len, in_atom_fea_len, bias=bias)
        self.ln = LayerNorm(in_atom_fea_len)
        self.e_lin = nn.Sequential(nn.Linear(in_edge_fea_len + in_atom_fea_len * 2 - num_l ** 2, 16),
                                   nn.SiLU(),
                                   nn.Linear(16, 16),
                                   nn.SiLU(),
                                   nn.Linear(16, out_edge_fea_len)
                                   )
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()

    def forward(self, atom_fea, edge_fea, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance,
                huge_structure, output_final_layer_neuron):
        num_edge = edge_fea.shape[0]
        z = torch.cat(
            [atom_fea[sub_atom_idx][:, 0, :], atom_fea[sub_atom_idx][:, 1, :], edge_fea[sub_edge_idx], sub_edge_ang],
            dim=-1)
        out = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))
        if self.if_exp:
            sigma = 3
            n = 2
            out = out * torch.exp(-distance[sub_edge_idx] ** n / sigma ** n / 2).view(-1, 1)
        out = scatter_add(out, sub_index, dim=0)
        out = out.reshape(num_edge, 2, -1)
        out = self.e_lin(torch.cat([out[:, 0, :], out[:, 1, :], edge_fea], dim=-1))
        return out


class RGNN(nn.Module):
    def __init__(self, num_species, in_atom_fea_len, in_edge_fea_len, num_orbital,
                 distance_expansion, gauss_stop, if_exp, if_MultipleLinear, if_edge_update, if_lcmp,
                 normalization, atom_update_net, separate_onsite,
                 trainable_gaussians, type_affine, num_l=5,):
        super(RGNN, self).__init__()
        self.num_species = num_species
        self.embed = nn.Embedding(num_species + 5, in_atom_fea_len)
        # pair-type aware affine
        if type_affine:
            self.type_affine = nn.Embedding(
                num_species ** 2, 2,
                _weight=torch.stack([torch.ones(num_species ** 2), torch.zeros(num_species ** 2)], dim=-1)
            )
        else:
            self.type_affine = None

        if if_edge_update or (if_edge_update is False and if_lcmp is False):
            distance_expansion_len = in_edge_fea_len
        else:
            distance_expansion_len = in_edge_fea_len - num_l ** 2
        if distance_expansion == 'GaussianBasis':
            self.distance_expansion = GaussianBasis(
                0.0, gauss_stop, distance_expansion_len, trainable=trainable_gaussians
            )
        elif distance_expansion == 'BesselBasis':
            self.distance_expansion = BesselBasisLayer(distance_expansion_len, gauss_stop, envelope_exponent=5)
        elif distance_expansion == 'ExpBernsteinBasis':
            self.distance_expansion = ExpBernsteinBasis(K=distance_expansion_len, gamma=0.5, cutoff=gauss_stop,
                                                        trainable=True)
        else:
            raise ValueError('Unknown distance expansion function: {}'.format(distance_expansion))

        self.if_MultipleLinear = if_MultipleLinear
        self.if_edge_update = if_edge_update
        self.if_lcmp = if_lcmp
        self.atom_update_net = atom_update_net
        self.separate_onsite = separate_onsite

        if if_lcmp == True:
            mp_output_edge_fea_len = in_edge_fea_len - num_l ** 2
        else:
            assert if_MultipleLinear == False
            mp_output_edge_fea_len = in_edge_fea_len
         
        if_exp2 = False
        if if_edge_update == True:
            self.mp1 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop, no_grad = True,)
            self.mp2 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop, no_grad = True,)
            self.mp3 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop, no_grad = True,)
            self.mp6 = MPLayer(in_atom_fea_len, in_edge_fea_len, mp_output_edge_fea_len, if_exp, if_edge_update,
                               normalization, atom_update_net, gauss_stop)
            
        else:
            self.mp1 = MPLayer(in_atom_fea_len, distance_expansion_len, None, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp2 = MPLayer(in_atom_fea_len, distance_expansion_len, None, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp3 = MPLayer(in_atom_fea_len, distance_expansion_len, None, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp4 = MPLayer(in_atom_fea_len, distance_expansion_len, None, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp5 = MPLayer(in_atom_fea_len, distance_expansion_len, None, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)

        if if_lcmp == True:
            if self.if_MultipleLinear == True:
                self.output = OutputLayer(in_atom_fea_len, in_edge_fea_len, 32, num_l, if_exp=if_exp)
                self.multiple_linear1 = MultipleLinear(num_orbital, 32, 16)
                self.multiple_linear2 = MultipleLinear(num_orbital, 16, 1)
            else:
                self.output = OutputLayer(in_atom_fea_len, in_edge_fea_len, num_orbital, num_l, if_exp=if_exp)
        else:
            self.mp_output = MPLayer(in_atom_fea_len, in_edge_fea_len, num_orbital, if_exp, if_edge_update=True,
                                     normalization=normalization, atom_update_net=atom_update_net,
                                     gauss_stop=gauss_stop, output_layer=True)


    def forward(self, atom_attr, edge_idx, edge_attr, batch,
                sub_atom_idx=None, sub_edge_idx=None, sub_edge_ang=None, sub_index=None,
                huge_structure=False, output_final_layer_neuron='', g_weights= None):
        batch_edge = batch[edge_idx[0]]
        atom_fea0 = self.embed(atom_attr)
        distance = edge_attr[:, 0]
        edge_vec = edge_attr[:, 1:4] - edge_attr[:, 4:7]
        if self.type_affine is None:
            edge_fea0 = self.distance_expansion(distance)
        else:
            affine_coeff = self.type_affine(self.num_species * atom_attr[edge_idx[0]] + atom_attr[edge_idx[1]])
            edge_fea0 = self.distance_expansion(distance * affine_coeff[:, 0] + affine_coeff[:, 1])
        if self.atom_update_net == "PAINN":
            atom_fea0 = PaninnNodeFea(atom_fea0)
        if self.if_edge_update == True:
            atom_fea, edge_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec, g_weights[0])
            atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
            atom_fea, edge_fea = self.mp2(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec, g_weights[1])
            atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
            atom_fea, edge_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec, g_weights[2])
            atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
            atom_fea, edge_fea = self.mp6(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            if self.if_lcmp == True:
                if self.atom_update_net == 'PAINN':
                    atom_fea_s = atom_fea.node_fea_s
                else:
                    atom_fea_s = atom_fea
                out = self.output(atom_fea_s, edge_fea, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance,
                                huge_structure, output_final_layer_neuron)
            else:
                atom_fea, edge_fea = self.mp_output(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)
                out = edge_fea
        else:
            atom_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea = self.mp2(atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea0 = atom_fea0 + atom_fea
            atom_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea = self.mp4(atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea0 = atom_fea0 + atom_fea
            atom_fea = self.mp5(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)

            if self.atom_update_net == 'PAINN':
                atom_fea_s = atom_fea.node_fea_s
            else:
                atom_fea_s = atom_fea
            if self.if_lcmp == True:
                out = self.output(atom_fea_s, edge_fea0, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance,
                                huge_structure, output_final_layer_neuron)
            else:
                atom_fea, edge_fea = self.mp_output(atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec)
                out = edge_fea

        if self.if_MultipleLinear == True:
            out = self.multiple_linear1(F.silu(out), batch_edge)
            out = self.multiple_linear2(F.silu(out), batch_edge)
            out = out.T

        return out
