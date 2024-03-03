#Reference:

#1. C.W. Park, M. Kornbluth, J. Vandermause, C. Wolverton, B. Kozinsky, J.P. Mailoa, Accurate and scalable graph neural network force field and molecular dynamics with direct force architecture, npj Comput. Mater. 7(1) (2021) 73. 
#This code is extended from https://github.com/ken2403/gnnff.git, which has the MIT License.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.


import numpy as np
from torch import Tensor
import torch.nn.functional as F


__all__ = ["softplus", "shifted_softplus"]


def softplus(x: Tensor) -> Tensor:
    """
    Compute shifted soft-plus activation function.
    .. math::
       y = \ln\left(1 + e^{-x}\right)

    Parameters
    ----------
    x : torch.Tensor
        input tensor.

    Returns
    -------
    torch.Tensor
        soft-plus of input.
    """
    return F.softplus(x)


def shifted_softplus(x: Tensor) -> Tensor:
    """
    Compute shifted soft-plus activation function.
    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Parameters
    ----------
    x : torch.Tensor
        input tensor.

    Returns
    -------
    torch.Tensor
        shifted soft-plus of input.

    References
    ----------
    .. [1] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.
    """
    return F.softplus(x) - np.log(2.0)
