#Reference:

#1. C.W. Park, M. Kornbluth, J. Vandermause, C. Wolverton, B. Kozinsky, J.P. Mailoa, Accurate and scalable graph neural network force field and molecular dynamics with direct force architecture, npj Comput. Mater. 7(1) (2021) 73. 
#This code is extended from https://github.com/ken2403/gnnff.git, which has the MIT License.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.


import torch.nn as nn
from torch import Tensor

from RGNN.model.functional import shifted_softplus


__all__ = ["ShiftedSoftplus"]


class ShiftedSoftplus(nn.Module):
    """
    Applies the element-wise function:

    .. math::
       y = \ln\left(1 + e^{-x}\right)

    Notes
    -----
    - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
    - Output: :math:`(*)`, same shape as the input.

    Examples
    --------
    >>> ss = gnnff.nn.ShiftedSoftplus()
    >>> input = torch.randn(2)
    >>> output = ss(input)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return shifted_softplus(input)
