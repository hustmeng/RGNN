#Reference:

#1. C.W. Park, M. Kornbluth, J. Vandermause, C. Wolverton, B. Kozinsky, J.P. Mailoa, Accurate and scalable graph neural network force field and molecular dynamics with direct force architecture, npj Comput. Mater. 7(1) (2021) 73. 
#This code is extended from https://github.com/ken2403/gnnff.git, which has the MIT License.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.


from torch import Tensor
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import calculate_gain
import torch

__all__ = ["Linear","reservoir_Linear"]

class reservoir_Linear(nn.Linear):
    # Custom fully connected (linear) layer for resistive memory-based calculation with random weights (untrainable)
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation=None,
        weight_init=xavier_uniform_,
        #bias_init=constant_,
        mean=0,
        std=1.0,
    ) -> None:
        """
        Initialize the custom fully connected layer.

        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            bias (bool, optional): Whether to include a bias term, default is True
            activation (callable, optional): Activation function to be applied after the linear transformation
            weight_init (callable, optional): Weight initialization function, default is xavier_uniform_
            bias_init (callable, optional): Bias initialization function, default is constant_
            mean (float, optional): Mean for the random weights, default is 0
            std (float, optional): Standard deviation for the random weights, default is 1.0
        """
        self.activation = activation
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.mean = mean
        self.std = std

        super().__init__(in_features, out_features, bias)

    def reset_parameters(self) -> None:
        """
        Reset the weights and biases of the layer with random values (untrainable).
        """
        weig = torch.randn_like(self.weight)
        new_weig = self.mean + weig * self.std
        self.weight = torch.nn.Parameter(new_weig, requires_grad=False)

        if self.bias is not None:
            self.bias_init(self.bias, val=0.0)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Perform the forward pass of the custom fully connected layer.

        Args:
            inputs (Tensor): Input tensor

        Returns:
            y (Tensor): Output tensor
        """
        y = super().forward(inputs)
        if self.activation:
            y = self.activation(y)
        return y

class Linear(nn.Linear):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation=None,
        weight_init=xavier_uniform_,
        bias_init=constant_,
    ) -> None:
        self.activation = activation
        self.weight_init = weight_init
        self.bias_init = bias_init
        super().__init__(in_features, out_features, bias)

    def reset_parameters(self) -> None:
        """
        Reinitialize model weight and bias values.
        """
        self.weight_init(self.weight, gain=calculate_gain("linear"))
        if self.bias is not None:
            self.bias_init(self.bias, val=0.0)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Compute layer output.
        """
        # compute linear layer y = xW^T + b
        y = super().forward(inputs)
        # add activation function
        if self.activation:
            y = self.activation(y)
        return y

