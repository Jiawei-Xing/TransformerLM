import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
        Initialize the Linear layer with given input and output features.

        Args:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights
        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), device=device, dtype=dtype))
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        
        Args:
            x: Input tensor of shape (batch_size, ..., in_features).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, ..., out_features).
        """
        # return x @ self.weight.T
        return einsum(x, self.weight, '... i, o i -> ... o')

