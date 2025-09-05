import torch
from torch import nn

class RMSNorm(nn.Module):  
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameter
        '''
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Process an input tensor of shape (batch_size, sequence_length, d_model) 
        and return a tensor of the same shape.
        '''
        in_type = x.dtype
        x = x.to(torch.float32) # upcast to float32 to prevent overflow
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = (x / norm) * self.weight
        return result.to(in_type)
