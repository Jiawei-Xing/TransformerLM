import torch
import torch.nn as nn
from einops import repeat

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        '''
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_len = max_seq_len
        self.device = device

        # get angle matrix (theta i, k)
        angle = 1 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        position = torch.arange(max_seq_len, device=device)
        matrix = torch.outer(position, angle) # (max_seq_len, d_k // 2)

        # save rotations as a buffer
        self.register_buffer('sin', torch.sin(matrix), persistent=False)
        self.register_buffer('cos', torch.cos(matrix), persistent=False)

    def forward(self, x, token_positions=None):
        '''
        x: torch.Tensor Input tensor of shape (..., seq_len, d_k)
        token_positions: torch.Tensor Positions of tokens in the sequence (..., seq_len)
        Returns:
            torch.Tensor: Tensor with RoPE applied, shape (..., seq_len, d_k)
        '''
        # If no token positions are provided, assume sequential positions
        if token_positions is None:
            positions = torch.arange(x.size(-2), device=x.device)  # shape: (len,)
            dims = [f'b{i}' for i in range(x.ndim - 2)] # ['b0', 'b1', ..., 'bn-3']
            shapes_dict = {dim: shape for dim, shape in zip(dims, x.shape[:-2])}
            token_positions = repeat(positions, f"l -> {' '.join(dims)} l", **shapes_dict) # (..., seq_len)

        # Get the sine and cosine values for the token positions
        sin = self.sin.to(x.device)[token_positions] # (..., seq_len, d_k // 2)
        cos = self.cos.to(x.device)[token_positions]

        # Apply RoPE separately by even rows and odd rows
        x_rotary = x.clone()
        x_rotary[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin # (..., seq_len, d_k // 2)
        x_rotary[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos

        return x_rotary

