import torch
import torch.nn as nn
from cs336_basics.linear import Linear
from cs336_basics.silu import silu

class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.d_ff = int(d_model * 8/3 // 64 * 64)  # Ensure d_ff is a multiple of 64

        # Initialize weights
        self.w1 = Linear(self.d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, self.d_model)
        self.w3 = Linear(self.d_model, self.d_ff)

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))

