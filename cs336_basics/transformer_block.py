import torch
import torch.nn as nn
from cs336_basics.multihead_self_attention import MultiheadSelfAttentionRoPE
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.positionwise_feedforward import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.attn = MultiheadSelfAttentionRoPE(d_model, num_heads, max_seq_len, theta)
        self.ln1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.ffn.d_ff = self.d_ff
        self.ln2 = RMSNorm(d_model)

    def forward(self, x):
        # multi-head self-attention with RoPE
        x_norm = self.ln1(x)
        y = x + self.attn(x_norm) # plus residual without norm

        # feed-forward network
        y_norm = self.ln2(y)
        z = y + self.ffn(y_norm)  # plus residual without norm

        return z

