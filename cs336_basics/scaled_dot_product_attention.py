import torch
from torch import nn
from cs336_basics.softmax import softmax

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute the scaled dot-product attention.

    Args:
        query: Query tensor of shape (batch_size, ..., seq_len, d_k).
        key: Key tensor of shape (batch_size, ..., seq_len, d_k).
        value: Value tensor of shape (batch_size, ..., seq_len, d_v).
        mask: Optional mask tensor of shape (seq_len, seq_len).

    Returns:
        Output tensor of shape (batch_size, ..., d_v).
    """
    d_k = query.size(-1)
    
    # Calculate the dot product attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5) # (batch_size, ..., seq_len, seq_len)
    
    # add -inf to false positions in the mask
    if mask is not None:
        # Move mask to the same device as scores
        mask = mask.to(scores.device)
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Apply softmax to get attention weights
    attn_weights = softmax(scores, -1)
    
    # Compute the output as a weighted sum of values
    output = torch.matmul(attn_weights, value) # (batch_size, ..., seq_len, d_v)
    
    return output

