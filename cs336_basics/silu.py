import torch

def silu(in_features):
    return in_features * torch.sigmoid(in_features)