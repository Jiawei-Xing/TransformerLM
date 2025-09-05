import torch

def softmax(x, i):
    max_i = torch.max(x, dim=i, keepdim=True)[0]
    # Subtract max for numerical stability
    exp_x = torch.exp(x - max_i)
    x = exp_x / torch.sum(exp_x, dim=i, keepdim=True)
    return x

