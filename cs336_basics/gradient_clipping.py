import torch

def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    # Collect all gradients and flatten them into a single vector
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
    
    if not grads:
        return  # No gradients to clip
    
    # Concatenate all flattened gradients
    all_grads = torch.cat(grads)
    l2_norm = torch.norm(all_grads, p=2)

    if l2_norm > max_l2_norm:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.data *= max_l2_norm / (l2_norm + eps)

