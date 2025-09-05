import torch
from einops import rearrange, reduce

def cross_entropy(logits, targets):
    """
    logits (batch, ..., vocab_size)
    targets (batch, ...) with values in [0, vocab_size-1]
    loss = mean(-log softmax(logits)[batch, ..., targets])
         = mean(logsumexp(logits) - logits[batch, ..., targets])
    """
    max_logits = reduce(logits, 'b ... v -> b ... 1', 'max')
    logits = logits - max_logits  # substracting max for numerical stability
    exp_logits = torch.exp(logits)  # (batch, ..., vocab_size)
    sum_exp_logits = reduce(exp_logits, 'b ... v -> b ...', 'sum')
    logsumexp = torch.mean(torch.log(sum_exp_logits)) # scalar

    logits = rearrange(logits, 'b ... v -> (b ...) v')
    targets = rearrange(targets, 'b ... -> (b ...)')
    targets = logits[torch.arange(targets.shape[0]), targets] # (b ...)
    loss = logsumexp - torch.mean(targets)  # scalar

    return loss

