import numpy as np
import torch

def data_load(x, batch_size, context_length, device):
    # max start index for samples
    max_start = len(x) - context_length

    # sample start index for each batch
    sample_start = np.random.randint(0, max_start, size=batch_size) # excluding max_start since next token labels are up to context_length + 1
    
    # get samples from start indices
    input_samples = np.stack([x[i : i + context_length] for i in sample_start])
    label_samples = np.stack([x[i + 1 : i + context_length + 1] for i in sample_start])

    input_tensor = torch.tensor(input_samples, dtype=torch.long, device=device)
    label_tensor = torch.tensor(label_samples, dtype=torch.long, device=device)

    return (input_tensor, label_tensor)

