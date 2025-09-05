import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        '''
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize the embedding matrix
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''
        input: Long[Tensor, "batch_size sequence_length"] Indices of the tokens to embed
        Returns:
            Float[Tensor, "batch_size sequence_length embedding_dim"]: The embedded tokens
        '''
        return self.weight[token_ids]

