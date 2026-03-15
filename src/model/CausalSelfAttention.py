"""
Casual self attention written from scratch.
We specify multiple heads for multi headed attention.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F



class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        
        '''d_model: length of each token embedding vector (256 for now)'''
        super().__init__()
        assert d_model % n_heads == 0

        # the number of subspaces we have to capture relationships
        # the vector is split into parts, one part for each head
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # 256 / 4 = 64 per head

        # Q, K, V projections combined into one linear layer for efficiency
        # Output: (batch, seq_len, 3 * d_model=768) → split into Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        # Projects concatenated head outputs back to d_model
        self.out_proj = nn.Linear(d_model, d_model)

        # what is a drop out
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, T, d_model)
        Total  of "batch" sequences, of at most T tokens,
        each token represented as a d_model dimensional vector.
        """

        B, T, C = x.size()
        # apply the projection of the query, key, value matrices onto tensor of sequences
        qkv = self.qkv_proj(x)
        # left with the compressed matrices for all heads
        q_compact, k_compact, v_compact = qkv.chunk(3, dim=-1)

        # think of q_compact as Batch size number of T by d_model attention matrices
        # we need to unpack them, keep as B matrices, keep T rows, 
        # now split the token vector length into 4 groups, each group has 64 features
        q = q_compact.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k_compact.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v_compact.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # compute the attention scores

        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        # Apply the casual mask for autoregressive generation

        # create an upper triangular matrix of 1s excluding the main diagonal
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        # apply the mask to the scores and replace masked entries with -inf
        scores = scores.masked_fill(mask, float('-inf'))

        # apply softmax on the scores for dim (-1), 
        # have the columns sum to 1 to get probabilities
        attn = F.softmax(scores, dim=-1)
        # dropout regularization to each row
        attn = self.attn_dropout(attn)
        # return outputed attention
        out = attn @ v

        # concatenate heads: (B, 4, T, 64) --> (B, T, 256) tensor
        concat = out.transpose(1, 2).reshape(B, T, -1)

        # mix across heads (creating an interpreable linera combination) and apply dropout
        # each dimension is a learned mix of all the head's outputs
        # concat @ W + bias
        out = self.resid_dropout(self.out_proj(concat))

        return out