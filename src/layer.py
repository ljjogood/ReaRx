import torch
from torch import nn
from torch_geometric.data import Data
import torch.nn.functional as F
import torch_geometric.nn as gnn

# Multi-head attention
class MultiHeadHerbAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.k_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.v_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.temperature = nn.Parameter(torch.ones(1))
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)

        self.residual_factor = nn.Parameter(torch.tensor(0.1))

    def forward(self, query, herb_emb):
        B = query.shape[0]

        if herb_emb.dim() == 2:
            herb_emb = herb_emb.unsqueeze(0).expand(B, -1, -1)
        H = herb_emb.shape[1]

        Q = self.q_proj(query).unsqueeze(1)
        K = self.k_proj(herb_emb)
        V = self.v_proj(herb_emb)

        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, H, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, H, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5 * self.temperature)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = (attn_weights @ V).transpose(1, 2).contiguous().view(B, 1, self.dim)
        attn_output = self.out_proj(attn_output.squeeze(1))

        attn_output = query + self.residual_factor * attn_output

        attn_weights = attn_weights.mean(dim=1).squeeze(1)

        return attn_output, attn_weights



