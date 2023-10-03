import typing

import torch
import torch.nn as nn
import torchcheck.C
try:
    from transformers import T5Config
except ImportError:
    T5Config = typing.Any


class T5MultiHeadAttention(torch.nn.Module):

    def __init__(self, config: T5Config):
        super().__init__()
        self.inner_dim = config.num_heads * config.d_kv
        self.num_heads = config.num_heads
        self.q = nn.Linear(config.d_ff, self.inner_dim, bias=False)
        self.k = nn.Linear(config.d_ff, self.inner_dim, bias=False)
        self.v = nn.Linear(config.d_ff, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, config.d_ff, bias=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor):
        return self.o(torchcheck.C.t5_self_attention(self.q(q), self.k(k), self.v(v), attn_mask, self.num_heads))
