from torch import Tensor
import torch
from typing import Optional


def batched_index_gen(x: Tensor, *, min_size: Optional[int] = None):
    max_indices: Tensor = x.to(torch.long).sum(-1).max()
    torch.clamp_min_(max_indices, min_size or 0)

    bs = x.size(0)
    seq_len = x.size(1)
    indices = torch.arange(seq_len, device=x.device).expand(bs, -1).clone()
    indices.masked_fill_(~x, seq_len)
    top_k, _ = torch.topk(indices, min_size or max_indices.item(), -1, False)
    top_k.masked_fill_(top_k.eq(seq_len), -1)
    return top_k


def batched_masked_select(x: Tensor, mask: Tensor, *, min_size: Optional[int] = None) -> tuple[Tensor, Tensor]:
    max_indices = mask.to(torch.long).sum(-1).max()
    max_indices.clamp_min_(min_size or 0)

    bs = mask.size(0)
    seq_len = mask.size(1)
    indices = torch.arange(seq_len, device=mask.device).expand(bs, -1).clone()
    indices.masked_fill_(~mask, seq_len)

    _, top_k_indices = torch.topk(indices, min_size or max_indices.item(), dim=-1, largest=False)
    mask_out = torch.gather(mask, -1, top_k_indices)
    if x.dim() > mask.dim():  # 3-dim case
        top_k_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
    values = torch.gather(x, -1, top_k_indices)
    return values, mask_out
