import warnings
from typing import Optional

import torch

from torchcheck.return_types import batched_masked_select_t


def batched_index_padded(
    self: torch.Tensor,
    pad_value: int = -1,
    *,
    out: torch.Tensor = None,
    min_size: Optional[torch.Tensor | int] = None,
    verify_outputs: Optional[bool] = None,
):
    """
    Generate a new ``torch.Tensor`` based on self, a mask. Returns a shorter Tensor with indices where self == True.

    """
    if self.dtype != torch.bool:
        raise ValueError(repr(self.dtype) + " not supported for `batched_index_gen`")

    if min_size is not None:
        warnings.warn(
            "Specifying min_size is deprecated and will be removed in a future release.",
            stacklevel=2
        )

    if out is None:
        dim_size = self.sum(dim=-1).amax()
        dims = self.shape[:-1] + (dim_size,)
        out = self.new_zeros(dims, dtype=torch.long)
        pass
    elif out.dtype != torch.long:
        raise ValueError("Out tensor must have dtype " + repr(torch.long))

    # move to int8, since topk is not supported for torch.bool
    self = self.to(torch.int8)
    value_out = torch.empty_like(out, dtype=self.dtype)

    torch.topk(self, k=out.shape[-1], dim=-1, out=(value_out, out))

    # convert back to bool
    value_mask = value_out.to(torch.bool)
    out.masked_fill_(~value_mask, pad_value)
    return out


def expand_as(
    self: torch.Tensor,
    ref: torch.Tensor,
    broadcastable: bool = False,
) -> torch.Tensor:
    self_dim = self.dim()

    ref_shape = ref.shape
    ref_dim = ref.dim()

    dim_diff = ref_dim - self_dim
    dims = self.shape + (1,) * dim_diff
    self = self.view(dims)
    if not broadcastable:
        expands = (-1,) * self_dim + ref_shape[self_dim:]
        self = self.expand(expands)
    return self


def batched_masked_select(
    self: torch.Tensor,
    mask: torch.Tensor,
    pad_value: int = -1,
    *,
    out: Optional[torch.Tensor] = None,
    min_size: Optional[int] = None,
) -> batched_masked_select_t:
    if mask.dtype != torch.bool:
        raise ValueError(repr(mask.dtype) + " is not supported for `batched_masked_select`")
    if self.dim() < mask.dim():
        raise ValueError("Expected self to have at least as many dims as mask, but got " + str(self.dim()) + " vs " + str(mask.dim()))

    if min_size is not None:
        warnings.warn(
            "Specifying min_size is deprecated and will be removed in a future release.",
            stacklevel=2
        )

    if out is None:
        dim_size = mask.sum(dim=-1).amax()
        out = self.new_empty(self.shape[:mask.dim()] + (dim_size,))

    mask = mask.to(torch.int8)
    # perform topk on mask ...
    values, indices = torch.topk(mask, k=out.shape[-1], dim=-1)
    # return to bool for values
    mask = values.to(torch.bool)

    indices = expand_as(indices, self)
    masked = torch.gather(self, dim=mask.dim() - 1, index=indices)
    masked.masked_fill_(
        ~expand_as(mask, masked, broadcastable=True),
        pad_value,
    )
    return batched_masked_select_t(masked, mask)


batched_index_gen = batched_index_padded

batched_index_gen(torch.randint(2, (3, 4), dtype=torch.bool), min_size=4)


batched_masked_select(
    torch.randn((32, 32, 32),),
    torch.randint(2, (32, 32), dtype=torch.bool),
)
