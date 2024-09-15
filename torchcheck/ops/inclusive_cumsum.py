import torch
from torch import Tensor


shape_t = tuple[int | torch.SymInt, ...]


def replace_in_shape(shape: shape_t, dim: int, size: int | torch.SymInt) -> shape_t:
    sizes = list(shape)
    sizes[dim] = size
    return tuple(sizes)


def inclusive_cumsum(self: Tensor, dim: int, *, full: bool = False):
    self_shape = self.shape

    inclusive_shape = replace_in_shape(self_shape, dim=dim, size=self_shape[dim] + 1)
    cumsum_out = self.new_zeros(inclusive_shape)

    empty_slice = slice(None, None, None)
    view = (empty_slice,) * dim + (slice(1, None, None),)
    torch.cumsum(self, dim=dim, out=cumsum_out[view])

    if not full:
        out_view = (empty_slice,) * dim + (slice(None, -1, None),)
        return cumsum_out[out_view].contiguous()
    return cumsum_out
