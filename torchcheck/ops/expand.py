from typing import overload

import torch
from torch import Tensor, SymInt

dim_with_size = tuple[int, int]


def maybe_wrap_dim(tensor: Tensor, dim: int | SymInt) -> int | SymInt:
    min_dim = -tensor.dim()
    max_dim = tensor.dim() + 1
    if dim < min_dim or dim > max_dim:
        raise ValueError(
            f"Dimension out of range (expected to be in range of [{min_dim}, {max_dim}], but got {dim}"
        )
    if dim < 0:
        return dim + tensor.dim()
    return dim


def expand_dim(tensor: Tensor, dim: int, expand_size: int) -> Tensor:
    expands = [-1] * tensor.dim()
    wrap_dim = maybe_wrap_dim(tensor, dim)
    expands[wrap_dim] = expand_size
    return tensor.expand(expands)


@overload
def expand_dims(self: Tensor, *dims_and_sizes: dim_with_size) -> Tensor: ...


@overload
def expand_dims(self: Tensor, dims_and_sizes: list[dim_with_size]) -> Tensor: ...


@overload
def expand_dims(self: Tensor, dims_and_sizes: tuple[dim_with_size, ...]) -> Tensor: ...


def expand_dims(
    self: Tensor, dims_and_sizes: None | dim_with_size | list[dim_with_size] | tuple[dim_with_size, ...] = None,
    *other_dims: dim_with_size
    ) -> Tensor:
    if (
        dims_and_sizes is None
        or (isinstance(dims_and_sizes[0], list) and len(dims_and_sizes[0]) == 0)
        or (isinstance(dims_and_sizes[0], tuple) and len(dims_and_sizes[0]) == 0)
    ):
        raise ValueError("You must specify at least one dimension, you specified zero.")
    if (
        isinstance(dims_and_sizes, list)
        and len(other_dims) > 0
    ):
        raise ValueError("Invalid combination of vararg and dim_and_sizes list")
    if (
        other_dims
        and isinstance(dims_and_sizes, tuple)
        and len(dims_and_sizes) == 2
        and isinstance(dims_and_sizes[0], int)
        and isinstance(dims_and_sizes[1], int)
    ):
        dims_and_sizes = [(dims_and_sizes[0], dims_and_sizes[1])] + list(other_dims)
    if isinstance(dims_and_sizes, tuple):
        assert not isinstance(dims_and_sizes[0], int)
        assert len(dims_and_sizes) == 1 or not isinstance(dims_and_sizes[1], int)
        dims_and_sizes = list(dims_and_sizes)
    if not isinstance(dims_and_sizes, list):
        assert False, dims_and_sizes
    expands = [-1] * self.dim()
    for dim, size in dims_and_sizes:
        wrap_dim = maybe_wrap_dim(self, dim)
        expands[wrap_dim] = size
    return self.expand(expands)


def expand_as(
    self: Tensor,
    ref: Tensor,
    broadcastable: bool = False,
    dim: int = -1
) -> Tensor:
    # todo use dim to determine where to put the new dimensions
    self_dim = self.dim()

    ref_shape = ref.shape
    ref_dim = ref.dim()

    dim_diff = ref_dim - self_dim
    dims = self.shape + (1,) * dim_diff
    self = self.view(dims)
    if not broadcastable:
        self, _ = torch.broadcast_tensors(self, ref)
        expands = (-1,) * self_dim + ref_shape[self_dim:]
        self = self.expand(expands)
    return self
