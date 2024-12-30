from typing import overload, Iterable
from functools import lru_cache

from torch import Tensor
from torchcheck.ops.expand import expand_dims
import torch


@lru_cache(255)
def _operand_to_sizes(shape: tuple[int, ...], operand: str) -> tuple[int, ...]:
    inp, outp = operand.split("->")
    inp_dims = inp.split(",")
    outp_dims = outp.split(",")

    if len(inp_dims) != len(shape):
        raise ValueError("Got an invalid number of dimensions for tensor of shape ", shape, " with operand ", operand)

    outp_inp_dims = [dim for dim in outp_dims if dim in inp_dims]
    if outp_inp_dims != inp_dims:
        raise ValueError(
            f"The order of dimensions must be the same for input and output, "
            f"got {inp_dims} vs {outp_dims} ({outp_inp_dims})"
        )

    sizes: tuple[int, ...] = tuple()
    for i, elem in enumerate(outp_dims):
        size = 1
        if elem in inp_dims:
            size = shape[inp_dims.index(elem)]

        sizes = sizes + (size,)
    return sizes

def unsqueeze(self: Tensor, operand: str) -> Tensor:
    """
    Unsqueeze a tensor using a string description of the desired shape:

    >>> t = unsqueeze(t, "abc->a,bc")
    will return a 4 dimensional tensor of shape `(a,1,b,c)`

    Format: [a-zA-Z]+->[,a-zA-Z]+
    """
    orig_sizes = tuple(self.size(dim) for dim in range(self.dim()))
    sizes = _operand_to_sizes(orig_sizes, operand)
    return self.view(sizes)

def unsqueeze_expand(self: Tensor, operand: str, *dims_and_sizes: tuple[int, int]):
    """
    Unsqueeze a tensor using a string description of the desired shape
    and expand it along the new shape.

    >>> t = unsqueeze_expand(t, "abc->a,bc", (1, 4))
    will return a 4 dimensional tensor of shape `(a,4,b,c)`

    For format, look at `unsqueeze`
    """
    self = unsqueeze(self, operand)
    return expand_dims(self, dims_and_sizes)

@overload
def unsqueeze_dims(self: Tensor, dims: tuple[int, ...]) -> Tensor: ...


@overload
def unsqueeze_dims(self: Tensor, *dims: int) -> Tensor: ...


def unsqueeze_dims(self: Tensor, dims: None | int | tuple[int, ...] = None, *other_dims: int) -> Tensor:
    """
        Sequentially unsqueeze a sequence of dimensions from a tensor

        Args:
            self (Tensor): The tensor
            dims (int): The dimensions

        Returns: A view of ``self`` with the given dims unsqueezed
        """
    if dims is None or (
        isinstance(dims, tuple)
        and len(dims) == 0
    ):
        raise ValueError("You must specify at least one dimension, you specified zero.")
    if not isinstance(dims, tuple):
        dims = (dims,) + other_dims

    for dim in dims:
        self = self.unsqueeze(dim)
    return self


def unsqueeze_and_expand_dims(self: Tensor, unsqueeze: int | tuple[int, ...], *dims_and_sizes: tuple[int, int]):
    self = unsqueeze_dims(self, unsqueeze)
    return expand_dims(self, dims_and_sizes)
