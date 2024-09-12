from typing import overload

from torch import Tensor
from torchcheck.ops.expand import expand_dims


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
