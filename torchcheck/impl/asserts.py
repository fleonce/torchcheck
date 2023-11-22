from typing import Union

from torch import Tensor, SymInt

_int = Union[int, SymInt]


def assert_shape(x: Tensor, shape: tuple[_int, ...]) -> int:
    sizes = x.size()
    assert len(sizes) == len(shape), f"Expected shape of tensor to be: {shape} but got: {x.size()}"
    for i, j in zip(sizes, shape):
        assert i == j, (i == j, i, j)
    assert sizes == shape, f"Expected shape of tensor to be: {shape} but got: {x.size()}"
    return 0

