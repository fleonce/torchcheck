from typing import Union

from torch import Tensor, SymInt

_int = Union[int, SymInt]


def assert_shape(x: Tensor, shape: tuple[_int, ...]) -> int:
    sizes = x.size()
    assert len(sizes) == len(shape)
    for i, j in zip(sizes, shape):
        pass
    assert x.size() == len(shape), f"Expected shape of tensor to be: {shape} but got: {x.size()}"
    assert sizes == shape, f"Expected shape of tensor to be: {shape} but got: {x.size()}"
    return 0

