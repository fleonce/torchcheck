from typing import Union

from torch import Tensor, SymInt

_int = Union[int, SymInt]


def assert_shape(x: Tensor, shape: tuple[_int, ...]) -> int:
    print(x.size(), shape)
    sizes = x.size()
    assert len(sizes) == len(shape)
    for i, j in zip(sizes, shape):
        if isinstance(i, SymInt):
            print(i, i.node, i.node.shape_env.var_to_val, x, x.size())
        # print(i, j)
    assert x.size() == shape, f"Expected shape of tensor to be: {shape} but got: {x.size()}"
    return 0

