from typing import Union

from torch import Tensor, SymInt

_int = Union[int, SymInt]


def assert_shape(x: Tensor, shape: tuple[_int, ...]) -> int:
    sizes = x.size()
    assert len(sizes) == len(shape), f"Expected shape of tensor to be: {shape} but got: {x.size()}"
    for dim, (i, j) in enumerate(zip(sizes, shape)):
        eq = i == j
        if isinstance(eq, bool):
            assert eq, f"Expected shape of tensor to be: {shape} but got {i} vs {j} in dimension {dim}"
        assert str(i) == str(j), f"Expected shape of tensor to be: {shape} but got {i} vs {j} in dimension {dim}"
        # eq: SymBool
        # node: SymNode = eq.node
        # env: ShapeEnv = node.shape_env
        # replacement = env.evaluate_expr(node.expr, None, None)
    return 0

