import torch  # noqa F401
from torch import Tensor, SymInt
from torch._subclasses import FakeTensor
from typing import Optional, Union
import torchcheck.C

from torchcheck.C import (
    assert_dim,
    assert_eq,
    assert_true,
    assert_dtype,
)
import torchcheck.config as config
import torchcheck.impl as impl
import torchcheck.C as C

from .attention import T5MultiHeadAttention


def assert_shape(x: Tensor, shape: tuple[Union[int, SymInt], ...]) -> int:
    if config.use_python_equivalents and False:
        return impl.assert_shape(x, shape)
    return C.assert_shape(x, shape)


def batched_index_gen(x: Tensor, *, min_size: Optional[int] = None) -> Tensor:
    if config.use_python_equivalents:
        return impl.batched_index_gen(x, min_size=min_size)
    return C.batched_index_gen(x, min_size=min_size)


def batched_masked_select(x: Tensor, mask: Tensor, *, min_size: Optional[int] = None) -> tuple[Tensor, Tensor]:
    if config.use_python_equivalents:
        return impl.batched_masked_select(x, mask, min_size=min_size)
    return C.batched_masked_select(x, mask, min_size=min_size)
