from numbers import Number

from torch import (
    dtype as _dtype,
    Tensor,
    SymInt,
    SymFloat,
)
from typing import SupportsInt, Optional

def assert_shape(x: Tensor, shape: tuple[int | SupportsInt, ...]) -> int: ...
def assert_dtype(x: Tensor, dtype: _dtype) -> int: ...
def assert_dim(x: Tensor, dim: int) -> int: ...
def assert_true(x: bool, msg: str) -> int: ...
def assert_eq(x: Tensor, y: Tensor) -> int: ...

def batched_index_gen(x: Tensor, *, min_size: Optional[int] = None) -> Tensor: ...
def batched_masked_select(x: Tensor, mask: Tensor, *, min_size: Optional[int] = None) -> tuple[Tensor, Tensor]: ...

def t5_self_attention(query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor, num_heads: int) -> Tensor: ...