import torch
from torch import (
    dtype as _dtype
)

def assert_shape(x: torch.Tensor, shape: tuple[int, ...]) -> int: ...

def assert_dtype(x: torch.Tensor, dtype: _dtype) -> int: ...

def batched_index_gen(x: torch.Tensor) -> torch.Tensor: ...