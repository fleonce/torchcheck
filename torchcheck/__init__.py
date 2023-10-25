import torch  # noqa F401
import torchcheck.C

from torchcheck.C import (
    assert_shape,
    assert_dim,
    assert_eq,
    assert_true,
    assert_dtype,
    batched_index_gen,
    batched_masked_select,
)

from .attention import T5MultiHeadAttention
