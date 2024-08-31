from typing import NamedTuple

from torch import Tensor


class batched_index_padded(NamedTuple):
    values: Tensor
    mask: Tensor


class batched_masked_select(NamedTuple):
    values: Tensor
    mask: Tensor
