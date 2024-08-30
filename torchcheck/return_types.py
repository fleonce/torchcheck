from typing import NamedTuple

from torch import Tensor


class batched_index_padded_t(NamedTuple):
    values: Tensor
    mask: Tensor


class batched_masked_select_t(NamedTuple):
    values: Tensor
    mask: Tensor
