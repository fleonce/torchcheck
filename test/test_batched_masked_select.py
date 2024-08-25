import unittest

import torch

from tools.test import BasicTestCase, foreach
from torchcheck import batched_masked_select


class BatchedMaskedSelectTestCase(BasicTestCase):

    @foreach(dtype={torch.float, torch.long, torch.bfloat16, torch.int, torch.uint8}, size={1, 2, 3, 4})
    def test_batched_masked_select(self, dtype: torch.dtype, size: int):
        x = torch.tensor([[0, 1], [1, 0]]).bool()
        y = self.create_random_dtype_tensor((2,) * size, dtype=dtype)

        if size < 2:
            with self.assertRaises(ValueError):
                _ = batched_masked_select(y, x)
        else:
            values, mask = batched_masked_select(y, x)
            self.assertTensorEqual(
                y[[0, 1], [1, 0], None],
                values
            )


if __name__ == "__main__":
    unittest.main()
