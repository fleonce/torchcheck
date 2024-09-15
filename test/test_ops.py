import unittest

import torch

import torchcheck.ops as ops
from tools.test import BasicTestCase, foreach


class OpsTestCase(BasicTestCase):
    @foreach(dim={0, 1, 2}, dtype={torch.long})
    def test_full_inclusive_cumsum(self, dim: int, dtype: torch.dtype):
        input = torch.randint(127, (3, 3, 3), dtype=dtype)
        output = input.new_zeros(ops.replace_in_shape(input.shape, dim, input.shape[dim] + 1))
        empty_slice = slice(None, None, None)
        output_view = output[(empty_slice,) * dim + (slice(1, None, None),)]
        torch.cumsum(input, dim, out=output_view)

        self.assertTensorEqual(ops.inclusive_cumsum(input, dim, full=True), output)


if __name__ == "__main__":
    unittest.main()
