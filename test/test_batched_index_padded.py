import unittest

import torch
from torch._dynamo.exc import Unsupported

from tools.test import BasicTestCase, foreach, TORCH_DEVICE_LIST
from torchcheck import batched_index_padded, assert_eq


class BatchedIndexPaddedTestCase(BasicTestCase):

    @foreach(device=TORCH_DEVICE_LIST)
    def test_batched_index_order(self, device: str):
        mask = torch.randint(2, (10, 10), dtype=torch.bool, device=device)
        out = batched_index_padded(mask, 10)
        sort = torch.sort(out)[0]
        self.assertTensorEqual(out, sort)

    @foreach(dtype={torch.int, torch.long, torch.uint8, torch.uint32, torch.float16, torch.bfloat16, torch.float32, torch.float64})
    def test_batched_index_padded_wrong_dtype(self, dtype: torch.dtype):
        if dtype.is_floating_point:
            mask = torch.randn((2, 10), dtype=dtype)
        else:
            iinfo = torch.iinfo(dtype)
            mask = torch.randint(iinfo.max, (2, 10), dtype=dtype)
        with self.assertRaises(ValueError):
            batched_index_padded(mask)

    @staticmethod
    def test_batched_index_padded_normal_dim():
        inp = torch.randint(2, (32, 64), dtype=torch.bool)
        out = batched_index_padded(inp)

        assert_eq(out.ne(-1).sum(dim=-1), inp.sum(dim=-1))

    @staticmethod
    def test_batched_index_padded_higher_dim():
        for i in range(1, 6):
            inp = torch.randint(2, (2,) * i + (10,), dtype=torch.bool)
            out = batched_index_padded(inp)


class CompileBatchedIndexPaddedTestCase(unittest.TestCase):

    @staticmethod
    def test_compile_batched_index_padded_size_hint():
        @torch.compile(fullgraph=True)
        def proxy(
            x: torch.Tensor,
            *,
            out: torch.Tensor,
        ):
            return batched_index_padded(x, out=out)

        inp = torch.tensor([[False, True], [True, False], [True, True], [False, False]])
        out = torch.tensor([[1, -1], [0, -1], [0, 1], [-1, -1]])
        out_hint = torch.zeros_like(out)

        assert_eq(
            proxy(inp, out=out_hint), out
        )

    def test_compile_batched_index_padded_no_args(
        self
    ):
        @torch.compile(fullgraph=True)
        def proxy(
            x: torch.Tensor,
        ):
            return batched_index_padded(x)

        inp = torch.tensor([[False, True], [True, False], [True, True], [False, False]])
        out = torch.tensor([[1, -1], [0, -1], [0, 1], [-1, -1]])

        self.assertRaises(
            Unsupported, lambda: proxy(inp)
        )


if __name__ == "__main__":
    unittest.main()
