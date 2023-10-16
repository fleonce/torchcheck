import unittest
import torch
from torchcheck import batched_index_gen, assert_shape, assert_eq


def _batched_index_gen(mask: torch.Tensor, invalid_index=-1) -> torch.Tensor:
    assert mask.dtype == torch.bool, mask.dtype
    # assert mask.is_cuda
    # input has shape (bs, seq_len)
    # output has shape (bs, n_indices)
    bs, seq_len = mask.shape[:2]
    n_indices = max(1, mask.sum(dim=-1).amax())
    dims = mask.shape
    out_dims = (dims[:-1]) + (n_indices,)
    indices = (torch.arange(seq_len, device=mask.device).expand(bs, -1)).clone()
    upper_bound = dims[-1]
    indices[~mask] = upper_bound
    indices = torch.cat((indices, torch.full(out_dims, upper_bound, device=mask.device)), dim=-1)
    topk, _ = indices.topk(k=n_indices, largest=False)
    topk[topk == upper_bound] = invalid_index
    assert topk.shape == out_dims
    return topk


class BatchedTestCase(unittest.TestCase):

    @staticmethod
    def test_batched_index_gen():
        for _ in range(1000):
            rand_bool = torch.randint(2, (1, 16), dtype=torch.bool)
            gold_mask = _batched_index_gen(rand_bool, -1)
            cpp_mask = batched_index_gen(rand_bool)

            assert_eq(cpp_mask, gold_mask)

    @staticmethod
    def test_min_size():
        inp = torch.zeros((1, 10), dtype=torch.bool)
        assert_eq(batched_index_gen(inp, min_size=1), torch.full((1, 1), -1, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
