import math
import unittest

import torch
import torchcheck


class T5AttentionTestCase(unittest.TestCase):

    def test_self_attention(self):
        num_heads = 4
        seq_len = 2
        per_head_dim = 256 // num_heads
        q = k = v = torch.randn((1, seq_len, 256), requires_grad=True)
        attn_mask = torch.ones((1, seq_len, seq_len), dtype=torch.bool)
        cpp_attn = torchcheck.C.t5_self_attention(q.clone(), k.clone(), v.clone(), attn_mask, num_heads)

        q = q.view(1, seq_len, num_heads, per_head_dim).transpose(1, 2)
        k = k.view(1, seq_len, num_heads, per_head_dim).transpose(1, 2)
        v = v.view(1, seq_len, num_heads, per_head_dim).transpose(1, 2)

        attn_mask = attn_mask.masked_fill(~attn_mask, float("-inf"))
        attn = (torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1)) + attn_mask).softmax(dim=-1) @ v
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(1, seq_len, 256)
        self.assertTrue(torch.allclose(attn, cpp_attn))
        self.assertRaises(RuntimeError, lambda: torchcheck.C.t5_self_attention(
            q.clone(), k.clone(), v.clone(), attn_mask.long(), num_heads))
        self.assertTrue(cpp_attn.requires_grad)
        self.assertIsNone(cpp_attn.sum().backward())
