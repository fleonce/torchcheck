import unittest
import torch
from torchcheck import *


class AssertTestCase(unittest.TestCase):

    def test_assert_shape(self):
        x = torch.randn((64,), dtype=torch.float32)
        self.assertRaises(RuntimeError, lambda: assert_shape(x, (32,)))
        self.assertEqual(assert_shape(x, (64,)), 0)

    def test_assert_dim(self):
        x = torch.randn((64,))
        self.assertRaises(RuntimeError, lambda: assert_dim(x, 2))
        self.assertEqual(0, assert_dim(x, 1))

    def test_assert_eq(self):
        x = torch.randn((64,))
        y = torch.randn((64,))
        self.assertRaises(RuntimeError, lambda: assert_eq(x, y))
        self.assertEqual(0, assert_eq(x, x))
        self.assertEqual(0, assert_eq(y, y))
