import unittest
import torch
from torchcheck import (
    assert_dtype,
    assert_eq,
    assert_dim,
    assert_shape,
    assert_true,
)


class AssertTestCase(unittest.TestCase):

    def test_assert_shape(self):
        x = torch.randn((64,), dtype=torch.float32)
        self.assertRaises(RuntimeError, lambda: assert_shape(x, (32,)))
        self.assertEqual(None, assert_shape(x, (64,)))

    def test_assert_dim(self):
        x = torch.randn((64,))
        self.assertRaises(RuntimeError, lambda: assert_dim(x, 2))
        self.assertEqual(None, assert_dim(x, 1))

    def test_assert_eq(self):
        x = torch.randn((64,))
        y = torch.randn((64,))
        self.assertRaises(RuntimeError, lambda: assert_eq(x, y))
        self.assertEqual(None, assert_eq(x, x))
        self.assertEqual(None, assert_eq(y, y))

    def test_assert_true(self):
        self.assertEqual(None, assert_true(True, ""))
        self.assertRaises(RuntimeError, lambda: assert_true(False, ""))

    def test_assert_dtype(self):
        x = torch.randn((64,), dtype=torch.float32)
        self.assertEqual(None, assert_dtype(x, torch.float32))
        self.assertRaises(RuntimeError, lambda: assert_dtype(x, torch.bool))
