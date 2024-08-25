import itertools
import unittest
from functools import partial, wraps
from typing import Any

import torch
from torch import Tensor


def foreach(**setup: list[Any] | set[Any]):
    def inner(func):
        @wraps(func)
        def wrapper(self, **kwargs):
            combinations = []
            for arg, values in setup.items():
                values = [(arg, value) for value in values]
                combinations.append(values)
            combinations = itertools.product(*combinations)

            for combination in combinations:
                combination_kwargs = {arg: value for arg, value in combination}

                with self.subTest(**combination_kwargs):
                    func(self, **combination_kwargs, **kwargs)
        return wrapper
    return inner


def patched_add_sub_test(self: unittest.TestResult, test, subtest, err, *, orig):
    orig(test, subtest, err)
    self.testsRun += 1


class SubtestCountingTestCase(unittest.TestCase):

    def run(self, result=None):
        if not isinstance(getattr(result, "addSubTest"), partial):
            patched_func = partial(patched_add_sub_test, result, orig=result.addSubTest)
            setattr(result, "addSubTest", patched_func)
        super().run(result)


class BasicTestCase(SubtestCountingTestCase):
    def assertTensorEqual(self, first: Tensor, second: Tensor):
        self.assertEqual(first.size(), second.size(), "Expected size of tensors to match")
        self.assertEqual(first.dtype, second.dtype, "Expected dtype of tensors to match")
        self.assertTrue(first.equal(second), "Expected elements of tensors to match: " + str(first) + " vs " + str(second))

    @staticmethod
    def create_random_dtype_tensor(shape: tuple[int, ...], dtype: torch.dtype):
        if dtype.is_floating_point:
            return torch.randn(shape, dtype=dtype)
        return torch.empty(shape, dtype=dtype)
