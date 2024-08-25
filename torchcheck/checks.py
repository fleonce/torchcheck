from typing import Sequence, Union, Callable

import torch


_dim_type = Union[int, torch.Tensor, torch.SymInt]


def assert_shape(self: torch.Tensor, shape: Sequence[_dim_type]):
    if self.shape != shape:
        raise RuntimeError(f"Expected shape of tensor to be: {shape} but got: {self.shape}")


def assert_dtype(self: torch.Tensor, dtype: torch.dtype):
    if self.dtype != dtype:
        raise RuntimeError(f"Expected dtype of tensor to be: {dtype} but got: {self.dtype}")


def assert_dim(self: torch.Tensor, dim: _dim_type):
    if self.dim() != dim:
        raise RuntimeError(f"Expected dim of tensor to be: {dim} but got: {self.dim()}")


def tensor_repr(tensor: torch.Tensor):
    return "tensor(dim=" + repr(tensor.dim()) + ", shape=" + repr(tuple(tensor.shape)) + ")"


def assert_eq(self: torch.Tensor, ref: torch.Tensor):
    if not self.equal(ref):
        raise RuntimeError(f"Expected tensors " + tensor_repr(self) + " and " + tensor_repr(ref) + " to be equal")


def assert_true(cond: bool, message: str | Callable[[], str]):
    if not cond:
        if callable(message):
            message = message()
        raise RuntimeError(message)

