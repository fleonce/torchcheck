# TorchCheck

A simple [PyTorch](https://github.com/pytorch/pytorch) add-on library that contains additional helping functions
that extend beyond the functionality given by PyTorch.

## Function Overview

`torchcheck.batched_index_padded(self: Tensor, pad_value: int, **kwargs) -> Tensor:`
- Given a boolean mask, return the indices where the mask is `True`, batched in dimension `0`, padded with `pad_value`

```python
import torch
import torchcheck

mask = torch.tensor([[0, 0, 1], [0, 1, 1]]).bool()
indices = torchcheck.batched_index_padded(mask)
indices
```
```text
>>> tensor([[2, -1], [1, 2]])
```

`torchcheck.batched_masked_select(self: Tensor, mask: Tensor, pad_value: int | float = -1) -> Tensor:`
- Given a tensor and a batched boolean mask, select the elements of `self` where `mask` is true

## More ops

Helper functions are contained in the subpackage``torchcheck.ops.``

`torchcheck.ops.xyz`
- TODO

## Installation:

Install via pip
````shell
pip install torchcheck
````

or install from source:

```shell
pip install git+https://github.com/fleonce/torchcheck.git
```
