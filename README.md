# TorchCheck

A simple PyTorch library written in c++ to add assert statements for shapes:

```python
import torch
from torch_check import verify_shape as assert_shape

x = torch.randn((30, 30))
# will not fail
assert_shape(x, (30, 30))
# will fail:
assert_shape(x, (60, 30))
```

The benefit of writing the code in c++ is that stacktraces in python then reference to the location where the function
has been called, not where the actual shape verification has failed

## Installation:

Make sure to install torch first

```shell
pip install torch
pip install git+https://github.com/fleonce/torchcheck.git
```