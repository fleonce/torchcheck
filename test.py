import torch
from torchcheck import assert_shape, assert_dtype, batched_index_gen


a = torch.randint(2, (10, 10), dtype=torch.bool)
b = batched_index_gen(a)

assert_shape(a, (10, 10))
assert_dtype(a, torch.bool)
assert_shape(b, (10, 10))

print(b.shape)
print((b != -1).sum(dim=-1).max())
