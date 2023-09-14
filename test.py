import torch
import torch_check

a = torch.randint(2, (10, 10), dtype=torch.bool)
b = torch_check.batched_index_gen(a)

print(b.shape)
print((b != -1).sum(dim=-1).max())
