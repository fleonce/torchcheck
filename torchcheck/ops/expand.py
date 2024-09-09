from torch import Tensor, SymInt


def maybe_wrap_dim(tensor: Tensor, dim: int | SymInt) -> int | SymInt:
    min_dim = -tensor.dim()
    max_dim = tensor.dim() + 1
    if dim < min_dim or dim > max_dim:
        raise ValueError(
            f"Dimension out of range (expected to be in range of [{min_dim}, {max_dim}], but got {dim}"
        )
    if dim < 0:
        return dim + tensor.dim()
    return dim


def expand_dim(tensor: Tensor, dim: int, expand_size: int) -> Tensor:
    expands = [-1] * tensor.dim()
    wrap_dim = maybe_wrap_dim(tensor, dim)
    expands[wrap_dim] = expand_size
    return tensor.expand(expands)
