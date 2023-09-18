#include "asserts.h"

int64_t assert_shape(
        const torch::Tensor &x,
        const at::IntArrayRef &shape) {
    TORCH_CHECK(
            x.sizes() == shape,
            "Expected shape of tensor to be: ", shape, " but got: ", x.sizes());
    return 0;
}

int64_t assert_dtype(
        const torch::Tensor &x,
        const torch::Dtype &dtype) {
    TORCH_CHECK(
            x.dtype() == dtype,
            "Expected dtype of tensor to be: ", dtype, " but got: ", x.dtype());
    return 0;
}

int64_t assert_dim(
        const torch::Tensor &x,
        const int64_t &dim) {
    TORCH_CHECK(
            x.dim() == dim,
            "Expected tensor dim to be: ", dim, "but got: ", x.dim())
    return 0;
}
