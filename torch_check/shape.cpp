#include "shape.h"

int64_t verify_shape(
        const torch::Tensor &x,
        const at::IntArrayRef &shape) {
    TORCH_CHECK(
            x.sizes() == shape,
            "Expected shape of tensor to be: ", shape, " but got: ", x.sizes());
    return 0;
}