#include <torch/extension.h>
#include <vector>


inline int64_t verify_shape(
        const torch::Tensor &x,
        const at::IntArrayRef &shape) {
    TORCH_CHECK(
            x.sizes() == shape,
            "Expected shape of tensor to be: ", shape, " but got: ", x.sizes());
    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("verify_shape", &verify_shape, "Verify shapes");
}