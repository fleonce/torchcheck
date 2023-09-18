#include <torch/extension.h>
#include <vector>
#include "index.h"
#include "asserts.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("assert_shape", &assert_shape, "Verify shapes");
    m.def("assert_dtype", &assert_dtype, "Verify dtype");
    m.def("assert_dim", &assert_dim, "Verify dim");
    m.def("batched_index_gen", &batched_index_gen, "Get a tensor of indices from a boolean mask");
}