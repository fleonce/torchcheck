#include <torch/extension.h>
#include <vector>
#include "index.h"
#include "asserts.h"
#include "pybind/torch_dtype.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("assert_shape", &assert_shape, "Verify shapes");
    m.def("assert_dtype", &assert_dtype, "Verify dtype");
    m.def("assert_dim", &assert_dim, "Verify dim");
    m.def("assert_true", &assert_true, "Verify condition");
    m.def("assert_eq", &assert_eq, "Verify shapes, dtype, dim and values");
    m.def("batched_index_gen", &batched_index_gen, "Get a tensor of indices from a boolean mask");
}