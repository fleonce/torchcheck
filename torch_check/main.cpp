#include <torch/extension.h>
#include <vector>
#include "index.h"
#include "shape.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("verify_shape", &verify_shape, "Verify shapes");
    m.def("batched_index_gen", &batched_index_gen, "Get a tensor of indices from a boolean mask");
}