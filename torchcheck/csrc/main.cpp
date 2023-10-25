#include <torch/extension.h>
#include <vector>
#include "index.h"
#include "asserts.h"
#include "pybind/torch_dtype.h"
#include "attention/attention_t5.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("assert_shape", &assert_shape, "Verify shapes");
    m.def("assert_dtype", &assert_dtype, "Verify dtype");
    m.def("assert_dim", &assert_dim, "Verify dim");
    m.def("assert_true", &assert_true, "Verify condition");
    m.def("assert_eq", &assert_eq, "Verify shapes, dtype, dim and values");
    m.def("batched_index_gen", &batched_index_gen, py::arg("x"), py::kw_only(), py::arg("min_size") = py::none(), "Get a tensor of indices from a boolean mask");
    m.def("batched_masked_select", &batched_masked_select,
            py::arg("x"),
            py::arg("mask"),
            py::kw_only(),
            py::arg("min_size") = py::none(),
            "Get a tuple of values and a mask from an input and a mask");
    m.def("t5_self_attention", &t5_self_attention, "T5 self attention");
}