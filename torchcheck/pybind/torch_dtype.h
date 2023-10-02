#pragma once

#include <torch/csrc/python_headers.h>
#include <torch/csrc/Dtype.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace pybind11 {
    namespace detail {
        template<>
        struct TORCH_PYTHON_API type_caster<torch::Dtype> {
        public:
        PYBIND11_TYPE_CASTER(torch::Dtype, _("torch.dtype"));

            bool load(handle src, bool) {
                PyObject * obj = src.ptr();
                if (THPDtype_Check(obj)) {
                    value = reinterpret_cast<THPDtype *>(obj)->scalar_type;
                    return true;
                }
                return false;
            }

            static handle cast(
                    torch::Dtype src,
                    return_value_policy,
                    handle) {
                return handle((PyObject * )
                torch::getTHPDtype(src));
            }
        };
    };
};

