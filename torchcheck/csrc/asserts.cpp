#include "asserts.h"
#include <iostream>

bool does_symint_match_symint(const at::SymInt &s1, const at::SymInt &s2) {
    auto opt_int1 = s1.maybe_as_int();
    auto opt_int2 = s2.maybe_as_int();
    if (opt_int1.has_value() && opt_int2.has_value()) {
        return opt_int1.value() == opt_int2.value();
    }
    if (opt_int1.has_value() or opt_int2.has_value()) {
        return false;
    }
    // both symbolic ints
    return C10_LIKELY(str(s1) == str(s2));
}


int64_t assert_shape(
        const torch::Tensor &x,
        const at::SymIntArrayRef &shape) {
    const auto sym_sizes = x.sym_sizes();
    TORCH_CHECK(sym_sizes.size() == shape.size(), "Expected shape to be of size ", x.dim(), " but got: ", shape.size());
    for(auto i = 0; i < x.dim(); i++) {
        TORCH_CHECK(
            does_symint_match_symint(sym_sizes[i], shape[i]),
            "Expected shape of tensor to be: ", shape, " but got: ", x.sym_sizes(), " in dimension ", i
        );
    }
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

int64_t assert_true(
        const bool &x,
        const char * msg) {
    TORCH_CHECK(x, msg);
    return 0;
}

int64_t assert_eq(
        const torch::Tensor &x,
        const torch::Tensor &y) {
    assert_dim(y, x.dim());
    // assert_shape(y, x.sizes());
    assert_dtype(y, x.scalar_type());
    TORCH_CHECK(torch::equal(x, y), "Expected tensors x and y to match: ", x, y);
    return 0;
}
