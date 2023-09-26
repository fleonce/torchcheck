#ifndef TORCHCHECK_ASSERTS_H
#define TORCHCHECK_ASSERTS_H

#include <torch/types.h>
#include <vector>

int64_t assert_shape(
        const torch::Tensor &x,
        const at::IntArrayRef &shape);

int64_t assert_dtype(
        const torch::Tensor &x,
        const torch::Dtype &dtype);

int64_t assert_dim(
        const torch::Tensor &x,
        const int64_t &dim);

int64_t assert_true(
        const bool &x,
        const char * msg);

int64_t assert_eq(
        const torch::Tensor &x,
        const torch::Tensor &y);


#endif //TORCHCHECK_ASSERTS_H
