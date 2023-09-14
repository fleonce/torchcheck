#ifndef TORCHCHECK_SHAPE_H
#define TORCHCHECK_SHAPE_H

#include <torch/types.h>
#include <vector>

int64_t verify_shape(
        const torch::Tensor &x,
        const at::IntArrayRef &shape);

#endif //TORCHCHECK_SHAPE_H
