#ifndef TORCHCHECK_INDEX_H
#define TORCHCHECK_INDEX_H

#include <torch/types.h>
#include <vector>

torch::Tensor batched_index_gen(
        const torch::Tensor &mask, const c10::optional<int64_t> & min_size = c10::nullopt);

#endif //TORCHCHECK_INDEX_H
