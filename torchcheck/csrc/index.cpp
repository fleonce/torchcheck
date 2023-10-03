#include "index.h"

torch::Tensor batched_index_gen(
        const torch::Tensor &mask) {
    TORCH_CHECK(mask.dtype() == at::kBool, "Expected a boolean mask but got ", mask.dtype());
    TORCH_CHECK(mask.dim() == 2, "Expected a 2D tensor, but got a ", mask.dim(), "D tensor!");
    auto max_indices = mask.to(at::kLong).sum({-1}).max();
    max_indices.clamp_min(at::Scalar(0));

    auto sizes = mask.sizes();
    auto bs = sizes[0];
    auto seq_len = sizes[1];
    auto indices = torch::arange(seq_len, at::TensorOptions().device(mask.device())).
            expand({bs, -1}).clone();
    indices.masked_fill_(~mask, seq_len);
    auto [top_k, ignored] = indices.topk(max_indices.item().toLong(), -1, false);
    top_k.masked_fill_(top_k.eq(torch::Scalar(seq_len)), torch::Scalar(-1));
    TORCH_CHECK(top_k.dim() == 2, "Expected output dim = 2, but got ", top_k.dim());
    return top_k;
}