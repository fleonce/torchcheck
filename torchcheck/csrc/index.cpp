#include "index.h"

torch::Tensor batched_index_gen(
        const torch::Tensor &mask, const c10::optional<int64_t> & min_size) {
    TORCH_CHECK(mask.dtype() == at::kBool, "Expected a boolean mask but got ", mask.dtype());
    TORCH_CHECK(mask.dim() == 2, "Expected a 2D tensor, but got a ", mask.dim(), "D tensor!");
    TORCH_CHECK(min_size.value_or(0) >= 0, "Expected min_size >= 0, but got ", min_size.value_or(0));
    TORCH_CHECK(min_size.value_or(0) <= mask.size(-1), "Expected min_size <= ", mask.size(-1), " but got ", min_size.value_or(0));
    auto max_indices = mask.to(at::kLong).sum({-1}).max();
    max_indices.clamp_min_(torch::Scalar(min_size.value_or(0)));

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


std::tuple<torch::Tensor, torch::Tensor> batched_masked_select(
        const torch::Tensor &x,
        const torch::Tensor &mask,
        const c10::optional<int64_t> & min_size) {
    TORCH_CHECK(x.dim() == 2, "Expected dim of self to be 2 but got ", x.dim());
    TORCH_CHECK(mask.dim() == x.dim(), "Expected dim of mask to match self.dim(), but got ", mask.dim(), " vs. ", x.dim());
    TORCH_CHECK(min_size.value_or(0) >= 0, "Expected min_size >= 0, but got ", min_size.value_or(0));
    TORCH_CHECK(min_size.value_or(0) <= mask.size(-1), "Expected min_size <= ", mask.size(-1), " but got ", min_size.value_or(0));

    auto max_indices = mask.to(at::kLong).sum({-1}).max();
    max_indices.clamp_min_(torch::Scalar(min_size.value_or(0)));

    auto sizes = mask.sizes();
    auto bs = sizes[0];
    auto seq_len = sizes[1];
    auto indices = torch::arange(seq_len, at::TensorOptions().device(mask.device())).
            expand({bs, -1}).clone();
    indices.masked_fill_(~mask, seq_len);

    auto [ignored, top_k_indices] = indices.topk(max_indices.item().toLong(), -1, false);
    auto mask_out = torch::gather(mask, -1, top_k_indices);
    auto values = torch::gather(x, -1, top_k_indices);
    return std::make_tuple(values, mask_out);
}
