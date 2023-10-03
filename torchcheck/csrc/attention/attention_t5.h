#pragma once

#include <torch/types.h>
#include "../asserts.h"

#pragma message "hi from T5"

torch::Tensor t5_self_attention(
        const torch::Tensor &query,
        const torch::Tensor &key,
        const torch::Tensor &value,
        const torch::Tensor &attn_mask,
        const int64_t &num_heads) {
    TORCH_CHECK(query.dim() == 3, "Expected query dim to be >= 3, but got ", query.dim());
    TORCH_CHECK(key.dim() == 3, "Expected key dim to be >= 3, but got ", key.dim());
    TORCH_CHECK(value.dim() == 3, "Expected value dim to be >= 3, but got ", value.dim());
    TORCH_CHECK(attn_mask.dtype() == torch::kBool, "Expected attn mask dtype ", torch::kBool, " but got ", attn_mask.dtype());
    auto q_sizes = query.sizes();
    auto bs = q_sizes[0];
    auto seq_len = q_sizes[1];
    auto inner_dim = q_sizes[2];
    auto head_dim = inner_dim / num_heads;
    TORCH_CHECK(inner_dim % num_heads == 0, "Expected num_heads to be a divisor of head_dim = ",
                head_dim, " but its not: ", inner_dim % num_heads);
    auto q = query.view({bs, seq_len, num_heads, head_dim}).transpose(1, 2);
    auto k = key.view({bs, seq_len, num_heads, head_dim}).transpose(1, 2);
    auto v = value.view({bs, seq_len, num_heads, head_dim}).transpose(1, 2);

    auto attention = torch::scaled_dot_product_attention(q, k, v, attn_mask);

    auto o = attention.transpose(1, 2).contiguous().view({bs, seq_len, inner_dim});
    return o;
}