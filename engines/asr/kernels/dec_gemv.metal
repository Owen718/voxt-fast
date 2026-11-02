// Quantized (6-bit affine, group 64) matmul kernels for the Qwen3 text decoder.
// Ported verbatim from MLX 0.32.3 (kernels/quantized.h) and instantiated with the same
// names/parameters the MLX dispatcher picks for this model's shapes:
//   decode (M == 1)      -> affine_qmv_fast_*   (MLX quantized.cpp qmv(), fast path)
//   prefill (M >= 2)     -> affine_qmm_t_*      (MLX quantized.cpp qmm(), transpose path)
//   embeddings (rows)    -> affine_dequantize_* (MLX dequantize)
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

using namespace metal;

// clang-format off
instantiate_kernel("affine_qmv_fast_bfloat16_t_gs_64_b_6_batch_0",
                   affine_qmv_fast, bfloat16_t, 64, 6, false)
instantiate_kernel("affine_qmm_t_bfloat16_t_gs_64_b_6_alN_true_batch_0",
                   affine_qmm_t, bfloat16_t, 64, 6, true, false)
instantiate_kernel("affine_qmm_t_splitk_bfloat16_t_gs_64_b_6_alN_true",
                   affine_qmm_t_splitk, bfloat16_t, 64, 6, true)
instantiate_kernel("affine_dequantize_bfloat16_t_gs_64_b_6",
                   affine_dequantize, bfloat16_t, 64, 6)
// Qwen3-ASR 0.6B checkpoints are 4-bit.
instantiate_kernel("affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0",
                   affine_qmv_fast, bfloat16_t, 64, 4, false)
instantiate_kernel("affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0",
                   affine_qmm_t, bfloat16_t, 64, 4, true, false)
instantiate_kernel("affine_qmm_t_splitk_bfloat16_t_gs_64_b_4_alN_true",
                   affine_qmm_t_splitk, bfloat16_t, 64, 4, true)
instantiate_kernel("affine_dequantize_bfloat16_t_gs_64_b_4",
                   affine_dequantize, bfloat16_t, 64, 4)
// clang-format on

// Gather packed rows of a 6-bit affine weight (embedding table) for a list of token ids.
// Produces contiguous [n, K*6/32] uint32 rows plus [n, K/64] scales/biases, exactly the
// arrays MLX's take(weight/scales/biases, ids) hands to the dequantize kernel.
[[kernel]] void dec_gather_rows(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device int32_t* ids [[buffer(3)]],
    device uint32_t* w_out [[buffer(4)]],
    device bfloat16_t* s_out [[buffer(5)]],
    device bfloat16_t* b_out [[buffer(6)]],
    constant int& n [[buffer(7)]],
    constant int& packed_per_row [[buffer(8)]],
    constant int& groups_per_row [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.y) >= n) {
    return;
  }
  const int row = ids[gid.y];
  const int i = gid.x;
  if (i < packed_per_row) {
    w_out[(size_t)gid.y * packed_per_row + i] =
        w[(size_t)row * packed_per_row + i];
  } else {
    const int g = i - packed_per_row;
    s_out[(size_t)gid.y * groups_per_row + g] =
        scales[(size_t)row * groups_per_row + g];
    b_out[(size_t)gid.y * groups_per_row + g] =
        biases[(size_t)row * groups_per_row + g];
  }
}
