// Fused decode kernels for the Qwen3 text decoder. The dot-product arithmetic is
// byte-for-byte the MLX 0.32.3 qmv_fast body (same lane->k mapping, same block
// order, same simd_sum); only the scheduling (which matrix a threadgroup reads)
// and the epilogues (residual add, silu*mul) are new, so results stay bit-exact.
#include <metal_math>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

using namespace metal;

// One qmv_fast threadgroup: 2 simdgroups x 4 rows = 8 output rows of one matrix.
// out_row_local selects the weight rows, out_row_global the output rows.
template <typename T, int group_size, int bits, bool ADD_RES, int NSG = 2, int RPS = 4>
METAL_FUNC void dec_qmv_body(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    const device T* res,
    device T* y,
    int out_row_local,
    int out_row_global,
    int in_vec_size,
    int out_vec_size,
    uint tid_x,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int results_per_simdgroup = RPS;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = out_row_local + simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid_x * in_vec_size + simd_lid * values_per_thread;
  y += tid_x * out_vec_size + out_row_global + simd_gid * results_per_simdgroup;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;
      U s = sl[0];
      U b = bl[0];
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      T v = static_cast<T>(result[row]);
      if (ADD_RES) {
        const int rrow = out_row_global + simd_gid * results_per_simdgroup + row;
        v = static_cast<T>(static_cast<float>(v) + static_cast<float>(res[rrow]));
      }
      y[row] = v;
    }
  }
}

// Fused q/k/v (or any 3 stacked matrices) in one dispatch: tile tid.y picks the
// matrix, so each row keeps exactly the qmv_fast arithmetic.
template <typename T, int group_size, int bits, int NSG = 2, int RPS = 4>
[[kernel]] void dec_qmv3(
    const device uint32_t* w0 [[buffer(0)]], const device T* s0 [[buffer(1)]],
    const device T* b0 [[buffer(2)]], const device uint32_t* w1 [[buffer(3)]],
    const device T* s1 [[buffer(4)]], const device T* b1 [[buffer(5)]],
    const device uint32_t* w2 [[buffer(6)]], const device T* s2 [[buffer(7)]],
    const device T* b2 [[buffer(8)]], const device T* x [[buffer(9)]],
    device T* y [[buffer(10)]], constant int& in_vec_size [[buffer(11)]],
    constant int& out_vec_size [[buffer(12)]], constant int& n0 [[buffer(13)]],
    constant int& n1 [[buffer(14)]], uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int row8 = (int)tid.y * (NSG * RPS);
  const device uint32_t* wp = w0;
  const device T* sp = s0;
  const device T* bp = b0;
  int local = row8;
  if (row8 >= n0) {
    if (row8 >= n0 + n1) {
      wp = w2; sp = s2; bp = b2; local = row8 - n0 - n1;
    } else {
      wp = w1; sp = s1; bp = b1; local = row8 - n0;
    }
  }
  dec_qmv_body<T, group_size, bits, false, NSG, RPS>(wp, sp, bp, x, y, y, local, row8,
                                                     in_vec_size, out_vec_size, tid.x, simd_gid,
                                                     simd_lid);
}

template <typename T, int group_size, int bits, int NSG = 2, int RPS = 4>
[[kernel]] void dec_qmv2(
    const device uint32_t* w0 [[buffer(0)]], const device T* s0 [[buffer(1)]],
    const device T* b0 [[buffer(2)]], const device uint32_t* w1 [[buffer(3)]],
    const device T* s1 [[buffer(4)]], const device T* b1 [[buffer(5)]],
    const device T* x [[buffer(6)]], device T* y [[buffer(7)]],
    constant int& in_vec_size [[buffer(8)]], constant int& out_vec_size [[buffer(9)]],
    constant int& n0 [[buffer(10)]], uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int row8 = (int)tid.y * (NSG * RPS);
  const device uint32_t* wp = w0;
  const device T* sp = s0;
  const device T* bp = b0;
  int local = row8;
  if (row8 >= n0) { wp = w1; sp = s1; bp = b1; local = row8 - n0; }
  dec_qmv_body<T, group_size, bits, false, NSG, RPS>(wp, sp, bp, x, y, y, local, row8,
                                                     in_vec_size, out_vec_size, tid.x, simd_gid,
                                                     simd_lid);
}

// qmv_fast + residual add in the epilogue (round the dot product to bf16 first,
// then add in bf16 exactly like MLX's binary add).
template <typename T, int group_size, int bits, int NSG = 2, int RPS = 4>
[[kernel]] void dec_qmv_add(
    const device uint32_t* w [[buffer(0)]], const device T* s [[buffer(1)]],
    const device T* b [[buffer(2)]], const device T* x [[buffer(3)]],
    const device T* res [[buffer(4)]], device T* y [[buffer(5)]],
    constant int& in_vec_size [[buffer(6)]], constant int& out_vec_size [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int row8 = (int)tid.y * (NSG * RPS);
  dec_qmv_body<T, group_size, bits, true, NSG, RPS>(w, s, b, x, res, y, row8, row8, in_vec_size,
                                                    out_vec_size, tid.x, simd_gid, simd_lid);
}

// silu(gate) * up in one pass; the silu result is rounded to bf16 first (MLX
// compiled silu), then multiplied in bf16 (MLX binary multiply).
template <typename T>
[[kernel]] void dec_silu_mul(const device T* g [[buffer(0)]], const device T* u [[buffer(1)]],
                             device T* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  T v = g[gid];
  T y = T(1) / (T(1) + metal::precise::exp(metal::abs(v)));
  T s = (v < T(0)) ? y : T(1) - y;
  T sg = v * s;
  out[gid] = static_cast<T>(static_cast<float>(sg) * static_cast<float>(u[gid]));
}

// clang-format off
instantiate_kernel("dec_qmv3_bf16_gs64_b6", dec_qmv3, bfloat16_t, 64, 6, 2, 4)
instantiate_kernel("dec_qmv2_bf16_gs64_b6", dec_qmv2, bfloat16_t, 64, 6, 2, 4)
instantiate_kernel("dec_qmv_add_bf16_gs64_b6", dec_qmv_add, bfloat16_t, 64, 6, 2, 4)
instantiate_kernel("dec_qmv3_s42_bf16_gs64_b6", dec_qmv3, bfloat16_t, 64, 6, 4, 2)
instantiate_kernel("dec_qmv2_s42_bf16_gs64_b6", dec_qmv2, bfloat16_t, 64, 6, 4, 2)
instantiate_kernel("dec_qmv_add_s42_bf16_gs64_b6", dec_qmv_add, bfloat16_t, 64, 6, 4, 2)
instantiate_kernel("dec_qmv3_s81_bf16_gs64_b6", dec_qmv3, bfloat16_t, 64, 6, 8, 1)
instantiate_kernel("dec_qmv2_s81_bf16_gs64_b6", dec_qmv2, bfloat16_t, 64, 6, 8, 1)
instantiate_kernel("dec_qmv_add_s81_bf16_gs64_b6", dec_qmv_add, bfloat16_t, 64, 6, 8, 1)
instantiate_kernel("dec_qmv3_s82_bf16_gs64_b6", dec_qmv3, bfloat16_t, 64, 6, 8, 2)
instantiate_kernel("dec_qmv2_s82_bf16_gs64_b6", dec_qmv2, bfloat16_t, 64, 6, 8, 2)
instantiate_kernel("dec_qmv_add_s82_bf16_gs64_b6", dec_qmv_add, bfloat16_t, 64, 6, 8, 2)
instantiate_kernel("dec_qmv3_s44_bf16_gs64_b6", dec_qmv3, bfloat16_t, 64, 6, 4, 4)
instantiate_kernel("dec_qmv2_s44_bf16_gs64_b6", dec_qmv2, bfloat16_t, 64, 6, 4, 4)
instantiate_kernel("dec_qmv_add_s44_bf16_gs64_b6", dec_qmv_add, bfloat16_t, 64, 6, 4, 4)
instantiate_kernel("dec_qmv3_bf16_gs64_b4", dec_qmv3, bfloat16_t, 64, 4, 2, 4)
instantiate_kernel("dec_qmv2_bf16_gs64_b4", dec_qmv2, bfloat16_t, 64, 4, 2, 4)
instantiate_kernel("dec_qmv_add_bf16_gs64_b4", dec_qmv_add, bfloat16_t, 64, 4, 2, 4)
instantiate_kernel("dec_qmv3_s42_bf16_gs64_b4", dec_qmv3, bfloat16_t, 64, 4, 4, 2)
instantiate_kernel("dec_qmv2_s42_bf16_gs64_b4", dec_qmv2, bfloat16_t, 64, 4, 4, 2)
instantiate_kernel("dec_qmv_add_s42_bf16_gs64_b4", dec_qmv_add, bfloat16_t, 64, 4, 4, 2)
instantiate_kernel("dec_qmv3_s81_bf16_gs64_b4", dec_qmv3, bfloat16_t, 64, 4, 8, 1)
instantiate_kernel("dec_qmv2_s81_bf16_gs64_b4", dec_qmv2, bfloat16_t, 64, 4, 8, 1)
instantiate_kernel("dec_qmv_add_s81_bf16_gs64_b4", dec_qmv_add, bfloat16_t, 64, 4, 8, 1)
instantiate_kernel("dec_qmv3_s82_bf16_gs64_b4", dec_qmv3, bfloat16_t, 64, 4, 8, 2)
instantiate_kernel("dec_qmv2_s82_bf16_gs64_b4", dec_qmv2, bfloat16_t, 64, 4, 8, 2)
instantiate_kernel("dec_qmv_add_s82_bf16_gs64_b4", dec_qmv_add, bfloat16_t, 64, 4, 8, 2)
instantiate_kernel("dec_qmv3_s44_bf16_gs64_b4", dec_qmv3, bfloat16_t, 64, 4, 4, 4)
instantiate_kernel("dec_qmv2_s44_bf16_gs64_b4", dec_qmv2, bfloat16_t, 64, 4, 4, 4)
instantiate_kernel("dec_qmv_add_s44_bf16_gs64_b4", dec_qmv_add, bfloat16_t, 64, 4, 4, 4)
instantiate_kernel("dec_silu_mul_bf16", dec_silu_mul, bfloat16_t)
// clang-format on
