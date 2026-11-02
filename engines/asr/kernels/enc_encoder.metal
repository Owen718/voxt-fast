// Audio encoder kernels (owner: encoder worker).
// GEMM (float32 activations x bfloat16 weights, float32 accumulate), implicit-im2col conv2d,
// gelu, layer norm (MLX's kernel, instantiated in enc_mlx.metal), attention, packing.
#include <metal_stdlib>
using namespace metal;

// MLX's f32 erf (kernels/erf.h), the exact function the reference's Gelu uses.
#include "mlx/backend/metal/kernels/erf.h"

// ---------------------------------------------------------------------------
// C[M,N] = A[M,K] @ B[N,K]^T + bias[N]   (A f32, B bf16, C f32)
// Tile 64x64, K step 16, 256 threads, 4x4 outputs per thread.
// ---------------------------------------------------------------------------
constant int GEMM_BM = 64;
constant int GEMM_BN = 64;
constant int GEMM_BK = 16;

kernel void enc_gemm(device const float *A [[buffer(0)]],
                     device const bfloat *B [[buffer(1)]],
                     device const float *bias [[buffer(2)]],
                     device float *C [[buffer(3)]],
                     constant int &M [[buffer(4)]],
                     constant int &N [[buffer(5)]],
                     constant int &K [[buffer(6)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint2 tid [[thread_position_in_threadgroup]]) {
  const int m0 = int(tg.y) * GEMM_BM;
  const int n0 = int(tg.x) * GEMM_BN;
  threadgroup float As[GEMM_BK][GEMM_BM];
  threadgroup float Bs[GEMM_BK][GEMM_BN];
  float acc[4][4] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const int i0 = int(tid.x) + 16 * int(tid.y);  // 0..255
  const int row = i0 >> 2;                      // 0..63
  const int kk = (i0 & 3) * 4;                  // 0,4,8,12
  const int gm = m0 + row;
  const int gn = n0 + row;
  const int tx4 = int(tid.x) * 4;
  const int ty4 = int(tid.y) * 4;

  for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
    // A tile
    if (k0 + 4 <= K) {
      float4 v = float4(0.0f);
      if (gm < M) v = *(const device float4 *)(A + (size_t)gm * K + k0 + kk);
      As[kk + 0][row] = v.x;
      As[kk + 1][row] = v.y;
      As[kk + 2][row] = v.z;
      As[kk + 3][row] = v.w;
    } else {
      for (int t = 0; t < 4; ++t) {
        const int k = k0 + kk + t;
        As[kk + t][row] = (gm < M && k < K) ? A[(size_t)gm * K + k] : 0.0f;
      }
    }
    // B tile
    if (k0 + 4 <= K) {
      float4 v = float4(0.0f);
      if (gn < N) {
        const device bfloat *p = B + (size_t)gn * K + k0 + kk;
        v = float4(float(p[0]), float(p[1]), float(p[2]), float(p[3]));
      }
      Bs[kk + 0][row] = v.x;
      Bs[kk + 1][row] = v.y;
      Bs[kk + 2][row] = v.z;
      Bs[kk + 3][row] = v.w;
    } else {
      for (int t = 0; t < 4; ++t) {
        const int k = k0 + kk + t;
        Bs[kk + t][row] = (gn < N && k < K) ? float(B[(size_t)gn * K + k]) : 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (int k = 0; k < GEMM_BK; ++k) {
      const float a0 = As[k][tx4 + 0], a1 = As[k][tx4 + 1], a2 = As[k][tx4 + 2], a3 = As[k][tx4 + 3];
      const float b0 = Bs[k][ty4 + 0], b1 = Bs[k][ty4 + 1], b2 = Bs[k][ty4 + 2], b3 = Bs[k][ty4 + 3];
      acc[0][0] += a0 * b0; acc[0][1] += a0 * b1; acc[0][2] += a0 * b2; acc[0][3] += a0 * b3;
      acc[1][0] += a1 * b0; acc[1][1] += a1 * b1; acc[1][2] += a1 * b2; acc[1][3] += a1 * b3;
      acc[2][0] += a2 * b0; acc[2][1] += a2 * b1; acc[2][2] += a2 * b2; acc[2][3] += a2 * b3;
      acc[3][0] += a3 * b0; acc[3][1] += a3 * b1; acc[3][2] += a3 * b2; acc[3][3] += a3 * b3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const int mbase = m0 + tx4;
  const int nbase = n0 + ty4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (mbase + i >= M) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int n = nbase + j;
      if (n >= N) continue;
      C[(size_t)(mbase + i) * N + n] = acc[i][j] + bias[n];
    }
  }
}

// ---------------------------------------------------------------------------
// Conv2d, stride 2, pad 1, 3x3, NHWC: Y[m, o] = sum_k X[im2col(m,k)] * W[o,k] + bias[o]
// with m = oh * Wo + ow and k = (kh * 3 + kw) * C + c.
// ---------------------------------------------------------------------------
struct ConvParams {
  int H, W, C;    // input spatial size and channels
  int Ho, Wo;     // output spatial size
  int N;          // output channels
  int K;          // 9 * C
  int xs;         // input frame stride (W for a contiguous buffer)
};

kernel void enc_conv(device const float *X [[buffer(0)]],
                     device const bfloat *Wt [[buffer(1)]],
                     device const float *bias [[buffer(2)]],
                     device float *Y [[buffer(3)]],
                     constant ConvParams &p [[buffer(4)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint2 tid [[thread_position_in_threadgroup]]) {
  const int m0 = int(tg.y) * GEMM_BM;
  const int n0 = int(tg.x) * GEMM_BN;
  threadgroup float As[GEMM_BK][GEMM_BM];
  threadgroup float Bs[GEMM_BK][GEMM_BN];
  float acc[4][4] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const int i0 = int(tid.x) + 16 * int(tid.y);
  const int row = i0 >> 2;
  const int kk = (i0 & 3) * 4;
  const int gm = m0 + row;
  const int gn = n0 + row;
  const bool c4 = (p.C % 4) == 0;
  const int oh = gm / p.Wo, ow = gm - oh * p.Wo;
  const int tx4 = int(tid.x) * 4;
  const int ty4 = int(tid.y) * 4;

  for (int k0 = 0; k0 < p.K; k0 += GEMM_BK) {
    // A tile: 4 consecutive k for one output pixel.
    {
      const int kbase = k0 + kk;
      float4 v = float4(0.0f);
      if (gm < p.Ho * p.Wo) {
        if (c4) {
          const int tap = kbase / p.C, c = kbase - tap * p.C;
          const int kh = tap / 3, kw = tap - kh * 3;
          const int ih = oh * 2 - 1 + kh, iw = ow * 2 - 1 + kw;
          if (ih >= 0 && ih < p.H && iw >= 0 && iw < p.W)
            v = *(const device float4 *)(X + ((size_t)(ih * p.xs + iw) * p.C + c));
        } else {
          for (int t = 0; t < 4; ++t) {
            const int k = kbase + t;
            if (k >= p.K) break;
            const int tap = k / p.C, c = k - tap * p.C;
            const int kh = tap / 3, kw = tap - kh * 3;
            const int ih = oh * 2 - 1 + kh, iw = ow * 2 - 1 + kw;
            if (ih >= 0 && ih < p.H && iw >= 0 && iw < p.W)
              v[t] = X[(size_t)(ih * p.xs + iw) * p.C + c];
          }
        }
      }
      As[kk + 0][row] = v.x;
      As[kk + 1][row] = v.y;
      As[kk + 2][row] = v.z;
      As[kk + 3][row] = v.w;
    }
    // B tile
    if (k0 + kk + 4 <= p.K) {
      float4 v = float4(0.0f);
      if (gn < p.N) {
        const device bfloat *q = Wt + (size_t)gn * p.K + k0 + kk;
        v = float4(float(q[0]), float(q[1]), float(q[2]), float(q[3]));
      }
      Bs[kk + 0][row] = v.x;
      Bs[kk + 1][row] = v.y;
      Bs[kk + 2][row] = v.z;
      Bs[kk + 3][row] = v.w;
    } else {
      for (int t = 0; t < 4; ++t) {
        const int k = k0 + kk + t;
        Bs[kk + t][row] = (gn < p.N && k < p.K) ? float(Wt[(size_t)gn * p.K + k]) : 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (int k = 0; k < GEMM_BK; ++k) {
      const float a0 = As[k][tx4 + 0], a1 = As[k][tx4 + 1], a2 = As[k][tx4 + 2], a3 = As[k][tx4 + 3];
      const float b0 = Bs[k][ty4 + 0], b1 = Bs[k][ty4 + 1], b2 = Bs[k][ty4 + 2], b3 = Bs[k][ty4 + 3];
      acc[0][0] += a0 * b0; acc[0][1] += a0 * b1; acc[0][2] += a0 * b2; acc[0][3] += a0 * b3;
      acc[1][0] += a1 * b0; acc[1][1] += a1 * b1; acc[1][2] += a1 * b2; acc[1][3] += a1 * b3;
      acc[2][0] += a2 * b0; acc[2][1] += a2 * b1; acc[2][2] += a2 * b2; acc[2][3] += a2 * b3;
      acc[3][0] += a3 * b0; acc[3][1] += a3 * b1; acc[3][2] += a3 * b2; acc[3][3] += a3 * b3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const int mbase = m0 + tx4;
  const int nbase = n0 + ty4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (mbase + i >= p.Ho * p.Wo) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int n = nbase + j;
      if (n >= p.N) continue;
      Y[(size_t)(mbase + i) * p.N + n] = acc[i][j] + bias[n];
    }
  }
}

// ---------------------------------------------------------------------------
// Elementwise helpers
// ---------------------------------------------------------------------------
kernel void enc_gelu(device float *x [[buffer(0)]], constant uint &n [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float v = x[i];
  const float t = erf(v / 1.4142135623730951f);
  x[i] = (v * (1.0f + t)) / 2.0f;
}

kernel void enc_add(device float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                    constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) a[i] += b[i];
}

// conv3 output [H][W][C] -> [W][C*H] (freq fastest), as the reference's transpose+reshape.
kernel void enc_pack_convout(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                             constant int &H [[buffer(2)]], constant int &W [[buffer(3)]],
                             constant int &C [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
  const int f = int(gid.x), ch = int(gid.y);
  if (f >= W || ch >= C * H) return;
  const int c = ch / H, h = ch - c * H;
  out[f * (C * H) + ch] = x[(h * W + f) * C + c];
}

// ---------------------------------------------------------------------------
// Attention over one window: q,k,v [L, 1024] (16 heads x 64), out [L, 1024].
// One warp per query; grid (heads, ceil(L/8)), 256 threads.
// ---------------------------------------------------------------------------
kernel void enc_attn(device const float *q [[buffer(0)]],
                     device const float *k [[buffer(1)]],
                     device const float *v [[buffer(2)]],
                     device float *out [[buffer(3)]],
                     constant int &L [[buffer(4)]],
                     constant float &scale [[buffer(5)]],
                     constant int &D [[buffer(6)]],  // d_model: row stride of q/k/v/out
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint warp [[simdgroup_index_in_threadgroup]]) {
  threadgroup float probs[8][128];
  const int head = int(tg.x);
  const int i = int(tg.y) * 8 + int(warp);
  const int base = head * 64 + int(lane) * 2;
  const bool active = i < L;
  const float q0 = active ? q[(size_t)i * D + base] : 0.0f;
  const float q1 = active ? q[(size_t)i * D + base + 1] : 0.0f;

  // One lane per key j: the dot product must be summed over the 64 head dims by ONE lane
  // (a warp reduction would mix different j's). Lanes then hold 4 logits each.
  float s[4];
  float mx = -INFINITY;
  const device float *qp = q + (size_t)i * D + head * 64;
  for (int t = 0; t < 4; ++t) {
    const int j = int(lane) + t * 32;
    const bool ok = active && j < L;
    float acc = 0.0f;
    if (ok) {
      const device float *kp = k + (size_t)j * D + head * 64;
      for (int d = 0; d < 64; ++d) acc += qp[d] * kp[d];
    }
    s[t] = ok ? acc * scale : -INFINITY;
    mx = metal::max(mx, s[t]);
  }
  mx = simd_max(mx);
  float sum = 0.0f;
  for (int t = 0; t < 4; ++t) {
    const int j = int(lane) + t * 32;
    if (active && j < L) {
      s[t] = metal::precise::exp(s[t] - mx);
      sum += s[t];
    } else {
      s[t] = 0.0f;
    }
  }
  sum = simd_sum(sum);
  const float inv = 1.0f / sum;
  for (int t = 0; t < 4; ++t) {
    const int j = int(lane) + t * 32;
    if (j < 128) probs[warp][j] = (j < L) ? s[t] * inv : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (active) {
    float a0 = 0.0f, a1 = 0.0f;
    for (int j = 0; j < L; ++j) {
      const float p = probs[warp][j];
      a0 += p * v[(size_t)j * D + base];
      a1 += p * v[(size_t)j * D + base + 1];
    }
    out[(size_t)i * D + base] = a0;
    out[(size_t)i * D + base + 1] = a1;
  }
}

// conv_out row store: dst[dst_row + r, :] = src[r, :] + pos[r, :]   (width = d_model)
kernel void enc_convout_store(device const float *src [[buffer(0)]],
                              device const float *pos [[buffer(1)]],
                              device float *dst [[buffer(2)]],
                              constant int &rows [[buffer(3)]],
                              constant int &width [[buffer(4)]],
                              constant int &dst_row [[buffer(5)]],
                              uint2 gid [[thread_position_in_grid]]) {
  const int j = int(gid.x), r = int(gid.y);
  if (r >= rows || j >= width) return;
  dst[(size_t)(dst_row + r) * width + j] = src[(size_t)r * width + j] + pos[(size_t)r * width + j];
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// MLX steel GEMM numerics (f32 A x bf16 B -> f32), bits identical to MLX:
//   split-K iff ceil(M/16)*ceil(N/16) <= 2048 && K/16 >= 8 && K >= max(M,N)
//   (mlx matmul.cpp), parts = min(max(2, next_pow2((K/16)/(ceil(M/32)*ceil(N/32)))), 32),
//   partition z accumulates k in [z*psize, (z+1)*psize) ascending; the reduce sums the
//   partitions in order (out = ((0+C0)+C1+..)) and applies the addmm epilogue acc + bias.
// The tile shape does not change the bits (each output is one ascending-k fma chain), so
// this uses 64x64 tiles / 256 threads / 4x4 per thread instead of MLX's 32x32.
// parts == 1 writes acc+bias straight out (one dispatch, same value as 0+C0+bias).
// ---------------------------------------------------------------------------
constant int SK_BM = 64;
constant int SK_BN = 64;
constant int SK_BK = 32;

kernel void enc_gemm_splitk(device const float *A [[buffer(0)]],
                            device const bfloat *B [[buffer(1)]],
                            device float *C [[buffer(2)]],
                            device const float *bias [[buffer(3)]],
                            constant int &M [[buffer(4)]],
                            constant int &N [[buffer(5)]],
                            constant int &K [[buffer(6)]],
                            constant int &psize [[buffer(7)]],
                            constant int &fuse [[buffer(8)]],
                            uint3 tg [[threadgroup_position_in_grid]],
                            uint3 tid [[thread_position_in_threadgroup]]) {
  const int m0 = int(tg.y) * SK_BM;
  const int n0 = int(tg.x) * SK_BN;
  const int kz = int(tg.z);
  const int k0 = kz * psize;
  const int k1 = min(K, k0 + psize);
  threadgroup float As[SK_BK][SK_BM];
  threadgroup float Bs[SK_BK][SK_BN];
  const int i0 = int(tid.x) + 16 * int(tid.y);  // 0..255
  const int row = i0 >> 2;                      // 0..63
  const int kk = (i0 & 3) * 4;                  // 0,4,8,12
  const int gm = m0 + row;
  const int gn = n0 + row;
  const int tx4 = int(tid.x) * 4;
  const int ty4 = int(tid.y) * 4;
  float acc[4][4] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int kb = k0; kb < k1; kb += SK_BK) {
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int kx = kk + hf * 16;
      if (kb + kx + 4 <= K) {
        float4 v = float4(0.0f);
        if (gm < M) v = *(const device float4 *)(A + (size_t)gm * K + kb + kx);
        As[kx + 0][row] = v.x;
        As[kx + 1][row] = v.y;
        As[kx + 2][row] = v.z;
        As[kx + 3][row] = v.w;
        float4 vb = float4(0.0f);
        if (gn < N) {
          bfloat4 bv = *(const device bfloat4 *)(B + (size_t)gn * K + kb + kx);
          vb = float4(float(bv.x), float(bv.y), float(bv.z), float(bv.w));
        }
        Bs[kx + 0][row] = vb.x;
        Bs[kx + 1][row] = vb.y;
        Bs[kx + 2][row] = vb.z;
        Bs[kx + 3][row] = vb.w;
      } else {
        for (int t = 0; t < 4; ++t) {
          const int k = kb + kx + t;
          As[kx + t][row] = (gm < M && k < K) ? A[(size_t)gm * K + k] : 0.0f;
          Bs[kx + t][row] = (gn < N && k < K) ? float(B[(size_t)gn * K + k]) : 0.0f;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (int k = 0; k < SK_BK; ++k) {
      const float4 a = *(const threadgroup float4 *)&As[k][tx4];
      const float4 b = *(const threadgroup float4 *)&Bs[k][ty4];
      acc[0][0] += a.x * b.x; acc[0][1] += a.x * b.y; acc[0][2] += a.x * b.z; acc[0][3] += a.x * b.w;
      acc[1][0] += a.y * b.x; acc[1][1] += a.y * b.y; acc[1][2] += a.y * b.z; acc[1][3] += a.y * b.w;
      acc[2][0] += a.z * b.x; acc[2][1] += a.z * b.y; acc[2][2] += a.z * b.z; acc[2][3] += a.z * b.w;
      acc[3][0] += a.w * b.x; acc[3][1] += a.w * b.y; acc[3][2] += a.w * b.z; acc[3][3] += a.w * b.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const int mbase = m0 + tx4;
  const int nbase = n0 + ty4;
  if (fuse) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (mbase + i >= M) continue;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int n = nbase + j;
        if (n >= N) continue;
        C[(size_t)(mbase + i) * N + n] = acc[i][j] + bias[n];
      }
    }
    return;
  }
  const size_t zoff = (size_t)kz * M * N;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (mbase + i >= M) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int n = nbase + j;
      if (n >= N) continue;
      C[zoff + (size_t)(mbase + i) * N + n] = acc[i][j];
    }
  }
}

// gemm_splitk_accum: out = ((0 + C0) + C1 + ...) then the addmm epilogue acc + bias.
kernel void enc_gemm_reduce(device const float *Cs [[buffer(0)]],
                            device const float *bias [[buffer(1)]],
                            device float *C [[buffer(2)]],
                            constant int &M [[buffer(3)]],
                            constant int &N [[buffer(4)]],
                            constant int &parts [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
  const size_t idx = (size_t)gid;
  if (idx >= (size_t)M * N) return;
  float out = 0.0f;
  for (int i = 0; i < parts; ++i) out += Cs[(size_t)i * M * N + idx];
  C[idx] = out + bias[idx % N];
}

// ---------------------------------------------------------------------------
// Conv2d 3x3 / stride 2 / pad 1 via f32 simdgroup MMA.
//
// Same math and the same accumulation order as enc_conv (k = tap*C + c ascending, one
// fp32 accumulator per output, bias added at the end), so the result is bit-identical;
// Apple's simdgroup_float8x8 fp32 MMA is a sequential 8-deep fp32 FMA chain
// (test_encoder --gemmcmp: bit_mismatch 0 vs the fma-tile GEMM).
//
// The A tile is gathered per tap: for one tap, k in [tap*C, (tap+1)*C) reads the
// contiguous C-float input row X[ih][iw][:], so a thread loads 4 consecutive k with one
// float4 and the (ih, iw) -> offset mapping is computed once per tap into rowoff[].
// ---------------------------------------------------------------------------
struct ConvSgParams {
  int H, W, C;  // input spatial size and channels
  int Ho, Wo;   // output spatial size
  int N;        // output channels
  int xs;       // input frame stride
};

template <int BM, int BN, int BK, int WM, int WN>
METAL_FUNC void conv_sg_body(device const float *X,
                             device const bfloat *Wt,
                             device float *Y,
                             device const float *bias,
                             constant ConvSgParams &p,
                             threadgroup float *smem,
                             threadgroup int *rowoff,
                             uint3 tg,
                             uint3 tid,
                             uint warp) {
  constexpr int TM = BM / (8 * WM);  // 8x8 fragments per simdgroup along M
  constexpr int TN = BN / (8 * WN);
  constexpr int NTHR = WM * WN * 32;
  constexpr int PAD = 4;  // floats of row padding (16 B), as MLX steel
  constexpr int AS = BK + PAD, BS = BN + PAD;
  threadgroup float *As = smem;
  threadgroup float *Bs = smem + BM * AS;
  threadgroup float *Ct = smem;  // overlays As+Bs; safe after the last k barrier
  const int m0 = int(tg.y) * BM;
  const int n0 = int(tg.x) * BN;
  const int M = p.Ho * p.Wo;
  const int K = 9 * p.C;
  const int t = int(tid.x);

  simdgroup_float8x8 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  for (int tap = 0; tap < 9; ++tap) {
    const int kh = tap / 3, kw = tap - kh * 3;
    for (int r = t; r < BM; r += NTHR) {
      const int m = m0 + r;
      int off = -1;
      if (m < M) {
        const int oh = m / p.Wo, ow = m - oh * p.Wo;
        const int ih = oh * 2 - 1 + kh, iw = ow * 2 - 1 + kw;
        if (ih >= 0 && ih < p.H && iw >= 0 && iw < p.W) off = (ih * p.xs + iw) * p.C;
      }
      rowoff[r] = off;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c0 = 0; c0 < p.C; c0 += BK) {
      const int kb = tap * p.C + c0;
      // A tile: BM*BK elements, 4 consecutive k per thread (one input row segment).
      for (int e = t * 4; e < BM * BK; e += NTHR * 4) {
        const int r = e / BK, kx = e - r * BK;
        const int off = rowoff[r];
        float4 v = float4(0.0f);
        if (off >= 0) v = *(const device float4 *)(X + off + c0 + kx);
        As[r * AS + kx + 0] = v.x;
        As[r * AS + kx + 1] = v.y;
        As[r * AS + kx + 2] = v.z;
        As[r * AS + kx + 3] = v.w;
      }
      // B tile: BK*BN elements, 4 consecutive k per output channel.
      for (int e = t * 4; e < BK * BN; e += NTHR * 4) {
        const int n = e / BK, kx = e - n * BK;
        const int gn = n0 + n;
        float4 v = float4(0.0f);
        if (gn < p.N) {
          bfloat4 bv = *(const device bfloat4 *)(Wt + (size_t)gn * K + kb + kx);
          v = float4(float(bv.x), float(bv.y), float(bv.z), float(bv.w));
        }
        Bs[(kx + 0) * BS + n] = v.x;
        Bs[(kx + 1) * BS + n] = v.y;
        Bs[(kx + 2) * BS + n] = v.z;
        Bs[(kx + 3) * BS + n] = v.w;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const int wm = (warp / WN) * (BM / WM);
      const int wn = (warp % WN) * (BN / WN);
#pragma unroll
      for (int k8 = 0; k8 < BK; k8 += 8) {
        simdgroup_float8x8 af[TM], bf[TN];
#pragma unroll
        for (int i = 0; i < TM; ++i)
          simdgroup_load(af[i], As + (wm + i * 8) * AS + k8, AS);
#pragma unroll
        for (int j = 0; j < TN; ++j)
          simdgroup_load(bf[j], Bs + k8 * BS + wn + j * 8, BS);
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j)
            simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  constexpr int CS = BN + PAD;
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      simdgroup_store(acc[i][j],
                      Ct + ((warp / WN) * (BM / WM) + i * 8) * CS + (warp % WN) * (BN / WN) + j * 8,
                      CS);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int e = t; e < BM * BN; e += NTHR) {
    const int r = e / BN, c = e - r * BN;
    const int m = m0 + r, n = n0 + c;
    if (m >= M || n >= p.N) continue;
    Y[(size_t)m * p.N + n] = Ct[r * CS + c] + bias[n];
  }
}

#define CONV_SG_KERNEL(NAME, BM, BN, BK, WM, WN)                                          \
  kernel void NAME(device const float *X [[buffer(0)]],                                   \
                   device const bfloat *Wt [[buffer(1)]],                                 \
                   device float *Y [[buffer(2)]],                                         \
                   device const float *bias [[buffer(3)]],                                \
                   constant ConvSgParams &p [[buffer(4)]],                                \
                   uint3 tg [[threadgroup_position_in_grid]],                             \
                   uint3 tid [[thread_position_in_threadgroup]],                          \
                   uint warp [[simdgroup_index_in_threadgroup]]) {                         \
    constexpr int SMEM = (BM * (BK + 4) + BK * (BN + 4) > BM * (BN + 4))                  \
                             ? BM * (BK + 4) + BK * (BN + 4)                              \
                             : BM * (BN + 4);                                             \
    threadgroup float smem[SMEM];                                                         \
    threadgroup int rowoff[BM];                                                           \
    conv_sg_body<BM, BN, BK, WM, WN>(X, Wt, Y, bias, p, smem, rowoff, tg, tid, warp);     \
  }

CONV_SG_KERNEL(enc_conv_sg1, 32, 32, 32, 2, 2)
CONV_SG_KERNEL(enc_conv_sg2, 32, 64, 32, 2, 2)
CONV_SG_KERNEL(enc_conv_sg3, 64, 64, 32, 2, 2)
CONV_SG_KERNEL(enc_conv_sg4, 32, 64, 64, 2, 2)
CONV_SG_KERNEL(enc_conv_sg5, 64, 32, 32, 2, 2)
CONV_SG_KERNEL(enc_conv_sg6, 32, 128, 32, 2, 4)
CONV_SG_KERNEL(enc_conv_sg7, 16, 64, 32, 2, 2)
CONV_SG_KERNEL(enc_conv_sg8, 64, 64, 32, 2, 4)
