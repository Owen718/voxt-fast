#include "decoder.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace dec {

namespace {

// Quantization bits of the loaded checkpoint (6: Qwen3-ASR 1.7B, 4: 0.6B), set by the Decoder
// constructor. Kernel names below are written for 6 bits; qk() swaps in the loaded width.
int g_bits = 6;
const char *qk(const char *name6) {
  if (g_bits == 6) return name6;
  static thread_local std::string ring[4];
  static thread_local int next = 0;
  std::string &s = ring[next++ & 3];
  s = name6;
  for (const char *pat : {"_b_6", "_b6"}) {
    const size_t at = s.find(pat);
    if (at != std::string::npos) {
      const size_t len = std::strlen(pat);
      s.replace(at + len - 1, 1, std::to_string(g_bits));
      break;
    }
  }
  return s.c_str();
}

// qmv_fast dispatch exactly as MLX quantized.cpp qmv() for M rows, one batch:
// grid (M, ceil(N/8), 1), group (32, 2, 1).
void dispatch_qmv_fast(gpu::Cmd &c, const gpu::Tensor &x, const gpu::Tensor &w,
                       const gpu::Tensor &s, const gpu::Tensor &b, gpu::Tensor &y,
                       int M, int N, int K) {
  c.use(qk("affine_qmv_fast_bfloat16_t_gs_64_b_6_batch_0"));
  c.set(w, 0);
  c.set(s, 1);
  c.set(b, 2);
  c.set(x, 3);
  c.set(y, 4);
  c.set_value(K, 5);
  c.set_value(N, 6);
  c.dispatch_groups(MTLSizeMake(M, (N + 7) / 8, 1), MTLSizeMake(32, 2, 1));
}

// MLX normalization.cpp: threadgroup = 32 * ceil(ceil(axis/4)/32) threads.
void dispatch_rms(gpu::Cmd &c, const gpu::Tensor &x, const gpu::Tensor &w, gpu::Tensor &out,
                  int rows, int axis, float eps) {
  int tg = 32 * (((axis + 3) / 4 + 31) / 32);
  uint32_t axis_u = (uint32_t)axis, wstride_u = 1;
  c.use("dec_rms_bf16");
  c.set(x, 0);
  c.set(w, 1);
  c.set(out, 2);
  c.set_value(eps, 3);
  c.set_value(axis_u, 4);
  c.set_value(wstride_u, 5);
  c.dispatch_threads(MTLSizeMake((size_t)rows * tg, 1, 1), MTLSizeMake(tg, 1, 1));
}


// sdpa_vector is templated on function constants (no mask, no causal, no sinks);
// build the specialised pipeline once.
id<MTLComputePipelineState> sdpa_pipeline() {
  static id<MTLComputePipelineState> p = nil;
  if (!p) {
    MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
    bool f = false;
    int blocks = 0;
    for (int i = 20; i <= 25; ++i) [cv setConstantValue:&f type:MTLDataTypeBool atIndex:i];
    [cv setConstantValue:&blocks type:MTLDataTypeInt atIndex:26];
    p = gpu::Context::get().pipeline("dec_sdpa_vector_bf16_128_128", "nomask_nc", cv);
  }
  return p;
}

// Same kernel with do_causal = true (MLX uses the vector kernel for qL <= 8).
id<MTLComputePipelineState> sdpa_pipeline_causal() {
  static id<MTLComputePipelineState> p = nil;
  if (!p) {
    MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
    bool f = false, t = true;
    for (int i = 20; i <= 25; ++i) [cv setConstantValue:&f type:MTLDataTypeBool atIndex:i];
    [cv setConstantValue:&t type:MTLDataTypeBool atIndex:22];  // do_causal
    int blocks = 0;
    [cv setConstantValue:&blocks type:MTLDataTypeInt atIndex:26];
    p = gpu::Context::get().pipeline("dec_sdpa_vector_bf16_128_128", "nomask_c", cv);
  }
  return p;
}

void dispatch_rope(gpu::Cmd &c, const gpu::Tensor &in, gpu::Tensor &out, int n_heads,
                   int head_dim, int offset, float theta) {
  int half = head_dim / 2;
  int64_t stride = head_dim;
  float scale = 1.0f;
  float base = std::log2(theta);
  c.use("dec_rope_single_bf16");
  c.set(in, 0);
  c.set(out, 1);
  c.set_value(offset, 2);
  c.set_value(scale, 3);
  c.set_value(stride, 4);
  c.set_bytes(&base, sizeof(base), 10);
  c.dispatch_threads(MTLSizeMake(half, n_heads, 1), MTLSizeMake(half, n_heads, 1));
}


// Fused qmv_fast variants (kernels/dec_fuse.metal): identical per-row arithmetic.
// shape 0 = 2 simdgroups x 4 rows/tg (MLX shape), 1 = 4 x 2 (more warps in flight).
struct QmvShape { const char *suffix; int nsg, rps; };
const QmvShape kQmvShapes[] = {{"", 2, 4}, {"_s42", 4, 2}, {"_s81", 8, 1}, {"_s82", 8, 2}, {"_s44", 4, 4}};
constexpr int kQmvShapeCount = 5;
const char *qmv_kern(const char *base, int shape) {
  static thread_local std::string k;
  k = base;
  const char *sfx = kQmvShapes[shape].suffix;
  if (sfx[0]) k.replace(k.find("_bf16"), 5, std::string(sfx) + "_bf16");
  return k.c_str();
}
int qmv_rows(int shape) { return kQmvShapes[shape].nsg * kQmvShapes[shape].rps; }
MTLSize qmv_tg(int shape) { return MTLSizeMake(32, kQmvShapes[shape].nsg, 1); }

void dispatch_qmv3(gpu::Cmd &c, const gpu::Tensor &x, const gpu::Tensor &w0,
                   const gpu::Tensor &s0, const gpu::Tensor &b0, const gpu::Tensor &w1,
                   const gpu::Tensor &s1, const gpu::Tensor &b1, const gpu::Tensor &w2,
                   const gpu::Tensor &s2, const gpu::Tensor &b2, gpu::Tensor &y, int M, int n0,
                   int n1, int n2, int K, int shape = 0) {
  c.use(qmv_kern(qk("dec_qmv3_bf16_gs64_b6"), shape));
  c.set(w0, 0); c.set(s0, 1); c.set(b0, 2);
  c.set(w1, 3); c.set(s1, 4); c.set(b1, 5);
  c.set(w2, 6); c.set(s2, 7); c.set(b2, 8);
  c.set(x, 9); c.set(y, 10);
  c.set_value(K, 11); c.set_value(n0 + n1 + n2, 12); c.set_value(n0, 13); c.set_value(n1, 14);
  c.dispatch_groups(MTLSizeMake(M, ((n0 + n1 + n2) + qmv_rows(shape) - 1) / qmv_rows(shape), 1), qmv_tg(shape));
}

void dispatch_qmv2(gpu::Cmd &c, const gpu::Tensor &x, const gpu::Tensor &w0,
                   const gpu::Tensor &s0, const gpu::Tensor &b0, const gpu::Tensor &w1,
                   const gpu::Tensor &s1, const gpu::Tensor &b1, gpu::Tensor &y, int M, int n0,
                   int n1, int K, int shape = 0) {
  c.use(qmv_kern(qk("dec_qmv2_bf16_gs64_b6"), shape));
  c.set(w0, 0); c.set(s0, 1); c.set(b0, 2);
  c.set(w1, 3); c.set(s1, 4); c.set(b1, 5);
  c.set(x, 6); c.set(y, 7);
  c.set_value(K, 8); c.set_value(n0 + n1, 9); c.set_value(n0, 10);
  c.dispatch_groups(MTLSizeMake(M, ((n0 + n1) + qmv_rows(shape) - 1) / qmv_rows(shape), 1), qmv_tg(shape));
}

// qmv_fast + residual add epilogue.
void dispatch_qmv_add(gpu::Cmd &c, const gpu::Tensor &x, const gpu::Tensor &w,
                      const gpu::Tensor &s, const gpu::Tensor &b, const gpu::Tensor &res,
                      gpu::Tensor &y, int M, int N, int K, int shape = 0) {
  c.use(qmv_kern(qk("dec_qmv_add_bf16_gs64_b6"), shape));
  c.set(w, 0); c.set(s, 1); c.set(b, 2); c.set(x, 3); c.set(res, 4); c.set(y, 5);
  c.set_value(K, 6); c.set_value(N, 7);
  c.dispatch_groups(MTLSizeMake(M, ((N) + qmv_rows(shape) - 1) / qmv_rows(shape), 1), qmv_tg(shape));
}

void dispatch_silu_mul(gpu::Cmd &c, const gpu::Tensor &g, const gpu::Tensor &u, gpu::Tensor &out,
                       int64_t n) {
  c.use("dec_silu_mul_bf16");
  c.set(g, 0); c.set(u, 1); c.set(out, 2);
  int64_t tg = n < 256 ? n : 256;
  c.dispatch_threads(MTLSizeMake((size_t)n, 1, 1), MTLSizeMake((size_t)tg, 1, 1));
}

// q rows then k rows (one threadgroup per row, MLX rms_single_row math).
void dispatch_rms_qk(gpu::Cmd &c, const gpu::Tensor &xq, const gpu::Tensor &wq,
                     gpu::Tensor &outq, const gpu::Tensor &xk, const gpu::Tensor &wk,
                     gpu::Tensor &outk, int nq, int nk, int axis, float eps) {
  int tg = 32 * (((axis + 3) / 4 + 31) / 32);
  uint32_t axis_u = (uint32_t)axis, nq_u = (uint32_t)nq;
  c.use("dec_rms_qk_bf16");
  c.set(xq, 0); c.set(wq, 1); c.set(outq, 2);
  c.set(xk, 3); c.set(wk, 4); c.set(outk, 5);
  c.set_value(eps, 6); c.set_value(axis_u, 7); c.set_value(nq_u, 8);
  c.dispatch_threads(MTLSizeMake((size_t)(nq + nk) * tg, 1, 1), MTLSizeMake(tg, 1, 1));
}

// q/k rope in one dispatch; the k result is written into the KV cache row `pos`.
void dispatch_rope_qk_store(gpu::Cmd &c, const gpu::Tensor &inq, gpu::Tensor &outq,
                            const gpu::Tensor &ink, gpu::Tensor &kcache, int nq, int nkv,
                            int hd, int cap, int pos, float theta) {
  int half = hd / 2;
  int64_t stride = hd;
  float scale = 1.0f;
  float base = std::log2(theta);
  c.use("dec_rope_qk_store_bf16");
  c.set(inq, 0); c.set(outq, 1); c.set(ink, 2); c.set(kcache, 3);
  c.set_value(pos, 4); c.set_value(scale, 5); c.set_value(stride, 6);
  c.set_value(nq, 7); c.set_value(cap, 8); c.set_value(hd, 9);
  c.set_bytes(&base, sizeof(base), 10);
  // threads_per_threadgroup must stay <= 1024: (half, 8) = 512 for head_dim 128
  c.dispatch_threads(MTLSizeMake(half, nq + nkv, 1), MTLSizeMake(half, 8, 1));
}

void dispatch_argmax(gpu::Cmd &c, const gpu::Tensor &x, gpu::Tensor &out, int64_t n) {
  uint32_t n_u = (uint32_t)n;
  c.use("dec_argmax_bf16");
  c.set(x, 0); c.set(out, 1); c.set_value(n_u, 2);
  c.dispatch_threads(MTLSizeMake(1024, 1, 1), MTLSizeMake(1024, 1, 1));
}

// Per-row top1/top2 over bf16 logits [rows, n]: one threadgroup of 1024 per row.
void dispatch_top2(gpu::Cmd &c, const gpu::Tensor &logits, gpu::Tensor &idx,
                   gpu::Tensor &margin, int rows, int64_t n) {
  uint32_t n_u = (uint32_t)n;
  c.use("dec_top2_bf16");
  c.set(logits, 0); c.set(idx, 1); c.set(margin, 2); c.set_value(n_u, 3);
  c.dispatch_groups(MTLSizeMake(rows, 1, 1), MTLSizeMake(1024, 1, 1));
}

// MLX quantized.cpp qmv_wide (M in [2, get_qmv_batch_limit)): vecs_per_tg
// re-reads the weights per input-vector tile; k_lanes = 8 for affine.
void dispatch_qmv_wide(gpu::Cmd &c, const gpu::Tensor &x, const gpu::Tensor &w,
                       const gpu::Tensor &s, const gpu::Tensor &b, gpu::Tensor &y, int M, int N,
                       int K) {
  int n_tiles = (M + 4) / 5;
  int vecs = (M + n_tiles - 1) / n_tiles;
  int rows_per_tg = (32 / 8) * 2;
  char name[64];
  if (g_bits == 6) std::snprintf(name, sizeof(name), "dec_qmv_wide_nv%d", vecs);
  else std::snprintf(name, sizeof(name), "dec_qmv_wide_b%d_nv%d", g_bits, vecs);
  c.use(name);
  c.set(w, 0);
  c.set(s, 1);
  c.set(b, 2);
  c.set(x, 3);
  c.set(y, 4);
  c.set_value(K, 5);
  c.set_value(N, 6);
  c.set_value(M, 7);
  c.dispatch_groups(MTLSizeMake((M + vecs - 1) / vecs, (N + rows_per_tg - 1) / rows_per_tg, 1),
                    MTLSizeMake(32, 2, 1));
}

// MLX get_qmv_batch_limit for this device (applegpu_g16s: gen>=15, non-'d') is 13
// for every shape in this model (D<=2048&&O<=2048, else 13).
constexpr int kQmvBatchLimit = 13;

struct QmmPlan {
  int split_k;
  int k_partition;
};

// MLX quantized.cpp qmm_splitk: choose split_k to target ~512 threadgroups.
QmmPlan qmm_plan(int M, int N, int K, int group_size) {
  int bm = 32, bn = 32;
  int n_tiles = (N + bn - 1) / bn, m_tiles = (M + bm - 1) / bm;
  int split_k = std::max(1, 512 / (n_tiles * m_tiles));
  int k_align = group_size > 32 ? group_size : 32;
  split_k = std::min(split_k, K / k_align);
  while (split_k > 1 && (K % (split_k * k_align) != 0)) split_k--;
  return {split_k, K / std::max(split_k, 1)};
}

void dispatch_qmm_t(gpu::Cmd &c, const gpu::Tensor &x, const gpu::Tensor &w, const gpu::Tensor &s,
                    const gpu::Tensor &b, gpu::Tensor &y, int M, int N, int K) {
  c.use(qk("affine_qmm_t_bfloat16_t_gs_64_b_6_alN_true_batch_0"));
  c.set(w, 0);
  c.set(s, 1);
  c.set(b, 2);
  c.set(x, 3);
  c.set(y, 4);
  c.set_value(K, 5);
  c.set_value(N, 6);
  c.set_value(M, 7);
  c.dispatch_groups(MTLSizeMake((N + 31) / 32, (M + 31) / 32, 1), MTLSizeMake(32, 2, 2));
}

void dispatch_qmm_splitk(gpu::Cmd &c, const gpu::Tensor &x, const gpu::Tensor &w,
                         const gpu::Tensor &s, const gpu::Tensor &b, gpu::Tensor &temp,
                         int M, int N, int K, const QmmPlan &plan) {
  int stride = M * N;
  c.use(qk("affine_qmm_t_splitk_bfloat16_t_gs_64_b_6_alN_true"));
  c.set(w, 0);
  c.set(s, 1);
  c.set(b, 2);
  c.set(x, 3);
  c.set(temp, 4);
  c.set_value(K, 5);
  c.set_value(N, 6);
  c.set_value(M, 7);
  c.set_value(plan.k_partition, 8);
  c.set_value(stride, 9);
  c.dispatch_groups(MTLSizeMake((N + 31) / 32, (M + 31) / 32, plan.split_k), MTLSizeMake(32, 2, 2));
}

// MLX get_block_dims_common(dim0, dim1, dim2, pow2 = 10)
void block_dims(int d0, int d1, int d2, int out[3]) {
  int pows[3] = {0, 0, 0}, sum = 0;
  while (true) {
    int presum = sum;
    if (d0 >= (1 << (pows[0] + 1))) { pows[0]++; sum++; }
    if (sum == 10) break;
    if (d1 >= (1 << (pows[1] + 1))) { pows[1]++; sum++; }
    if (sum == 10) break;
    if (d2 >= (1 << (pows[2] + 1))) { pows[2]++; sum++; }
    if (sum == presum || sum == 10) break;
  }
  for (int i = 0; i < 3; ++i) out[i] = 1 << pows[i];
}

// steel attention, no mask, causal; function constants align_Q / align_K / causal.
id<MTLComputePipelineState> steel_attention_pipeline(bool align_q, bool align_k) {
  static id<MTLComputePipelineState> cache[4] = {nil, nil, nil, nil};
  int idx = (align_q ? 1 : 0) + (align_k ? 2 : 0);
  if (!cache[idx]) {
    MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
    bool t = true, f = false;
    [cv setConstantValue:(align_q ? &t : &f) type:MTLDataTypeBool atIndex:200];
    [cv setConstantValue:(align_k ? &t : &f) type:MTLDataTypeBool atIndex:201];
    [cv setConstantValue:&f type:MTLDataTypeBool atIndex:300];
    [cv setConstantValue:&t type:MTLDataTypeBool atIndex:301];
    [cv setConstantValue:&f type:MTLDataTypeBool atIndex:302];
    std::string key = std::string("q") + (align_q ? '1' : '0') + "k" + (align_k ? '1' : '0');
    cache[idx] = gpu::Context::get().pipeline(
        "dec_steel_attention_bf16_bq32_bk16_bd128_bv128_wm4_wn1", key, cv);
  }
  return cache[idx];
}

}  // namespace

void KVCache::reserve(int new_cap, int n_kv, int head_dim) {
  if (new_cap <= cap) return;
  auto &ctx = gpu::Context::get();
  gpu::Tensor nk = ctx.empty({n_kv, new_cap, head_dim}, gpu::DType::BF16);
  gpu::Tensor nv = ctx.empty({n_kv, new_cap, head_dim}, gpu::DType::BF16);
  if (cap > 0 && len > 0) {  // growth keeps the first len rows of every kv head
    const size_t row_bytes = (size_t)len * head_dim * 2;
    for (int h = 0; h < n_kv; ++h) {
      memcpy((char *)nk.host() + (size_t)h * new_cap * head_dim * 2,
             (const char *)keys.host() + (size_t)h * cap * head_dim * 2, row_bytes);
      memcpy((char *)nv.host() + (size_t)h * new_cap * head_dim * 2,
             (const char *)values.host() + (size_t)h * cap * head_dim * 2, row_bytes);
    }
  }
  keys = nk;
  values = nv;
  cap = new_cap;
}

void Decoder::Trace::add(const std::string &name, const gpu::Tensor &t) {
  if (!enabled) return;
  names.push_back(name);
  tensors.push_back(t);
}

Decoder::Decoder(const Weights &w) : w_(w) {
  const json::Value &tc = w.config["thinker_config"]["text_config"];
  cfg_.n_layers = (int)tc["num_hidden_layers"].as_int();
  cfg_.n_heads = (int)tc["num_attention_heads"].as_int();
  cfg_.n_kv = (int)tc["num_key_value_heads"].as_int();
  cfg_.head_dim = (int)tc["head_dim"].as_int();
  cfg_.hidden = (int)tc["hidden_size"].as_int();
  cfg_.inter = (int)tc["intermediate_size"].as_int();
  cfg_.vocab = (int)tc["vocab_size"].as_int();
  cfg_.eps = (float)tc["rms_norm_eps"].as_num();
  cfg_.rope_theta = (float)tc["rope_theta"].as_num();
  const json::Value &q = w.config["quantization"];
  const int bits = (int)q["bits"].as_int(), group = (int)q["group_size"].as_int();
  if ((bits != 6 && bits != 4) || group != 64)
    throw std::runtime_error("unsupported quantization: bits " + std::to_string(bits) +
                             ", group size " + std::to_string(group) +
                             " (supported: 4 or 6 bits, group size 64)");
  g_bits = bits;
}

gpu::Tensor Decoder::rms_norm(const gpu::Tensor &x, const std::string &wname, int rows, int axis) {
  gpu::Tensor out = gpu::Context::get().empty({(int64_t)rows, (int64_t)axis}, gpu::DType::BF16);
  gpu::Cmd c;
  dispatch_rms(c, x, w_.get(wname), out, rows, axis, cfg_.eps);
  c.commit_and_wait();
  return out;
}

gpu::Tensor Decoder::qmv(const gpu::Tensor &x, const std::string &prefix, int n, int k) {
  const gpu::Tensor &w = w_.get(prefix + ".weight");
  int N = (int)w.shape[0];
  gpu::Tensor y = gpu::Context::get().empty({(int64_t)n, (int64_t)N}, gpu::DType::BF16);
  gpu::Cmd c;
  dispatch_qmv_fast(c, x, w, w_.get(prefix + ".scales"), w_.get(prefix + ".biases"), y, n, N, k);
  c.commit_and_wait();
  return y;
}

gpu::Tensor Decoder::embed(const std::vector<int> &ids) {
  auto &ctx = gpu::Context::get();
  gpu::Tensor out = ctx.empty({(int64_t)ids.size(), (int64_t)cfg_.hidden}, gpu::DType::BF16);
  gpu::Cmd c;
  embed_into(c, ids, out);
  c.commit_and_wait();
  return out;
}

void Decoder::embed_into(gpu::Cmd &c, const std::vector<int> &ids, gpu::Tensor &out) {
  int n = (int)ids.size();
  int K = cfg_.hidden, packed = K * g_bits / 32, groups = K / 64;
  auto &ctx = gpu::Context::get();
  if (n > gather_cap_) {
    gather_cap_ = n;
    gather_w_ = ctx.empty({n, packed}, gpu::DType::U32);
    gather_s_ = ctx.empty({n, groups}, gpu::DType::BF16);
    gather_b_ = ctx.empty({n, groups}, gpu::DType::BF16);
  }
  // ids: <= 4 KB go inline via setBytes (constant address space), so several embed
  // calls can be recorded into one command buffer; larger lists get their own buffer.
  const bool ids_inline = (size_t)n * sizeof(int) <= 4096;
  gpu::Tensor ids_alloc;
  if (!ids_inline) {
    ids_alloc = ctx.empty({n}, gpu::DType::I32);
    memcpy(ids_alloc.host(), ids.data(), sizeof(int) * n);
  }
  const gpu::Tensor &w = w_.get("model.embed_tokens.weight");
  const gpu::Tensor &s = w_.get("model.embed_tokens.scales");
  const gpu::Tensor &b = w_.get("model.embed_tokens.biases");
  c.use(ids_inline ? "dec_gather_rows_c_bf16" : "dec_gather_rows");
  c.set(w, 0);
  c.set(s, 1);
  c.set(b, 2);
  if (ids_inline) c.set_bytes(ids.data(), sizeof(int) * n, 3);
  else c.set(ids_alloc, 3);
  c.set(gather_w_, 4);
  c.set(gather_s_, 5);
  c.set(gather_b_, 6);
  c.set_value(n, 7);
  c.set_value(packed, 8);
  c.set_value(groups, 9);
  c.dispatch_threads(MTLSizeMake(packed + groups, n, 1), MTLSizeMake(32, 1, 1));
  // dequantize: one thread per pack (4 values at 6 bits, 2 at 4 bits; MLX get_pack_factor),
  // group = min(1024, nthreads) (MLX get_quantize_kernel_dims)
  size_t nthreads = (size_t)n * K / (g_bits == 6 ? 4 : 8 / g_bits);
  c.use(qk("affine_dequantize_bfloat16_t_gs_64_b_6"));
  c.set(gather_w_, 0);
  c.set(gather_s_, 1);
  c.set(gather_b_, 2);
  c.set(out, 3);
  size_t tg = nthreads < 1024 ? nthreads : 1024;
  c.dispatch_threads(MTLSizeMake(nthreads, 1, 1), MTLSizeMake(tg, 1, 1));
}

int Decoder::argmax(const gpu::Tensor &logits) {
  const uint16_t *p = logits.ptr<uint16_t>();
  int64_t n = logits.numel();
  float best = -INFINITY;
  int best_i = 0;
  for (int64_t i = 0; i < n; ++i) {
    float v = gpu::bf16_to_f32(p[i]);
    if (v > best) { best = v; best_i = (int)i; }
  }
  return best_i;
}


namespace {

void dispatch_ew(gpu::Cmd &c, const char *kernel, const gpu::Tensor &a, const gpu::Tensor *b,
                 gpu::Tensor &out, int64_t n) {
  c.use(kernel);
  c.set(a, 0);
  if (b) c.set(*b, 1);
  c.set(out, b ? 2 : 1);
  int64_t tg = n < 256 ? n : 256;
  c.dispatch_threads(MTLSizeMake((size_t)n, 1, 1), MTLSizeMake((size_t)tg, 1, 1));
}

}  // namespace

namespace {
constexpr int64_t kArenaBytes = 48 << 20;  // decode + batched prefill (n up to ~400)
}

gpu::Tensor Decoder::buf(std::vector<int64_t> shape) {
  if (trace.enabled) return gpu::Context::get().empty(shape, gpu::DType::BF16);
  if (!arena_.buf) arena_ = gpu::Context::get().empty({kArenaBytes / 2}, gpu::DType::BF16);
  int64_t numel = 1;
  for (auto d : shape) numel *= d;
  int64_t bytes = (numel * 2 + 255) / 256 * 256;
  if (arena_off_ + bytes > (int64_t)arena_.nbytes()) {
    // Grow: tensors already handed out keep the old buffer alive (Tensor retains it), so a fresh,
    // larger arena is safe mid-pass. Offsets below arena_off_ in the new buffer stay unused.
    int64_t need = std::max<int64_t>((int64_t)arena_.nbytes() * 2, (arena_off_ + bytes) * 2);
    arena_ = gpu::Context::get().empty({need / 2}, gpu::DType::BF16);
  }
  gpu::Tensor t = arena_;
  t.offset = (size_t)arena_off_;
  t.shape = std::move(shape);
  t.dtype = gpu::DType::BF16;
  arena_off_ += bytes;
  return t;
}

void Decoder::arena_reset() { arena_off_ = arena_keep_; }

// Fused decode: one dispatch for q/k/v, one for q/k rmsnorm, one for q/k rope with
// the k result stored straight into the cache, residual adds in the o/down
// epilogues, silu*up in one pass. Per-row arithmetic is identical to MLX qmv_fast.
gpu::Tensor Decoder::decode_layers_fused(gpu::Cmd &c, const gpu::Tensor &embeddings,
                                         std::vector<KVCache> &caches) {
  const int H = cfg_.hidden, nh = cfg_.n_heads, nkv = cfg_.n_kv, hd = cfg_.head_dim;
  const int inter = cfg_.inter;
  if ((int)caches.size() != cfg_.n_layers) caches.assign(cfg_.n_layers, KVCache{});
  auto view = [](const gpu::Tensor &t, int64_t off_elems, std::vector<int64_t> shape) {
    gpu::Tensor v = t;
    v.offset = t.offset + (size_t)off_elems * 2;
    v.shape = std::move(shape);
    return v;
  };
  arena_off_ = 0;
  arena_keep_ = 0;
  gpu::Tensor xa = buf({1, H}), xb = buf({1, H});
  arena_keep_ = arena_off_;
  gpu::Tensor x = embeddings;
  for (int l = 0; l < cfg_.n_layers; ++l) {
    const std::string p = "model.layers." + std::to_string(l);
    KVCache &cache = caches[l];
    const int pos = cache.len;
    if (pos + 1 > cache.cap) cache.reserve(((pos + 1) + 255) / 256 * 256, nkv, hd);
    const std::string tag = "L" + std::to_string(l) + "_";
    const std::string sa = p + ".self_attn.", sm = p + ".mlp.";
    arena_reset();

    gpu::Tensor normed = buf({1, H});
    dispatch_rms(c, x, w_.get(p + ".input_layernorm.weight"), normed, 1, H, cfg_.eps);
    trace.add(tag + "input_layernorm", normed);

    gpu::Tensor qkv = buf({1, nh * hd + 2 * nkv * hd});
    dispatch_qmv3(c, normed,
                  w_.get(sa + "q_proj.weight"), w_.get(sa + "q_proj.scales"), w_.get(sa + "q_proj.biases"),
                  w_.get(sa + "k_proj.weight"), w_.get(sa + "k_proj.scales"), w_.get(sa + "k_proj.biases"),
                  w_.get(sa + "v_proj.weight"), w_.get(sa + "v_proj.scales"), w_.get(sa + "v_proj.biases"),
                  qkv, 1, nh * hd, nkv * hd, nkv * hd, H, qmv_shape);
    gpu::Tensor q = view(qkv, 0, {1, nh * hd});
    gpu::Tensor k = view(qkv, nh * hd, {1, nkv * hd});
    gpu::Tensor v = view(qkv, nh * hd + nkv * hd, {1, nkv * hd});
    trace.add(tag + "q_proj", q);
    trace.add(tag + "k_proj", k);
    trace.add(tag + "v_proj", v);

    gpu::Tensor qn = buf({nh, hd}), kn = buf({nkv, hd});
    dispatch_rms_qk(c, q, w_.get(sa + "q_norm.weight"), qn, k, w_.get(sa + "k_norm.weight"), kn,
                    nh, nkv, hd, cfg_.eps);
    trace.add(tag + "q_norm", qn);
    trace.add(tag + "k_norm", kn);

    gpu::Tensor qr = buf({nh, hd});
    dispatch_rope_qk_store(c, qn, qr, kn, cache.keys, nh, nkv, hd, cache.cap, pos, cfg_.rope_theta);
    trace.add(tag + "q_rope", qr);

    c.use("dec_kv_store_bf16");
    c.set(v, 0);
    c.set(cache.values, 1);
    c.set_value(cache.cap, 2);
    c.set_value(pos, 3);
    c.set_value(hd, 4);
    c.dispatch_threads(MTLSizeMake(nkv * hd, 1, 1), MTLSizeMake(128, 1, 1));
    cache.len = pos + 1;

    gpu::Tensor attn = buf({1, nh * hd});
    {
      int gqa = nh / nkv, N = cache.len;
      int64_t k_head_stride = (int64_t)cache.cap * hd, k_seq_stride = hd;
      float scale = 1.0f / std::sqrt((float)hd);
      c.use(sdpa_pipeline());
      c.set(qr, 0);
      c.set(cache.keys, 1);
      c.set(cache.values, 2);
      c.set(attn, 3);
      c.set_value(gqa, 4);
      c.set_value(N, 5);
      c.set_value(k_head_stride, 6);
      c.set_value(k_seq_stride, 7);
      c.set_value(k_head_stride, 8);
      c.set_value(k_seq_stride, 9);
      c.set_value(scale, 10);
      c.dispatch_groups(MTLSizeMake(nh, 1, 1), MTLSizeMake(1024, 1, 1));
    }
    trace.add(tag + "attn", attn);

    gpu::Tensor hidden = trace.enabled ? buf({1, H}) : ((l % 2) ? xb : xa);
    dispatch_qmv_add(c, attn, w_.get(sa + "o_proj.weight"), w_.get(sa + "o_proj.scales"),
                     w_.get(sa + "o_proj.biases"), x, hidden, 1, H, nh * hd, qmv_shape);
    trace.add(tag + "hidden", hidden);

    gpu::Tensor mlp_in = buf({1, H});
    dispatch_rms(c, hidden, w_.get(p + ".post_attention_layernorm.weight"), mlp_in, 1, H, cfg_.eps);
    trace.add(tag + "post_attention_layernorm", mlp_in);

    gpu::Tensor gu_raw = buf({1, 2 * inter});
    dispatch_qmv2(c, mlp_in, w_.get(sm + "gate_proj.weight"), w_.get(sm + "gate_proj.scales"),
                  w_.get(sm + "gate_proj.biases"), w_.get(sm + "up_proj.weight"),
                  w_.get(sm + "up_proj.scales"), w_.get(sm + "up_proj.biases"), gu_raw, 1, inter,
                  inter, H, qmv_shape);
    gpu::Tensor gate = view(gu_raw, 0, {1, inter});
    gpu::Tensor up = view(gu_raw, inter, {1, inter});
    trace.add(tag + "gate", gate);
    trace.add(tag + "up", up);

    gpu::Tensor gu = buf({1, inter});
    dispatch_silu_mul(c, gate, up, gu, inter);
    trace.add(tag + "gate_up", gu);

    gpu::Tensor out = trace.enabled ? buf({1, H}) : ((l % 2) ? xa : xb);
    dispatch_qmv_add(c, gu, w_.get(sm + "down_proj.weight"), w_.get(sm + "down_proj.scales"),
                     w_.get(sm + "down_proj.biases"), hidden, out, 1, H, inter, qmv_shape);
    trace.add(tag + "out", out);
    x = out;
  }
  return x;
}

gpu::Tensor Decoder::forward(const gpu::Tensor &embeddings, int n, std::vector<KVCache> &caches) {
  if (n != 1) return forward_prefill(embeddings, n, caches);
  return use_fused ? forward_decode(embeddings, caches) : decode_layers_traced(embeddings, caches);
}

// Fused decode step (all 28 layers, one command buffer).
gpu::Tensor Decoder::forward_decode(const gpu::Tensor &embeddings, std::vector<KVCache> &caches) {
  gpu::Cmd c;
  gpu::Tensor x = decode_layers_fused(c, embeddings, caches);
  c.commit_and_wait();
  last_gpu_ms = c.gpu_ms();
  return x;
}

gpu::Tensor Decoder::decode_layers_traced(const gpu::Tensor &embeddings, std::vector<KVCache> &caches) {
  const int n = 1;
  auto &ctx = gpu::Context::get();
  const int H = cfg_.hidden, nh = cfg_.n_heads, nkv = cfg_.n_kv, hd = cfg_.head_dim;
  const int inter = cfg_.inter;
  if ((int)caches.size() != cfg_.n_layers) caches.assign(cfg_.n_layers, KVCache{});
  auto bf = [&](std::vector<int64_t> s) { return ctx.empty(s, gpu::DType::BF16); };

  gpu::Tensor x = embeddings;
  gpu::Cmd c;
  for (int l = 0; l < cfg_.n_layers; ++l) {
    const std::string p = "model.layers." + std::to_string(l);
    KVCache &cache = caches[l];
    const int pos = cache.len;
    if (pos + 1 > cache.cap) cache.reserve(((pos + 1) + 255) / 256 * 256, nkv, hd);
    const std::string tag = "L" + std::to_string(l) + "_";

    gpu::Tensor normed = bf({1, H});
    dispatch_rms(c, x, w_.get(p + ".input_layernorm.weight"), normed, 1, H, cfg_.eps);
    trace.add(tag + "input_layernorm", normed);

    gpu::Tensor q = bf({1, nh * hd}), k = bf({1, nkv * hd}), v = bf({1, nkv * hd});
    dispatch_qmv_fast(c, normed, w_.get(p + ".self_attn.q_proj.weight"),
                      w_.get(p + ".self_attn.q_proj.scales"), w_.get(p + ".self_attn.q_proj.biases"),
                      q, 1, nh * hd, H);
    dispatch_qmv_fast(c, normed, w_.get(p + ".self_attn.k_proj.weight"),
                      w_.get(p + ".self_attn.k_proj.scales"), w_.get(p + ".self_attn.k_proj.biases"),
                      k, 1, nkv * hd, H);
    dispatch_qmv_fast(c, normed, w_.get(p + ".self_attn.v_proj.weight"),
                      w_.get(p + ".self_attn.v_proj.scales"), w_.get(p + ".self_attn.v_proj.biases"),
                      v, 1, nkv * hd, H);
    trace.add(tag + "q_proj", q);
    trace.add(tag + "k_proj", k);
    trace.add(tag + "v_proj", v);

    gpu::Tensor qn = bf({nh, hd}), kn = bf({nkv, hd});
    dispatch_rms(c, q, w_.get(p + ".self_attn.q_norm.weight"), qn, nh, hd, cfg_.eps);
    dispatch_rms(c, k, w_.get(p + ".self_attn.k_norm.weight"), kn, nkv, hd, cfg_.eps);
    trace.add(tag + "q_norm", qn);
    trace.add(tag + "k_norm", kn);

    gpu::Tensor qr = bf({nh, hd}), kr = bf({nkv, hd});
    dispatch_rope(c, qn, qr, nh, hd, pos, cfg_.rope_theta);
    dispatch_rope(c, kn, kr, nkv, hd, pos, cfg_.rope_theta);
    trace.add(tag + "q_rope", qr);
    trace.add(tag + "k_rope", kr);

    // append to the KV cache
    c.use("dec_kv_store_bf16");
    c.set(kr, 0);
    c.set(cache.keys, 1);
    c.set_value(cache.cap, 2);
    c.set_value(pos, 3);
    c.set_value(hd, 4);
    c.dispatch_threads(MTLSizeMake(nkv * hd, 1, 1), MTLSizeMake(128, 1, 1));
    c.use("dec_kv_store_bf16");
    c.set(v, 0);
    c.set(cache.values, 1);
    c.set_value(cache.cap, 2);
    c.set_value(pos, 3);
    c.set_value(hd, 4);
    c.dispatch_threads(MTLSizeMake(nkv * hd, 1, 1), MTLSizeMake(128, 1, 1));
    cache.len = pos + 1;

    gpu::Tensor attn = bf({1, nh * hd});
    {
      int gqa = nh / nkv, N = cache.len;
      int64_t k_head_stride = (int64_t)cache.cap * hd, k_seq_stride = hd;
      float scale = 1.0f / std::sqrt((float)hd);
      c.use(sdpa_pipeline());
      c.set(qr, 0);
      c.set(cache.keys, 1);
      c.set(cache.values, 2);
      c.set(attn, 3);
      c.set_value(gqa, 4);
      c.set_value(N, 5);
      c.set_value(k_head_stride, 6);
      c.set_value(k_seq_stride, 7);
      c.set_value(k_head_stride, 8);
      c.set_value(k_seq_stride, 9);
      c.set_value(scale, 10);
      c.dispatch_groups(MTLSizeMake(nh, 1, 1), MTLSizeMake(1024, 1, 1));
    }
    trace.add(tag + "attn", attn);

    gpu::Tensor o = bf({1, H});
    dispatch_qmv_fast(c, attn, w_.get(p + ".self_attn.o_proj.weight"),
                      w_.get(p + ".self_attn.o_proj.scales"), w_.get(p + ".self_attn.o_proj.biases"),
                      o, 1, H, nh * hd);
    trace.add(tag + "o_proj", o);

    gpu::Tensor hidden = bf({1, H});
    dispatch_ew(c, "dec_add_bf16", x, &o, hidden, H);
    trace.add(tag + "hidden", hidden);

    gpu::Tensor mlp_in = bf({1, H});
    dispatch_rms(c, hidden, w_.get(p + ".post_attention_layernorm.weight"), mlp_in, 1, H, cfg_.eps);
    trace.add(tag + "post_attention_layernorm", mlp_in);

    gpu::Tensor gate = bf({1, inter}), up = bf({1, inter});
    dispatch_qmv_fast(c, mlp_in, w_.get(p + ".mlp.gate_proj.weight"),
                      w_.get(p + ".mlp.gate_proj.scales"), w_.get(p + ".mlp.gate_proj.biases"),
                      gate, 1, inter, H);
    dispatch_qmv_fast(c, mlp_in, w_.get(p + ".mlp.up_proj.weight"),
                      w_.get(p + ".mlp.up_proj.scales"), w_.get(p + ".mlp.up_proj.biases"),
                      up, 1, inter, H);
    trace.add(tag + "gate", gate);
    trace.add(tag + "up", up);

    gpu::Tensor sg = bf({1, inter}), gu = bf({1, inter});
    dispatch_ew(c, "dec_silu_bf16", gate, nullptr, sg, inter);
    dispatch_ew(c, "dec_mul_bf16", sg, &up, gu, inter);
    trace.add(tag + "silu_gate", sg);
    trace.add(tag + "gate_up", gu);

    gpu::Tensor down = bf({1, H});
    dispatch_qmv_fast(c, gu, w_.get(p + ".mlp.down_proj.weight"),
                      w_.get(p + ".mlp.down_proj.scales"), w_.get(p + ".mlp.down_proj.biases"),
                      down, 1, H, inter);
    trace.add(tag + "down", down);

    gpu::Tensor out = bf({1, H});
    dispatch_ew(c, "dec_add_bf16", hidden, &down, out, H);
    trace.add(tag + "out", out);
    x = out;
  }
  c.commit_and_wait();
  last_gpu_ms = c.gpu_ms();
  return x;
}

int Decoder::step(int id, std::vector<KVCache> &caches, gpu::Tensor *logits_out) {
  auto &ctx = gpu::Context::get();
  const int H = cfg_.hidden;
  const int N = (int)w_.get("model.embed_tokens.weight").shape[0];
  if (!emb_buf_.buf) {
    emb_buf_ = ctx.empty({1, H}, gpu::DType::BF16);
    logits_buf_ = ctx.empty({1, N}, gpu::DType::BF16);
    normed_buf_ = ctx.empty({1, H}, gpu::DType::BF16);
    argmax_buf_ = ctx.empty({1}, gpu::DType::I32);
  }
  gpu::Cmd c;
  embed_into(c, {id}, emb_buf_);
  gpu::Tensor h = decode_layers_fused(c, emb_buf_, caches);
  dispatch_rms(c, h, w_.get("model.norm.weight"), normed_buf_, 1, H, cfg_.eps);
  dispatch_qmv_fast(c, normed_buf_, w_.get("model.embed_tokens.weight"),
                    w_.get("model.embed_tokens.scales"), w_.get("model.embed_tokens.biases"),
                    logits_buf_, 1, N, H);
  dispatch_argmax(c, logits_buf_, argmax_buf_, N);
  c.commit_and_wait();
  last_gpu_ms = c.gpu_ms();
  trace.add("final_norm", normed_buf_);
  trace.add("logits", logits_buf_);
  if (logits_out) *logits_out = logits_buf_;
  return *argmax_buf_.ptr<int>();
}

gpu::Tensor Decoder::logits(const gpu::Tensor &hidden_row) {
  auto &ctx = gpu::Context::get();
  const int H = cfg_.hidden;
  gpu::Tensor normed = ctx.empty({1, H}, gpu::DType::BF16);
  const gpu::Tensor &W = w_.get("model.embed_tokens.weight");
  int N = (int)W.shape[0];
  gpu::Tensor out = ctx.empty({1, N}, gpu::DType::BF16);
  gpu::Cmd c;
  dispatch_rms(c, hidden_row, w_.get("model.norm.weight"), normed, 1, H, cfg_.eps);
  dispatch_qmv_fast(c, normed, W, w_.get("model.embed_tokens.scales"),
                    w_.get("model.embed_tokens.biases"), out, 1, N, H);
  c.commit_and_wait();
  last_gpu_ms = c.gpu_ms();
  trace.add("final_norm", normed);
  trace.add("logits", out);
  return out;
}


void Decoder::qmm_into(gpu::Cmd &c, const gpu::Tensor &x, const std::string &prefix, int M, int N,
                       int K, gpu::Tensor &y) {
  const gpu::Tensor &w = w_.get(prefix + ".weight");
  const gpu::Tensor &s = w_.get(prefix + ".scales");
  const gpu::Tensor &b = w_.get(prefix + ".biases");
  if (M == 1) {
    dispatch_qmv_fast(c, x, w, s, b, y, M, N, K);
    return;
  }
  if (M < kQmvBatchLimit) {
    dispatch_qmv_wide(c, x, w, s, b, y, M, N, K);
    return;
  }
  QmmPlan plan = qmm_plan(M, N, K, 64);
  if (plan.split_k <= 1) {
    dispatch_qmm_t(c, x, w, s, b, y, M, N, K);
    return;
  }
  int64_t need = (int64_t)plan.split_k * M * N;
  gpu::Tensor temp;
  if (trace.enabled) {
    temp = gpu::Context::get().empty({plan.split_k, M, N}, gpu::DType::BF16);
  } else {
    if (!splitk_temp_.buf || splitk_temp_.numel() < need)
      splitk_temp_ = gpu::Context::get().empty({need}, gpu::DType::BF16);
    temp = splitk_temp_;
    temp.shape = {plan.split_k, M, N};
  }
  dispatch_qmm_splitk(c, x, w, s, b, temp, M, N, K, plan);
  c.use("dec_splitk_sum_bf16");
  c.set(temp, 0);
  c.set(y, 1);
  c.set_value(plan.split_k, 2);
  int64_t stride = (int64_t)M * N;
  c.set_value(stride, 3);
  int64_t total = (int64_t)M * N;
  c.dispatch_threads(MTLSizeMake((size_t)total, 1, 1), MTLSizeMake(256, 1, 1));
}

gpu::Tensor Decoder::prefill_layers(gpu::Cmd &c, const gpu::Tensor &embeddings, int n,
                                    std::vector<KVCache> &caches) {
  auto &ctx = gpu::Context::get();
  const int H = cfg_.hidden, nh = cfg_.n_heads, nkv = cfg_.n_kv, hd = cfg_.head_dim;
  const int inter = cfg_.inter;
  if ((int)caches.size() != cfg_.n_layers) caches.assign(cfg_.n_layers, KVCache{});
  auto bf = [&](std::vector<int64_t> s) {
    return trace.enabled ? ctx.empty(s, gpu::DType::BF16) : buf(s);
  };
  const int half = hd / 2;
  uint32_t nheads_u = (uint32_t)nh;
  int gqa = nh / nkv;

  arena_off_ = 0;
  arena_keep_ = 0;
  gpu::Tensor xa, xb;  // ping-pong residual stream, kept across layers
  if (!trace.enabled) {
    xa = buf({n, H});
    xb = buf({n, H});
    arena_keep_ = arena_off_;
  }
  gpu::Tensor x = embeddings;
  for (int l = 0; l < cfg_.n_layers; ++l) {
    // All 28 layers are recorded into one command buffer: the encoder is
    // MTLDispatchTypeSerial, so dispatches never overlap and a later layer reusing the arena
    // memory of an earlier one is safe (the fused decode path already relies on this). One
    // command buffer per layer cost 28 commit+wait round trips per update: ~14 ms of CPU
    // latency and ~4 ms of GPU time (0.13 ms per commit+wait).
    // Peak memory is unchanged: the arena is reused per layer (arena_reset below).
    const std::string p = "model.layers." + std::to_string(l);
    @autoreleasepool {  // per layer: host-side temporaries of the finished layer are released here
    KVCache &cache = caches[l];
    const int pos = cache.len;
    if (pos + n > cache.cap) cache.reserve(((pos + n) + 255) / 256 * 256, nkv, hd);
    const std::string tag = "L" + std::to_string(l) + "_";
    arena_reset();

    gpu::Tensor normed = bf({n, H});
    dispatch_rms(c, x, w_.get(p + ".input_layernorm.weight"), normed, n, H, cfg_.eps);
    trace.add(tag + "input_layernorm", normed);

    gpu::Tensor q = bf({n, nh * hd}), k = bf({n, nkv * hd}), v = bf({n, nkv * hd});
    qmm_into(c, normed, p + ".self_attn.q_proj", n, nh * hd, H, q);
    qmm_into(c, normed, p + ".self_attn.k_proj", n, nkv * hd, H, k);
    qmm_into(c, normed, p + ".self_attn.v_proj", n, nkv * hd, H, v);
    trace.add(tag + "q_proj", q);
    trace.add(tag + "k_proj", k);
    trace.add(tag + "v_proj", v);

    gpu::Tensor qn = bf({n * nh, hd}), kn = bf({n * nkv, hd});
    dispatch_rms(c, q, w_.get(p + ".self_attn.q_norm.weight"), qn, n * nh, hd, cfg_.eps);
    dispatch_rms(c, k, w_.get(p + ".self_attn.k_norm.weight"), kn, n * nkv, hd, cfg_.eps);
    trace.add(tag + "q_norm", qn);
    trace.add(tag + "k_norm", kn);

    // rope: in [n, heads, hd] contiguous, out [heads, n, hd] contiguous
    gpu::Tensor qr = bf({nh, n, hd}), kr = bf({nkv, n, hd});
    int off = pos;
    float scale = 1.0f, base = std::log2(cfg_.rope_theta);
    int64_t in_strides[3] = {hd, (int64_t)nh * hd, 1};
    int64_t out_strides[3] = {(int64_t)n * hd, hd, 1};
    int64_t offset_stride = 0;
    int bd[3];
    block_dims(half, n, (nh + 3) / 4, bd);
    c.use("dec_rope_bf16");
    c.set(qn, 0);
    c.set(qr, 1);
    c.set_bytes(&off, sizeof(off), 2);
    c.set_bytes(&scale, sizeof(scale), 3);
    c.set_bytes(in_strides, sizeof(in_strides), 4);
    c.set_bytes(out_strides, sizeof(out_strides), 5);
    c.set_bytes(&offset_stride, sizeof(offset_stride), 6);
    c.set_bytes(&nheads_u, sizeof(nheads_u), 7);
    c.set_bytes(&base, sizeof(base), 10);
    c.dispatch_threads(MTLSizeMake(half, n, (nh + 3) / 4), MTLSizeMake(bd[0], bd[1], bd[2]));
    int64_t kin_strides[3] = {hd, (int64_t)nkv * hd, 1};
    int64_t kout_strides[3] = {(int64_t)n * hd, hd, 1};
    uint32_t nkv_u = (uint32_t)nkv;
    block_dims(half, n, (nkv + 3) / 4, bd);
    c.use("dec_rope_bf16");
    c.set(kn, 0);
    c.set(kr, 1);
    c.set_bytes(&off, sizeof(off), 2);
    c.set_bytes(&scale, sizeof(scale), 3);
    c.set_bytes(kin_strides, sizeof(kin_strides), 4);
    c.set_bytes(kout_strides, sizeof(kout_strides), 5);
    c.set_bytes(&offset_stride, sizeof(offset_stride), 6);
    c.set_bytes(&nkv_u, sizeof(nkv_u), 7);
    c.set_bytes(&base, sizeof(base), 10);
    c.dispatch_threads(MTLSizeMake(half, n, (nkv + 3) / 4), MTLSizeMake(bd[0], bd[1], bd[2]));
    trace.add(tag + "q_rope", qr);
    trace.add(tag + "k_rope", kr);

    // KV cache rows
    c.use("dec_kv_store_rows_bf16");
    c.set(kr, 0);
    c.set(cache.keys, 1);
    c.set_value(cache.cap, 2);
    c.set_value(pos, 3);
    c.set_value(hd, 4);
    c.set_value(nkv, 5);
    c.set_value(n, 6);
    c.set_value(1, 7);
    c.dispatch_threads(MTLSizeMake((size_t)n * nkv * hd, 1, 1), MTLSizeMake(256, 1, 1));
    c.use("dec_kv_store_rows_bf16");
    c.set(v, 0);
    c.set(cache.values, 1);
    c.set_value(cache.cap, 2);
    c.set_value(pos, 3);
    c.set_value(hd, 4);
    c.set_value(nkv, 5);
    c.set_value(n, 6);
    c.set_value(0, 7);
    c.dispatch_threads(MTLSizeMake((size_t)n * nkv * hd, 1, 1), MTLSizeMake(256, 1, 1));
    cache.len = pos + n;

    // steel causal attention; output written in [n, nh, hd] layout directly
    gpu::Tensor attn = bf({n, nh * hd});
    {
      struct AttnParams {
        int B, H, D, qL, kL, gqa_factor;
        float scale;
        int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
        int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
      } pr;
      int kL = cache.len;
      pr.B = 1; pr.H = nh; pr.D = hd; pr.qL = n; pr.kL = kL; pr.gqa_factor = gqa;
      pr.scale = 1.0f / std::sqrt((float)hd);
      pr.NQ = (n + 31) / 32; pr.NK = (kL + 15) / 16;
      pr.NQ_aligned = n / 32; pr.NK_aligned = kL / 16;
      pr.qL_rem = n - pr.NQ_aligned * 32; pr.kL_rem = kL - pr.NK_aligned * 16;
      pr.qL_off = kL - n;
      pr.Q_strides[0] = (int64_t)nh * n * hd; pr.Q_strides[1] = (int64_t)n * hd; pr.Q_strides[2] = hd;
      pr.K_strides[0] = (int64_t)nkv * cache.cap * hd; pr.K_strides[1] = (int64_t)cache.cap * hd;
      pr.K_strides[2] = hd;
      pr.V_strides[0] = pr.K_strides[0]; pr.V_strides[1] = pr.K_strides[1]; pr.V_strides[2] = hd;
      pr.O_strides[0] = (int64_t)nh * n * hd; pr.O_strides[1] = hd; pr.O_strides[2] = (int64_t)nh * hd;
      if (n <= 8) {
        // MLX routes qL <= 8 to sdpa_vector (do_causal when qL > 1). Its output is
        // head-major [nh,n,hd], so transpose into the seq-major o_proj input.
        gpu::Tensor hm = bf({nh, n, hd});
        int64_t khs = (int64_t)cache.cap * hd, kss = hd;
        float sc = 1.0f / std::sqrt((float)hd);
        c.use(sdpa_pipeline_causal());
        c.set(qr, 0);
        c.set(cache.keys, 1);
        c.set(cache.values, 2);
        c.set(hm, 3);
        c.set_value(gqa, 4);
        c.set_value(kL, 5);
        c.set_value(khs, 6);
        c.set_value(kss, 7);
        c.set_value(khs, 8);
        c.set_value(kss, 9);
        c.set_value(sc, 10);
        c.dispatch_groups(MTLSizeMake(nh, n, 1), MTLSizeMake(1024, 1, 1));
        c.use("dec_transpose_nh_bf16");
        c.set(hm, 0);
        c.set(attn, 1);
        c.set_value(n, 2);
        c.set_value(nh, 3);
        c.set_value(hd, 4);
        int64_t tot = (int64_t)n * nh * hd;
        c.dispatch_threads(MTLSizeMake((size_t)tot, 1, 1), MTLSizeMake(256, 1, 1));
      } else {
        c.use(steel_attention_pipeline(n % 32 == 0, kL % 16 == 0));
        c.set(qr, 0);
        c.set(cache.keys, 1);
        c.set(cache.values, 2);
        c.set(attn, 3);
        c.set_bytes(&pr, sizeof(pr), 4);
        c.dispatch_groups(MTLSizeMake(pr.NQ, nh, 1), MTLSizeMake(32, 4, 1));
      }
    }
    trace.add(tag + "attn", attn);

    gpu::Tensor o = bf({n, H});
    qmm_into(c, attn, p + ".self_attn.o_proj", n, H, nh * hd, o);
    trace.add(tag + "o_proj", o);

    gpu::Tensor hidden = bf({n, H});
    dispatch_ew(c, "dec_add_bf16", x, &o, hidden, (int64_t)n * H);
    trace.add(tag + "hidden", hidden);

    gpu::Tensor mlp_in = bf({n, H});
    dispatch_rms(c, hidden, w_.get(p + ".post_attention_layernorm.weight"), mlp_in, n, H, cfg_.eps);
    trace.add(tag + "post_attention_layernorm", mlp_in);

    gpu::Tensor gate = bf({n, inter}), up = bf({n, inter});
    qmm_into(c, mlp_in, p + ".mlp.gate_proj", n, inter, H, gate);
    qmm_into(c, mlp_in, p + ".mlp.up_proj", n, inter, H, up);
    trace.add(tag + "gate", gate);
    trace.add(tag + "up", up);

    gpu::Tensor sg = bf({n, inter}), gu = bf({n, inter});
    dispatch_ew(c, "dec_silu_bf16", gate, nullptr, sg, (int64_t)n * inter);
    dispatch_ew(c, "dec_mul_bf16", sg, &up, gu, (int64_t)n * inter);
    trace.add(tag + "silu_gate", sg);
    trace.add(tag + "gate_up", gu);

    gpu::Tensor down = bf({n, H});
    qmm_into(c, gu, p + ".mlp.down_proj", n, H, inter, down);
    trace.add(tag + "down", down);

    gpu::Tensor out = trace.enabled ? bf({n, H}) : ((l % 2) ? xa : xb);
    dispatch_ew(c, "dec_add_bf16", hidden, &down, out, (int64_t)n * H);
    trace.add(tag + "out", out);
    x = out;
    }
  }
  return x;
}

gpu::Tensor Decoder::forward_prefill(const gpu::Tensor &embeddings, int n,
                                     std::vector<KVCache> &caches) {
  gpu::Cmd c;
  gpu::Tensor x = prefill_layers(c, embeddings, n, caches);
  c.commit_and_wait();
  last_gpu_ms = c.gpu_ms();
  return x;
}

// ---- Streaming-verification API ----

void Decoder::ensure_rows_buffers(int k, int vocab) {
  if (k <= rows_cap_ && vocab == rows_vocab_) return;
  auto &ctx = gpu::Context::get();
  const int cap = k > rows_cap_ ? k : rows_cap_;
  rows_logits_ = ctx.empty({cap, vocab}, gpu::DType::BF16);
  rows_normed_ = ctx.empty({cap, cfg_.hidden}, gpu::DType::BF16);
  rows_idx_ = ctx.empty({cap}, gpu::DType::U32);
  rows_margin_ = ctx.empty({cap}, gpu::DType::F32);
  rows_cap_ = cap;
  rows_vocab_ = vocab;
}

std::vector<Decoder::RowOut> Decoder::forward_rows(const gpu::Tensor &embeddings, int n,
                                                   std::vector<KVCache> &caches, int offset,
                                                   int k, gpu::Tensor *logits_out) {
  gpu::StageScope stage(2);
  std::vector<RowOut> out;
  if (n <= 0) return out;
  if (k <= 0 || k > n) k = n;
  if (offset < 0) offset = 0;
  const int H = cfg_.hidden, nkv = cfg_.n_kv, hd = cfg_.head_dim;
  if ((int)caches.size() != cfg_.n_layers) caches.assign(cfg_.n_layers, KVCache{});
  for (auto &cache : caches) {  // roll back to the reusable prefix, grow if needed
    if (offset + n > cache.cap) cache.reserve(((offset + n) + 255) / 256 * 256, nkv, hd);
    cache.len = offset;
  }
  const int N = (int)w_.get("model.embed_tokens.weight").shape[0];
  ensure_rows_buffers(k, N);
  gpu::Tensor normed = rows_normed_;
  normed.shape = {k, H};
  gpu::Tensor lg = rows_logits_;
  lg.shape = {k, N};
  gpu::Tensor idx = rows_idx_;
  idx.shape = {k};
  gpu::Tensor mg = rows_margin_;
  mg.shape = {k};
  gpu::Cmd c;
  gpu::Tensor h = prefill_layers(c, embeddings, n, caches);
  gpu::Tensor last = h;  // last k rows of the last-layer hidden states
  last.offset += (size_t)(n - k) * H * 2;
  last.shape = {k, H};
  dispatch_rms(c, last, w_.get("model.norm.weight"), normed, k, H, cfg_.eps);
  qmm_into(c, normed, "model.embed_tokens", k, N, H, lg);
  dispatch_top2(c, lg, idx, mg, k, N);
  c.commit_and_wait();
  last_gpu_ms = c.gpu_ms();
  const uint32_t *ip = idx.ptr<uint32_t>();
  const float *mp = mg.ptr<float>();
  out.resize(k);
  for (int i = 0; i < k; ++i) {
    out[i].argmax = (int)ip[i];
    out[i].margin = mp[i];
  }
  if (logits_out) *logits_out = lg;
  return out;
}

int Decoder::step_margin(int id, std::vector<KVCache> &caches, float *margin,
                         gpu::Tensor *logits_out) {
  gpu::StageScope stage(3);
  auto &ctx = gpu::Context::get();
  const int H = cfg_.hidden;
  const int N = (int)w_.get("model.embed_tokens.weight").shape[0];
  if (!emb_buf_.buf) {
    emb_buf_ = ctx.empty({1, H}, gpu::DType::BF16);
    logits_buf_ = ctx.empty({1, N}, gpu::DType::BF16);
    normed_buf_ = ctx.empty({1, H}, gpu::DType::BF16);
    argmax_buf_ = ctx.empty({1}, gpu::DType::I32);
  }
  if (!margin_buf_.buf) margin_buf_ = ctx.empty({1}, gpu::DType::F32);
  gpu::Cmd c;
  embed_into(c, {id}, emb_buf_);
  gpu::Tensor h = decode_layers_fused(c, emb_buf_, caches);
  dispatch_rms(c, h, w_.get("model.norm.weight"), normed_buf_, 1, H, cfg_.eps);
  dispatch_qmv_fast(c, normed_buf_, w_.get("model.embed_tokens.weight"),
                    w_.get("model.embed_tokens.scales"), w_.get("model.embed_tokens.biases"),
                    logits_buf_, 1, N, H);
  dispatch_top2(c, logits_buf_, argmax_buf_, margin_buf_, 1, N);
  c.commit_and_wait();
  last_gpu_ms = c.gpu_ms();
  trace.add("final_norm", normed_buf_);
  trace.add("logits", logits_buf_);
  if (logits_out) *logits_out = logits_buf_;
  if (margin) *margin = *margin_buf_.ptr<float>();
  return (int)*argmax_buf_.ptr<uint32_t>();
}

void Decoder::reserve_kv(std::vector<KVCache> &caches, int cap) {
  if ((int)caches.size() != cfg_.n_layers) caches.assign(cfg_.n_layers, KVCache{});
  for (auto &cache : caches) cache.reserve(cap, cfg_.n_kv, cfg_.head_dim);
}

void Decoder::rollback(std::vector<KVCache> &caches, int p) {
  for (auto &cache : caches) cache.len = p < 0 ? 0 : (p > cache.len ? cache.len : p);
}

gpu::Tensor Decoder::embedding_buffer(int rows) {
  return gpu::Context::get().empty({rows, cfg_.hidden}, gpu::DType::BF16);
}

void Decoder::splice_embeddings(gpu::Cmd &c, gpu::Tensor &emb, const gpu::Tensor &f32_rows,
                                int row_offset, int rows) {
  const int H = cfg_.hidden;
  if (f32_rows.numel() < (int64_t)rows * H)
    throw std::runtime_error("splice_embeddings: source rows too small");
  if ((int64_t)(row_offset + rows) * H > emb.numel())
    throw std::runtime_error("splice_embeddings: destination rows too small");
  uint32_t cols = (uint32_t)H, row0 = (uint32_t)row_offset;
  c.use("dec_splice_f32_bf16");
  c.set(f32_rows, 0);
  c.set(emb, 1);
  c.set_value(cols, 2);
  c.set_value(row0, 3);
  const int64_t total = (int64_t)rows * H;
  c.dispatch_threads(MTLSizeMake((size_t)total, 1, 1), MTLSizeMake(256, 1, 1));
}

void Decoder::splice_embeddings(gpu::Tensor &emb, const gpu::Tensor &f32_rows, int row_offset,
                                int rows) {
  gpu::Cmd c;
  splice_embeddings(c, emb, f32_rows, row_offset, rows);
  c.commit_and_wait();
}

void Decoder::embed_rows_into(gpu::Cmd &c, const std::vector<int> &ids, gpu::Tensor &emb,
                              int row_offset) {
  const int n = (int)ids.size();
  if (n == 0) return;
  if ((int64_t)(row_offset + n) * cfg_.hidden > emb.numel())
    throw std::runtime_error("embed_rows_into: destination rows too small");
  gpu::Tensor view = emb;
  view.offset += (size_t)row_offset * cfg_.hidden * 2;
  view.shape = {(int64_t)n, (int64_t)cfg_.hidden};
  embed_into(c, ids, view);
}

}  // namespace dec
