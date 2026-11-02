// Audio encoder for Qwen3-ASR (owner: encoder worker). See encoder.h for the reference.
#include "encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

namespace enc {
namespace {

constexpr int kConvFramesMax = 13;  // conv_frames(100)
constexpr int kConvOutK = 480 * 16;

int FloorDiv(int a, int b) {
  const int q = a / b;
  return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

int RemainderTokens(int frames) {
  const int rem = frames % kChunkFrames;
  const int after_first = FloorDiv(rem - 1, 2) + 1;
  return FloorDiv(FloorDiv(after_first - 1, 2) + 1 - 1, 2) + 1;
}

struct ConvParams {
  int H, W, C;  // input spatial size and channels
  int Ho, Wo;   // output spatial size
  int N;        // output channels
  int K;        // 9 * C
  int xs;       // input frame stride
};

// Mirrors ConvSgParams in kernels/enc_encoder.metal (the simdgroup MMA conv).
struct ConvSgParams {
  int H, W, C;
  int Ho, Wo;
  int N;
  int xs;
};

int NextPow2(int x) {
  int p = 1;
  while (p < x) p <<= 1;
  return p;
}

// Scratch for the split-K partials; grows on demand, reused across calls.
gpu::Tensor &SplitScratch(size_t n) {
  static gpu::Tensor s;
  if (s.buf == nil || (size_t)s.numel() < n)
    s = gpu::Context::get().empty({(int64_t)n}, gpu::DType::F32);
  return s;
}

// MLX steel split-K addmm numerics (see kernels/enc_encoder.metal), computed with 32x32
// simdgroup MMA tiles: bit-identical to MLX's own GEMM and the fastest tile on the streaming
// window shapes.
void Gemm(gpu::Cmd &c, const gpu::Tensor &A, const gpu::Tensor &B, const gpu::Tensor &bias,
          gpu::Tensor &C) {
  const int M = (int)A.dim(0), K = (int)A.dim(1), N = (int)B.dim(0);
  // MLX matmul.cpp: split-K only when ceil(M/16)*ceil(N/16) <= 2048 (devc 's'), K/16 >= 8
  // and K >= max(M, N); otherwise the regular single-pass steel GEMM (ascending k).
  const int mt = (M + 15) / 16, nt = (N + 15) / 16;
  const bool use_splitk = (mt * nt) <= 2048 && (K / 16) >= 8 && K >= std::max(M, N);
  const int tk = K / 16, tm = (M + 31) / 32, tn = (N + 31) / 32;
  const int parts = use_splitk ? std::min(std::max(2, NextPow2(tk / (tm * tn))), 32) : 1;
  const int psize = ((K / 16) / parts) * 16;
  const int fuse = parts == 1 ? 1 : 0;
  gpu::Tensor &Cs = fuse ? C : SplitScratch((size_t)parts * M * N);
  const int bm = 32, bn = 32;
  c.use("enc_gemm_sg32");
  c.set(A, 0);
  c.set(B, 1);
  c.set(Cs, 2);
  c.set(bias, 3);
  c.set_value(M, 4);
  c.set_value(N, 5);
  c.set_value(K, 6);
  c.set_value(psize, 7);
  c.set_value(fuse, 8);
  c.dispatch_groups(MTLSizeMake((N + bn - 1) / bn, (M + bm - 1) / bm, parts), MTLSizeMake(128, 1, 1));
  if (!fuse) {
    c.use("enc_gemm_reduce");
    c.set(Cs, 0);
    c.set(bias, 1);
    c.set(C, 2);
    c.set_value(M, 3);
    c.set_value(N, 4);
    c.set_value(parts, 5);
    c.dispatch_threads(MTLSizeMake((size_t)M * N, 1, 1), MTLSizeMake(256, 1, 1));
  }
  c.barrier();
}

void Conv(gpu::Cmd &c, const gpu::Tensor &X, int H, int W, int C, int xs, const gpu::Tensor &Wt,
          const gpu::Tensor &bias, gpu::Tensor &Y) {
  ConvParams p{H, W, C, (H - 1) / 2 + 1, (W - 1) / 2 + 1, (int)Wt.dim(0), 9 * C, xs};
  c.use("enc_conv");
  c.set(X, 0);
  c.set(Wt, 1);
  c.set(bias, 2);
  c.set(Y, 3);
  c.set_bytes(&p, sizeof(p), 4);
  c.dispatch_groups(MTLSizeMake((p.N + 63) / 64, (p.Ho * p.Wo + 63) / 64, 1), MTLSizeMake(16, 16, 1));
  c.barrier();
}

void Gelu(gpu::Cmd &c, gpu::Tensor &x, uint32_t n) {
  c.use("enc_gelu");
  c.set(x, 0);
  c.set_value(n, 1);
  c.dispatch_threads(MTLSizeMake(n, 1, 1), MTLSizeMake(256, 1, 1));
  c.barrier();
}

void Add(gpu::Cmd &c, gpu::Tensor &a, const gpu::Tensor &b, uint32_t n) {
  c.use("enc_add");
  c.set(a, 0);
  c.set(b, 1);
  c.set_value(n, 2);
  c.dispatch_threads(MTLSizeMake(n, 1, 1), MTLSizeMake(256, 1, 1));
  c.barrier();
}

void LayerNorm(gpu::Cmd &c, const gpu::Tensor &x, const gpu::Tensor &w, const gpu::Tensor &b,
               gpu::Tensor &out) {
  const int rows = (int)x.dim(0);
  const uint32_t axis = (uint32_t)x.dim(-1);
  float eps = kLayerNormEps;
  c.use("enc_layer_norm");
  c.set(x, 0);
  c.set(w, 1);
  c.set(b, 2);
  c.set(out, 3);
  c.set_value(eps, 4);
  c.set_value(axis, 5);
  c.dispatch_threads(MTLSizeMake((size_t)rows * 128, 1, 1), MTLSizeMake(128, 1, 1));
  c.barrier();
}

void Attention(gpu::Cmd &c, const gpu::Tensor &q, const gpu::Tensor &k, const gpu::Tensor &v,
               gpu::Tensor &out, int L) {
  int l = L;
  float scale = 0.125f;  // 64 ** -0.5
  c.use("enc_attn");
  c.set(q, 0);
  c.set(k, 1);
  c.set(v, 2);
  c.set(out, 3);
  c.set_value(l, 4);
  c.set_value(scale, 5);
  c.set_value(kDModel, 6);
  c.dispatch_groups(MTLSizeMake(kHeads, (L + 7) / 8, 1), MTLSizeMake(256, 1, 1));
  c.barrier();
}

// One command buffer plus accumulated GPU time; flush() commits and starts a new one.
// NOTE: Metal's command queue allows only ~64 *uncommitted* command buffers; an abandoned
// (never committed) buffer permanently keeps its slot, so the queue deadlocks once enough of
// them pile up. flush() therefore creates a fresh buffer for the next phase, and the destructor
// commits the leftover (empty) one so no buffer is ever abandoned.
struct Runner {
  gpu::Cmd c;
  double gpu_ms = 0;
  void flush() {
    c.commit_and_wait();
    gpu_ms += c.gpu_ms();
    c.reset();
  }
  ~Runner() { c.commit_and_wait(); }  // empty tail buffer: frees the in-flight slot
};

}  // namespace

void conv_f32(gpu::Cmd &c, const gpu::Tensor &X, int H, int W, int C, int xs,
              const gpu::Tensor &Wt, const gpu::Tensor &bias, gpu::Tensor &Y) {
  // Simdgroup MMA tiles (32x32, k step 32). The tap-structured A gather needs C to be a multiple
  // of the k step; conv1 has C = 1 and uses the fma-tile kernel. Both are bit-identical.
  if (C % 32 == 0) {
    const int bm = 32, bn = 32;
    ConvSgParams p{H, W, C, (H - 1) / 2 + 1, (W - 1) / 2 + 1, (int)Wt.dim(0), xs};
    c.use("enc_conv_sg1");
    c.set(X, 0);
    c.set(Wt, 1);
    c.set(Y, 2);
    c.set(bias, 3);
    c.set_bytes(&p, sizeof(p), 4);
    c.dispatch_groups(MTLSizeMake((p.N + bn - 1) / bn, (p.Ho * p.Wo + bm - 1) / bm, 1),
                      MTLSizeMake(128, 1, 1));
    c.barrier();
    return;
  }
  Conv(c, X, H, W, C, xs, Wt, bias, Y);
}

int conv_frames(int frames) {
  for (int i = 0; i < 3; ++i) frames = (frames - 1) / 2 + 1;
  return frames;
}

int swift_token_count(int frames) {
  const float chunks = (float)frames / (float)kChunkFrames;
  return (int)std::trunc((float)RemainderTokens(frames) + chunks * 13.0f);
}

struct Encoder::Impl {
  const Weights &w;
  std::map<std::string, gpu::Tensor> f32cache;
  std::vector<float> pos;  // [kConvFramesMax][kDModel] sinusoidal positions
  // Per-chunk conv front-end cache: chunk index -> [conv_frames(100), kDModel] f32 with the
  // sinusoidal positions already added. A full 100-frame chunk's conv output depends only on
  // its own frames (the reference pads to the longest chunk, and a full chunk is always the
  // longest), so it can be reused across window calls. Cleared when the mel buffer changes.
  std::map<int, gpu::Tensor> conv_cache;
  int cache_hits = 0, cache_miss = 0;
  const void *conv_cache_mel = nullptr;
  int conv_cache_frames = -1;

  // Persistent workspaces. Every encode_window call used to allocate and zero ~13 MB of fresh
  // MTLBuffers (15 ctx.empty calls); the shapes repeat on every update (only the token count L
  // grows), so the buffers are kept and grown geometrically. Every kernel writes the elements it
  // reads, so stale contents are harmless (the positions buffer is memcpy'd and the chunk pad
  // buffer is memset before use, exactly as before).
  gpu::Tensor ws_y1, ws_y2, ws_y3, ws_packed, ws_cout, ws_pos, ws_padbuf;
  gpu::Tensor ws_hidden, ws_normed, ws_q, ws_k, ws_v, ws_att, ws_proj, ws_mlp;
  int ws_rows = 0;
  void ensure_ws(int L) {
    auto &ctx = gpu::Context::get();
    if (!ws_y1.buf) {
      ws_y1 = ctx.empty({64, 50, 480}, gpu::DType::F32);
      ws_y2 = ctx.empty({32, 25, 480}, gpu::DType::F32);
      ws_y3 = ctx.empty({16, kConvFramesMax, 480}, gpu::DType::F32);
      ws_packed = ctx.empty({kConvFramesMax, kConvOutK}, gpu::DType::F32);
      ws_cout = ctx.empty({kConvFramesMax, kDModel}, gpu::DType::F32);
      ws_pos = ctx.empty({kConvFramesMax, kDModel}, gpu::DType::F32);
      ws_padbuf = ctx.empty({kMelBins, kChunkFrames}, gpu::DType::F32);
    }
    if (L > ws_rows) {
      ws_rows = std::max(L, ws_rows + ws_rows / 2 + 8);
      ws_hidden = ctx.empty({ws_rows, kDModel}, gpu::DType::F32);
      ws_normed = ctx.empty({ws_rows, kDModel}, gpu::DType::F32);
      ws_q = ctx.empty({ws_rows, kDModel}, gpu::DType::F32);
      ws_k = ctx.empty({ws_rows, kDModel}, gpu::DType::F32);
      ws_v = ctx.empty({ws_rows, kDModel}, gpu::DType::F32);
      ws_att = ctx.empty({ws_rows, kDModel}, gpu::DType::F32);
      ws_proj = ctx.empty({ws_rows, kDModel}, gpu::DType::F32);
      ws_mlp = ctx.empty({ws_rows, kMlp}, gpu::DType::F32);
    }
  }

  explicit Impl(const Weights &ww) : w(ww) {
    const int half = kDModel / 2;
    const double step = std::log(10000.0) / (double)(half - 1);
    const float neg_step = (float)(-step);
    std::vector<float> inv(half);
    for (int j = 0; j < half; ++j) inv[j] = std::exp(neg_step * (float)j);
    pos.assign((size_t)kConvFramesMax * kDModel, 0.0f);
    for (int i = 0; i < kConvFramesMax; ++i) {
      for (int j = 0; j < half; ++j) {
        const float t = (float)i * inv[j];
        pos[(size_t)i * kDModel + j] = std::sin(t);
        pos[(size_t)i * kDModel + half + j] = std::cos(t);
      }
    }
  }

  const gpu::Tensor &bf(const std::string &name) { return w.get("audio_tower." + name); }

  // bf16 -> float32 copy, exactly as MLX's dtype promotion does (lossless).
  const gpu::Tensor &f32(const std::string &name) {
    auto it = f32cache.find(name);
    if (it != f32cache.end()) return it->second;
    if (!w.has("audio_tower." + name)) {  // e.g. conv_out has no bias: zeros
      return f32cache.emplace(name, gpu::Context::get().empty({kDModel}, gpu::DType::F32))
          .first->second;
    }
    const gpu::Tensor &src = bf(name);
    gpu::Tensor dst = gpu::Context::get().empty({src.numel()}, gpu::DType::F32);
    const uint16_t *p = src.ptr<uint16_t>();
    float *q = dst.ptr<float>();
    for (int64_t i = 0; i < src.numel(); ++i) q[i] = gpu::bf16_to_f32(p[i]);
    return f32cache.emplace(name, dst).first->second;
  }
};

namespace {
// Reads the encoder sizes from the checkpoint and rejects shapes the kernels do not handle.
const Weights &configure(const Weights &w) {
  const json::Value &a = w.config["thinker_config"]["audio_config"];
  const int d = (int)a["d_model"].as_int(), heads = (int)a["encoder_attention_heads"].as_int();
  if (heads <= 0 || d != heads * kHeadDim || (int)a["downsample_hidden_size"].as_int() != 480 ||
      (int)a["num_mel_bins"].as_int() != kMelBins || (int)a["n_window"].as_int() * 2 != kChunkFrames ||
      (int)a["n_window_infer"].as_int() != kChunkFrames * kChunksPerWindow)
    throw std::runtime_error("unsupported audio encoder configuration");
  kDModel = d;
  kHeads = heads;
  kLayers = (int)a["encoder_layers"].as_int();
  kMlp = (int)a["encoder_ffn_dim"].as_int();
  kOutputDim = (int)a["output_dim"].as_int();
  return w;
}
}  // namespace

Encoder::Encoder(const Weights &w) : impl_(new Impl(configure(w))) {}

Encoder::~Encoder() = default;

gpu::Tensor Encoder::encode_window(const gpu::Tensor &mel, int frames_total, int f0, int f1,
                                  int longest_chunk) {
  gpu::StageScope stage(1);
  const double t0 = gpu::now_ms();
  if (f1 <= f0) throw std::runtime_error("encode_window: empty frame range");
  if (f0 < 0 || f1 > frames_total) throw std::runtime_error("encode_window: range out of bounds");
  auto &ctx = gpu::Context::get();
  Impl &im = *impl_;

  const int nchunks = (f1 - f0 + kChunkFrames - 1) / kChunkFrames;
  std::vector<int> keeps(nchunks), lens(nchunks);
  int longest = std::min(std::max(longest_chunk, 0), kChunkFrames);
  for (int i = 0; i < nchunks; ++i) {
    lens[i] = std::min(kChunkFrames, f1 - (f0 + i * kChunkFrames));
    longest = std::max(longest, lens[i]);
  }
  // The reference pads every chunk to the longest chunk of the whole clip, so the conv
  // output (and the sinusoidal positions) always have conv_frames(clip longest) rows; a
  // chunk keeps min(SwiftTokenCount(len), conv_frames) of them.
  const int conv_rows = conv_frames(longest);
  int L = 0;
  for (int i = 0; i < nchunks; ++i) {
    keeps[i] = std::min(swift_token_count(lens[i]), conv_rows);
    L += keeps[i];
  }

  // Per-chunk workspace (sized for a full 100-frame chunk; reused for every chunk). The buffers
  // are persistent (Impl::ensure_ws) and viewed at the current L.
  im.ensure_ws(L);
  auto view = [](gpu::Tensor &t, std::vector<int64_t> sh) {
    gpu::Tensor v = t;
    v.shape = std::move(sh);
    return v;
  };
  gpu::Tensor y1 = view(im.ws_y1, {64, 50, 480});
  gpu::Tensor y2 = view(im.ws_y2, {32, 25, 480});
  gpu::Tensor y3 = view(im.ws_y3, {16, kConvFramesMax, 480});
  gpu::Tensor packed = view(im.ws_packed, {kConvFramesMax, kConvOutK});
  gpu::Tensor cout = view(im.ws_cout, {kConvFramesMax, kDModel});
  gpu::Tensor pos = view(im.ws_pos, {kConvFramesMax, kDModel});
  memcpy(pos.host(), im.pos.data(), im.pos.size() * 4);
  gpu::Tensor hidden = view(im.ws_hidden, {L, kDModel});

  const gpu::Tensor &w1 = im.bf("conv2d1.weight"), &w2 = im.bf("conv2d2.weight"),
                    &w3 = im.bf("conv2d3.weight"), &wc = im.bf("conv_out.weight");
  const gpu::Tensor &b1 = im.f32("conv2d1.bias"), &b2 = im.f32("conv2d2.bias"),
                    &b3 = im.f32("conv2d3.bias"), &bc = im.f32("conv_out.bias");

  Runner r;
  int row = 0;
  const int w_1 = (longest - 1) / 2 + 1, w_2 = (w_1 - 1) / 2 + 1, w_3 = (w_2 - 1) / 2 + 1;
  gpu::Tensor padbuf = view(im.ws_padbuf, {kMelBins, kChunkFrames});
  const void *mel_id = (__bridge const void *)mel.buf;
  const bool cacheable = (im.conv_cache_mel == mel_id &&
                          im.conv_cache_frames <= frames_total);
  if (!cacheable) {
    im.conv_cache.clear();
    im.conv_cache_mel = mel_id;
  }
  im.conv_cache_frames = frames_total;
  for (int i = 0; i < nchunks; ++i) {
    const int start = f0 + i * kChunkFrames;
    const int len = lens[i];
    const int chunk_index = start / kChunkFrames;
    const bool chunk_final = (len == kChunkFrames && start % kChunkFrames == 0 &&
                              start + kChunkFrames <= conv_cache_limit);
    if (chunk_final) {
      auto it = im.conv_cache.find(chunk_index);
      if (it != im.conv_cache.end()) {  // finished chunk: copy the cached rows
        ++im.cache_hits;
        const gpu::Tensor &src = it->second;
        r.c.use("copy_u32");
        r.c.set(src, 0);
        gpu::Tensor dst = hidden;
        dst.offset += (size_t)row * kDModel * 4;
        r.c.set(dst, 1);
        r.c.set_value((uint32_t)(keeps[i] * kDModel), 2);
        r.c.dispatch_threads(MTLSizeMake((size_t)keeps[i] * kDModel, 1, 1), MTLSizeMake(256, 1, 1));
        r.c.barrier();
        row += keeps[i];
        continue;
      }
    }
    gpu::Tensor xv = mel;
    int xs = frames_total;
    // A partial chunk is zero-padded to the clip's longest chunk, and the conv
    // chain then computes conv_frames(longest) rows of which only keeps[i] are credited. Only
    // the first keeps[i] rows can depend on the real (non-padded) frames, so compute exactly
    // those: with a 3x3/stride-2 chain, credited row k-1 reads input frames 0..8*k-1 (conv1 row
    // 4k-1 reads 8k-2..8k, conv2 row 2k-1 reads y1 4k-2..4k, conv3 row k-1 reads y2 2k-2..2k),
    // so 8*keeps input frames are needed. The credited rows are bit-identical to the full padded
    // computation (same taps, same accumulation order, same zero padding beyond `len`).
    int long_e = longest, w1e = w_1, w2e = w_2, w3e = w_3;
    if (len < longest) {
      const int need = 8 * keeps[i];
      if (need > 0 && need < longest) {
        long_e = need;
        w1e = (long_e - 1) / 2 + 1;
        w2e = (w1e - 1) / 2 + 1;
        w3e = (w2e - 1) / 2 + 1;
      }
    }
    if (len < longest) {  // zero-pad the tail of a short chunk, as the reference does
      float *dst = padbuf.ptr<float>();
      memset(dst, 0, (size_t)kMelBins * long_e * 4);
      const float *src = mel.ptr<float>();
      for (int h = 0; h < kMelBins; ++h)
        memcpy(dst + (size_t)h * long_e, src + (size_t)h * frames_total + start,
               (size_t)len * 4);
      xv = padbuf;
      xs = long_e;
    } else {
      xv.offset += (size_t)start * 4;
    }
    ++im.cache_miss;
    conv_f32(r.c, xv, kMelBins, long_e, 1, xs, w1, b1, y1);
    Gelu(r.c, y1, (uint32_t)(64 * w_1 * 480));
    conv_f32(r.c, y1, 64, w1e, 480, w1e, w2, b2, y2);
    Gelu(r.c, y2, (uint32_t)(32 * w_2 * 480));
    conv_f32(r.c, y2, 32, w2e, 480, w2e, w3, b3, y3);
    Gelu(r.c, y3, (uint32_t)(16 * w_3 * 480));
    {
      gpu::Tensor pk = packed;
      pk.shape = {w3e, kConvOutK};
      gpu::Tensor co = cout;
      co.shape = {w3e, kDModel};
      r.c.use("enc_pack_convout");
      r.c.set(y3, 0);
      r.c.set(pk, 1);
      r.c.set_value(16, 2);
      r.c.set_value(w3e, 3);
      r.c.set_value(480, 4);
      r.c.dispatch_threads(MTLSizeMake(w3e, 16 * 480, 1), MTLSizeMake(16, 8, 1));
      r.c.barrier();
      Gemm(r.c, pk, wc, bc, co);
      // + sinusoidal positions, then copy the credited rows into `hidden`
      r.c.use("enc_convout_store");
      r.c.set(cout, 0);
      r.c.set(pos, 1);
      r.c.set(hidden, 2);
      r.c.set_value(keeps[i], 3);
      r.c.set_value((int)kDModel, 4);
      r.c.set_value(row, 5);
      r.c.dispatch_threads(MTLSizeMake(kDModel, keeps[i], 1), MTLSizeMake(64, 4, 1));
      r.c.barrier();
      row += keeps[i];
      // Remember a finished chunk's front-end result for the next window call.
      if (chunk_final) {
        gpu::Tensor cc = ctx.empty({keeps[i], kDModel}, gpu::DType::F32);
        gpu::Tensor src = hidden;
        src.offset += (size_t)(row - keeps[i]) * kDModel * 4;
        r.c.use("copy_u32");
        r.c.set(src, 0);
        r.c.set(cc, 1);
        r.c.set_value((uint32_t)(keeps[i] * kDModel), 2);
        r.c.dispatch_threads(MTLSizeMake((size_t)keeps[i] * kDModel, 1, 1), MTLSizeMake(256, 1, 1));
        r.c.barrier();
        im.conv_cache[chunk_index] = cc;
      }
    }
  }
  // The conv front end and the 24 layers go into one command buffer.

  const double conv_ms = r.gpu_ms;
  r.gpu_ms = 0;
  // ---- 24 pre-LN transformer layers over the window's tokens ----
  gpu::Tensor normed = view(im.ws_normed, {L, kDModel});
  gpu::Tensor q = view(im.ws_q, {L, kDModel});
  gpu::Tensor k = view(im.ws_k, {L, kDModel});
  gpu::Tensor v = view(im.ws_v, {L, kDModel});
  gpu::Tensor att = view(im.ws_att, {L, kDModel});
  gpu::Tensor proj = view(im.ws_proj, {L, kDModel});
  gpu::Tensor mlp = view(im.ws_mlp, {L, kMlp});
  const uint32_t n_tok = (uint32_t)(L * kDModel);

  for (int layer = 0; layer < kLayers; ++layer) {
    const std::string p = "layers." + std::to_string(layer) + ".";
    LayerNorm(r.c, hidden, im.bf(p + "self_attn_layer_norm.weight"),
              im.bf(p + "self_attn_layer_norm.bias"), normed);
    Gemm(r.c, normed, im.bf(p + "self_attn.q_proj.weight"), im.f32(p + "self_attn.q_proj.bias"), q);
    Gemm(r.c, normed, im.bf(p + "self_attn.k_proj.weight"), im.f32(p + "self_attn.k_proj.bias"), k);
    Gemm(r.c, normed, im.bf(p + "self_attn.v_proj.weight"), im.f32(p + "self_attn.v_proj.bias"), v);
    Attention(r.c, q, k, v, att, L);
    Gemm(r.c, att, im.bf(p + "self_attn.out_proj.weight"), im.f32(p + "self_attn.out_proj.bias"),
         proj);
    Add(r.c, hidden, proj, n_tok);
    LayerNorm(r.c, hidden, im.bf(p + "final_layer_norm.weight"), im.bf(p + "final_layer_norm.bias"),
              normed);
    Gemm(r.c, normed, im.bf(p + "fc1.weight"), im.f32(p + "fc1.bias"), mlp);
    Gelu(r.c, mlp, (uint32_t)(L * kMlp));
    Gemm(r.c, mlp, im.bf(p + "fc2.weight"), im.f32(p + "fc2.bias"), proj);
    Add(r.c, hidden, proj, n_tok);
  }

  const double layer_ms = r.gpu_ms;
  r.gpu_ms = 0;
  gpu::Tensor out = ctx.empty({L, kOutputDim}, gpu::DType::F32);
  LayerNorm(r.c, hidden, im.bf("ln_post.weight"), im.bf("ln_post.bias"), normed);
  Gemm(r.c, normed, im.bf("proj1.weight"), im.f32("proj1.bias"), proj);
  Gelu(r.c, proj, (uint32_t)(L * kDModel));
  Gemm(r.c, proj, im.bf("proj2.weight"), im.f32("proj2.bias"), out);
  r.flush();

  last_gpu_ms = conv_ms + layer_ms + r.gpu_ms;
  last_ms = gpu::now_ms() - t0;
  return out;
}

gpu::Tensor Encoder::encode(const gpu::Tensor &mel, int frames) {
  const double t0 = gpu::now_ms();
  auto &ctx = gpu::Context::get();
  const int per_window = kChunksPerWindow * kChunkFrames;
  const int clip_longest = std::min(kChunkFrames, frames);
  std::vector<gpu::Tensor> outs;
  int L = 0;
  double gpu_ms = 0;
  for (int f0 = 0; f0 < frames; f0 += per_window) {
    const int f1 = std::min(frames, f0 + per_window);
    outs.push_back(encode_window(mel, frames, f0, f1, clip_longest));
    L += (int)outs.back().dim(0);
    gpu_ms += last_gpu_ms;
  }
  gpu::Tensor out = ctx.empty({L, kOutputDim}, gpu::DType::F32);
  float *dst = out.ptr<float>();
  for (const gpu::Tensor &t : outs) {
    memcpy(dst, t.host(), t.nbytes());
    dst += t.numel();
  }
  last_gpu_ms = gpu_ms;
  last_ms = gpu::now_ms() - t0;
  return out;
}

}  // namespace enc
