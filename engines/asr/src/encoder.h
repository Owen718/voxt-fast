// Audio encoder for Qwen3-ASR (owner: encoder worker).
//
// Reference: qwen3_asr/qwen3_asr_model.cpp EncodeAudio / AudioEncoderLayer.
//   * mel is sliced into 100-frame chunks (n_window * 2), each zero-padded to the longest
//     chunk, 3x conv2d(stride 2, pad 1, 3x3) + gelu, conv_out Linear, + sinusoidal
//     positions, then each chunk keeps SwiftTokenCount(chunk_frames) rows.
//   * attention runs inside windows of 8 chunks (n_window_infer 800 / 100), equal-length
//     windows batched together in the reference (which is row-independent).
//   * 24 pre-LN transformer layers, d_model 1024, 16 heads of 64, biases everywhere,
//     gelu (erf form), then ln_post, proj1 + gelu, proj2 -> 2048.
// The checkpoint's encoder weights are bf16 and the mel is float32, so MLX promotes the
// whole encoder to float32; we run float32 and convert bf16 weights exactly on load.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "gpu.h"
#include "weights.h"

namespace enc {

// Model sizes: Qwen3-ASR 1.7B values; Encoder's constructor sets them from the checkpoint's
// audio_config (0.6B: d_model 896, 14 heads, 18 layers, ffn 3584, output 1024).
inline int kDModel = 1024;
inline int kHeads = 16;
constexpr int kHeadDim = 64;
inline int kLayers = 24;
inline int kMlp = 4096;
constexpr int kChunkFrames = 100;           // n_window * 2
constexpr int kChunksPerWindow = 8;         // n_window_infer / kChunkFrames
constexpr int kMelBins = 128;
inline int kOutputDim = 2048;
constexpr float kLayerNormEps = 1e-5f;

int conv_frames(int frames);          // three stride-2 convs: (f - 1) / 2 + 1, three times
int swift_token_count(int frames);    // tokens the Swift layout credits a chunk of `frames`

struct Encoder {
  explicit Encoder(const Weights &w);
  ~Encoder();

  // Encode mel frames [f0, f1) of `mel` ([128 x frames_total] float32, mel-bin major) as
  // one attention window of at most kChunksPerWindow chunks. Returns [tokens, 2048] f32.
  // `longest_chunk` is the chunk padding length (the reference pads every chunk of the
  // whole clip to the longest chunk of the clip); 0 = longest chunk of this call.
  gpu::Tensor encode_window(const gpu::Tensor &mel, int frames_total, int f0, int f1,
                            int longest_chunk = 0);

  // Whole clip: windows of kChunksPerWindow chunks, concatenated. [tokens, 2048] f32.
  gpu::Tensor encode(const gpu::Tensor &mel, int frames);

  // Chunks that end at or before this frame offset may be served from (and stored into) the
  // per-chunk conv front-end cache. The streaming caller sets it to `frames - 16` because the
  // last mel frames are recomputed with a reflect pad on every update, so a chunk that overlaps
  // them is not reproducible yet. Default: no limit (a whole-clip encode is always final).
  int conv_cache_limit = 1 << 30;

  // Wall-clock (host + GPU) time of the last encode_window/encode call, milliseconds.
  double last_ms = 0;

  // GPU time of the last window (sum of command-buffer GPU times), milliseconds.
  double last_gpu_ms = 0;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// 3x3 stride-2 conv (padding 1) over a [H][W][C] float32 map (row stride xs) with weights
// [N][9C].
void conv_f32(gpu::Cmd &c, const gpu::Tensor &X, int H, int W, int C, int xs,
              const gpu::Tensor &Wt, const gpu::Tensor &bias, gpu::Tensor &Y);

}  // namespace enc
