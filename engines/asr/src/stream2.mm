// Streaming session: incremental log-mel, encoder-window cache, decoder KV reuse and a
// margin-gated monotone display.
#include "stream2.h"

#include <algorithm>
#include <climits>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <stdexcept>

#include "decoder.h"
#include "encoder.h"
#include "frontend.h"
#include "gpu.h"
#include "json.h"
#include "prompt.h"
#include "text.h"
#include "tokenizer.h"
#include "weights.h"

namespace s2 {
namespace {

constexpr int kMaxNew = 1024;         // engine.mm's max_new_tokens
constexpr int kSegmentFrames = 1600;  // 16 s: cut a long utterance here
constexpr int kChunkFrames = 100;  // 1 s
constexpr int kChunkTokens = 13;
constexpr int kWindowFrames = 800;  // 8 chunks per encoder attention window
constexpr int kMelBins = 128;
// Final fast path: a preview at most this many mel frames (10 ms each) behind the end can serve
// the final directly (the last <= 400 ms of audio is not re-encoded; the LM continues the
// hypothesis). Only for long utterances, where the re-encode + re-prefill is the stop cost.
constexpr int kFastFinalFrames = 40;      // 400 ms
constexpr int kFastFinalMinFrames = 3000;  // 30 s of audio: the long group

bool is_cjk(uint32_t cp) {
  return (cp >= 0x3400 && cp <= 0x9FFF) || (cp >= 0xF900 && cp <= 0xFAFF);
}
// Decoder-loop detector for the retained final: a greedy preview can repeat a phrase (0068: "of
// Recognition will be one of the marks"; 0076: "Instead of pulling this way and that way, and")
// and the teacher-forced final then keeps the loop, while the exact final would not. A 6-token
// phrase repeated within 18 tokens is a loop, not speech. When one is seen in a preview the
// final/commit falls back to the exact greedy decode (retained_final off for that segment).
bool local_repeat(const std::vector<int> &v, int from) {
  const int n = (int)v.size();
  const int k = 6;
  for (int i = from; i + k <= n; ++i) {
    for (int j = i + k; j + k <= n && j <= i + 18; ++j) {
      bool same = true;
      for (int t = 0; t < k; ++t) {
        if (v[i + t] != v[j + t]) { same = false; break; }
      }
      if (same) return true;
    }
  }
  return false;
}
// Traditional-only CJK codepoints (common ones; none of them is used in Simplified Chinese).
// Used to detect that the empty-hint preview is transcribing a Mandarin clip in Traditional
// script (電視報道 instead of 电视报道), which the final would then rewrite completely.
const uint32_t kTradOnly[] = {
    0x96FB, 0x8996, 0x5831, 0x5C0E, 0x986F, 0x7159, 0x5EE0, 0x88E1, 0x7576, 0x8ECA, 0x8F1B, 0x96E2,
    0x5EAB, 0x73FE, 0x5834, 0x9019, 0x500B, 0x5011, 0x8AAA, 0x70BA, 0x5C0D, 0x6703, 0x4F86, 0x6642,
    0x9593, 0x958B, 0x95DC, 0x570B, 0x5B78, 0x767C, 0x7D93, 0x904E, 0x9084, 0x9032, 0x9F8D, 0x84BC,
    0x9802, 0x7D1A, 0x9EBC, 0x6A23, 0x9EDE, 0x8B93, 0x8A8D, 0x8B58, 0x8A9E, 0x8A71, 0x8B80, 0x5BEB,
    0x807D, 0x898B, 0x89BA, 0x6771, 0x8CB7, 0x8CE3, 0x9322, 0x9280, 0x9435, 0x9580, 0x554F, 0x805E,
    0x969B, 0x968A, 0x968E, 0x9678, 0x96AA, 0x96A8, 0x96D6, 0x96D9, 0x96E3, 0x96F2, 0x9748, 0x975C,
    0x97FF, 0x9801, 0x9806, 0x9808, 0x9810, 0x9818, 0x982D, 0x984C, 0x984F, 0x9858, 0x985E, 0x98A8,
    0x98DB, 0x98EF, 0x98F2, 0x9928, 0x99AC, 0x99D5, 0x9A0E, 0x9A57, 0x9AD4, 0x9AEE, 0x9B5A, 0x9CE5,
    0x9E97, 0x9EC3, 0x9EE8, 0x9F4A, 0x842C, 0x8207, 0x5C08, 0x696D, 0x54E1, 0x5718, 0x5716, 0x5712,
    0x5713, 0x58DE, 0x58D3, 0x8072, 0x8655, 0x5099, 0x50F9, 0x5104, 0x5100, 0x5091, 0x50B3, 0x50B7,
    0x50C5, 0x5F9E, 0x512A, 0x5098, 0x5049, 0x5074, 0x5075, 0x50D1, 0x5169, 0x56B4, 0x55AA, 0x502B,
    0x507D, 0x50BE, 0x511F, 0x5132, 0x5152, 0x5167, 0x518A, 0x8ECD, 0x8FB2, 0x6C96, 0x6C7A, 0x6CC1,
    0x51CD, 0x6DE8, 0x6E96, 0x6DBC, 0x6E1B, 0x6E4A, 0x6E6F, 0x6E9D, 0x6EC5, 0x6F32, 0x6F38, 0x6F54,
    0x7051, 0x7063, 0x6FD5, 0x71C8, 0x71D2, 0x71DF, 0x723A, 0x7246, 0x734E, 0x7368, 0x7372, 0x737B,
    0x74B0, 0x7522, 0x756B, 0x7570, 0x7642, 0x76E1, 0x76E3, 0x76E4, 0x773E, 0x7926, 0x78BC, 0x78BA,
    0x7919, 0x79AE, 0x798D, 0x79AA, 0x7A2E, 0x7A4D, 0x7A31, 0x7A69, 0x7AAE, 0x7ACA, 0x7AF6, 0x7B46,
    0x7BC0, 0x7BC4, 0x7BC9, 0x7C21, 0x7C3D, 0x7C43, 0x7C4C, 0x7CE7, 0x7D00, 0x7D04, 0x7D05, 0x7D14,
    0x7D19, 0x7D1B, 0x7DCA, 0x7DDA, 0x7DF4, 0x7D44, 0x7D30, 0x7D42, 0x7D50, 0x7D55, 0x7D71, 0x7D72,
    0x7DA0, 0x7DAD, 0x7DB2, 0x7DB1, 0x7F70, 0x7F75, 0x7FA9, 0x7FD2, 0x806F, 0x8070, 0x8077, 0x8085,
    0x8105, 0x8108, 0x8173, 0x812B, 0x81C9, 0x81E8, 0x8209, 0x820A, 0x8259, 0x85DD, 0x865F, 0x87F2,
    0x8853, 0x885B, 0x88DD, 0x88DC, 0x88FD, 0x8907, 0x89C0, 0x898F, 0x89AA, 0x89F8, 0x8A08, 0x8A02,
    0x8A0E, 0x8A13, 0x8B70, 0x8A0A, 0x8A18, 0x8B1B, 0x8A31, 0x8AD6, 0x8A2D, 0x8A2A, 0x8B49, 0x8A55,
    0x8A5E, 0x8A66, 0x8A69, 0x8A72, 0x8A73, 0x8A8C, 0x8AA0, 0x8AA4, 0x8AB2, 0x8ABF, 0x8AC7, 0x8ACB,
    0x8AF8, 0x8AFE, 0x8B00, 0x8B02, 0x8B1D, 0x8B20, 0x8B6F, 0x8B77, 0x8B8A, 0x8B9A, 0x8C50, 0x8C9D,
    0x8CA0, 0x8CA1, 0x8CA2, 0x8CA7, 0x8CA8, 0x8CA9, 0x8CAA, 0x8CAC, 0x8CB4, 0x8CB8, 0x8CBB, 0x8CBC,
    0x8CBF, 0x8CC0, 0x8CC7, 0x8CD3, 0x8CDC, 0x8CDE, 0x8CE0, 0x8CEA, 0x8CEC, 0x8CED, 0x8CFC, 0x8CFD,
    0x8D0A, 0x8D08, 0x8D95, 0x8D99, 0x8E8D, 0x8ECC, 0x8ED2, 0x8F49, 0x8F2A, 0x8EDF, 0x8F15, 0x8F09,
    0x8F03, 0x8F14, 0x8F38, 0x8F44, 0x8F5F, 0x8FA6, 0x8FAD, 0x908A, 0x907C, 0x9054, 0x9077, 0x9078,
    0x907A, 0x90F5, 0x9109, 0x912D, 0x91AB, 0x91AC, 0x91CB, 0x91DD, 0x91D8, 0x91E3, 0x9234, 0x9285,
    0x9298, 0x92FC, 0x9304, 0x932B, 0x932F, 0x9375, 0x9418, 0x9451, 0x9444, 0x9577, 0x9583, 0x9589,
    0x95D6, 0x9592, 0x95E1, 0x967D, 0x9670, 0x9673, 0x96B1, 0x96DC, 0x96DE, 0x9F9C};
bool is_traditional(uint32_t cp) {
  for (uint32_t t : kTradOnly)
    if (t == cp) return true;
  return false;
}
// True when the text is written in Traditional script: at least two Traditional-only CJK
// characters and at least half of its CJK characters Traditional. A single incidental Traditional
// character (a code-switch clip, a homophone) must not trigger the script fix.
bool has_traditional(const std::string &text) {
  const size_t n = text.size();
  int cjk = 0, trad = 0;
  for (size_t i = 0; i + 2 < n; ++i) {
    const unsigned char c = (unsigned char)text[i];
    if (c < 0xE0) continue;
    const uint32_t cp = ((c & 0x0Fu) << 12) | (((unsigned char)text[i + 1] & 0x3Fu) << 6) |
                        ((unsigned char)text[i + 2] & 0x3Fu);
    if (!is_cjk(cp)) continue;
    ++cjk;
    if (is_traditional(cp)) ++trad;
  }
  // Traditional-dominant only: a Chinese clip that emits one or two Traditional forms, or a
  // code-switch clip with Latin words, must not trigger it -- switching those to the English hint
  // makes the model translate and stalls the display for seconds.
  bool latin = false;
  for (unsigned char c : text)
    latin = latin || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  return !latin && trad >= 3 && trad * 10 >= cjk * 7;
}
bool is_word_char(unsigned char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '\'';
}

struct Word {
  int begin = 0, end = 0;
  float margin = 0;
  std::string text;
};

// Display words: runs of [a-z0-9'] (lowercased) and single CJK code points; every other
// character is a separator.
void segment_words(const std::string &text, std::vector<Word> *out) {
  out->clear();
  const size_t n = text.size();
  size_t i = 0;
  while (i < n) {
    const unsigned char c = (unsigned char)text[i];
    if (c >= 0xE0 && i + 2 < n) {
      const uint32_t cp = ((c & 0x0Fu) << 12) | (((unsigned char)text[i + 1] & 0x3Fu) << 6) |
                          ((unsigned char)text[i + 2] & 0x3Fu);
      if (is_cjk(cp)) {
        Word w;
        w.begin = (int)i;
        w.end = (int)i + 3;
        w.text = text.substr(i, 3);
        out->push_back(w);
        i += 3;
        continue;
      }
    }
    if (is_word_char(c)) {
      size_t j = i;
      std::string t;
      while (j < n && is_word_char((unsigned char)text[j])) {
        t.push_back((char)std::tolower((unsigned char)text[j]));
        ++j;
      }
      Word w;
      w.begin = (int)i;
      w.end = (int)j;
      w.text = t;
      out->push_back(w);
      i = j;
      continue;
    }
    ++i;
  }
}

}  // namespace

struct Stream::Impl {
  Weights w;
  Tokenizer tok;
  enc::Encoder enc;
  dec::Decoder dec;
  host::PromptOptions popts;
  host::StopRules rules;
  Options o;

  // Audio is written by the reader thread and read by the update thread; `published` is the
  // only synchronisation needed because the buffer never reallocates.
  std::vector<float> audio;
  std::atomic<size_t> published{0};

  std::vector<float> mel_store;  // [kMelBins, mel_cap], bin-major
  int mel_cap = 0, mel_frames = 0;
  float clamp = -INFINITY;
  int clamp_changes = 0;

  int frozen_rows = 0;  // audio tokens whose KV rows are already in the cache
  int kv_f0 = 0;        // frame offset of the frozen KV boundary (rows < it are valid)
  int prev_audio_rows = 0;  // audio rows the KV cache covered after the previous update
  int frozen_windows = 0;  // complete 8-chunk windows whose rows are frozen and exact
  bool enc_dirty = true;

  std::vector<dec::KVCache> caches;
  gpu::Tensor emb;
  int emb_cap = 0;
  int aps = -1;  // audio_pad_start in the prompt
  std::vector<int> prefix_ids, suffix_ids;
  bool prompt_ready = false;
  bool prefix_kv_ok = false;  // the KV cache holds the prompt prefix rows of a previous session

  std::string committed_text;
  std::vector<std::string> committed_words;
  int committed_tokens = 0;
  int seg_f0 = 0;
  int segments = 0;
  int audio_pad_id = -1;
  std::vector<int> pad_ids;
  std::vector<int> ids;
  std::vector<float> margins;
  std::string hyp_text;
  std::string finish = "length";
  bool stopped = false;

  std::vector<std::string> shown, prev_words, prev2_words;
  // Hypothesis history for the time-based confirm: (segment audio frames, word texts) per update.
  std::deque<std::pair<int, std::vector<std::string>>> hist;
  std::vector<std::string> cur_words;  // word texts of the last update (pushed to hist)
  int cur_rel = 0;                     // segment audio frames of the last update
  std::string shown_text;
  int disagree = 0;
  // Early partial emit: callback + the text already emitted inside this update, and the peek flag
  // that restricts the display policy to provably complete words.
  std::function<void(const std::string &)> on_partial;
  std::string early_text;
  bool display_strict = false;
  std::vector<int> ids_prev, ids_prev2;  // hypotheses of the last two updates (stable-prefix rule)
  std::deque<std::pair<int, std::vector<int>>> ids_hist;  // (segment audio frames, hypothesis)
  int tstart = 0;      // first transcript token in ids (after the model's "language X<asr_text>")
  int asr_text_id = -1;
  std::string preview_lang, preview_lang0, final_lang = "English", final_lang0, detected_lang;
  bool committing = false;  // a pause16 segment commit, not the utterance final
  bool hyp_truncated = false;  // the last preview hit max_new_preview/hyp_cap (hypothesis incomplete)
  bool unpaced = false;        // the caller feeds audio faster than real time: no freeze

  double last_gpu = 0;
  int last_frames = -1;
  std::string last_lang;
  double ema_wall = 0;
  int period = 150;
  int updates = 0;
  int trad_streak = 0;  // consecutive updates whose hypothesis has Traditional-only characters
  bool lang_adopted = false;  // the preview hint was switched to the model's detected English
  bool loop_seen = false;  // a preview hypothesis repeated a 6-token phrase (decoder loop)
  int max_hyp = 1024;

  explicit Impl(const std::string &dir) : w(dir), tok(dir), enc(w), dec(w) {
    popts.language = "";  // previews: let the model detect the language (no translation)
    preview_lang = popts.language;
    preview_lang0 = preview_lang;  // the per-session reset restores this (script_fix/adopt_lang change preview_lang)
    final_lang = "English";        // default final hint when the caller sends none
    final_lang0 = final_lang;
    popts.layout = host::AudioLayout::kVoxtSwift;
    rules.im_end_id = tok.AddedTokenId("<|im_end|>");
    rules.end_of_text_id = tok.AddedTokenId("<|endoftext|>");
    rules.stop_at_end_of_text = true;
    rules.stop_on_token_loop = true;
    audio_pad_id = tok.AddedTokenId("<|audio_pad|>");
    asr_text_id = tok.AddedTokenId("<asr_text>");
    audio.assign((size_t)16000 * 300, 0.0f);  // 5 min of 16 kHz mono
    mel_cap = 1 + 300 * 100;                  // mel frames for 5 min
    mel_store.assign((size_t)kMelBins * mel_cap, 0.0f);
  }

  // ---- mel ----
  // The reference's dynamic-range clamp is the max of the whole recording, which streaming
  // cannot know. It is frozen from the first ~0.3 s and used for every frame (the clamp only
  // moves the floor of near-silent frames; keeping it fixed keeps every cached window exact
  // w.r.t. the mel it was computed from).
  int ensure_mel() {
    const size_t n = published.load(std::memory_order_acquire);
    if (n < 4800) return 0;  // 0.3 s: enough to fix the clamp
    int frames = fe::frame_count(n);
    if (frames > mel_cap) frames = mel_cap;
    int f0 = mel_frames;
    if (f0 > 0) f0 = std::max(0, f0 - 16);  // the last frames depend on the reflect padding
    fe::MelResult r = fe::log_mel(audio.data(), n, f0, frames, clamp);
    if (!std::isfinite(clamp)) {
      clamp = r.raw_max;
      ++clamp_changes;
      r = fe::log_mel(audio.data(), n, 0, frames, clamp);
      f0 = 0;
      ++mel_gen;  // the whole mel changed: the encoder's conv cache must not be reused
    }
    const float *src = r.mel.ptr<float>();
    for (int b = 0; b < kMelBins; ++b)
      memcpy(&mel_store[(size_t)b * mel_cap + f0], src + (size_t)b * r.frames,
             (size_t)r.frames * sizeof(float));
    mel_frames = frames;
    return frames;
  }

  // The encoder caches per-chunk conv outputs keyed by the mel buffer identity; that cache
  // cannot know about a clamp change, so a new session (or a clamp change) switches to the other
  // buffer and the cache is dropped. Between those events the buffer is kept, so the chunks that
  // are already final hit the cache (the encoder's limit excludes the reflect-pad frames).
  gpu::Tensor mel_buf[2];
  int mel_buf_cur = 0;
  int mel_gen = 0;       // bumped when the whole mel content changes (reset / clamp change)
  int mel_gen_done = -1;  // the generation the current buffer holds

  gpu::Tensor mel_tensor(int frames) {
    if (!mel_buf[0].buf) {
      for (int i = 0; i < 2; ++i)
        mel_buf[i] = gpu::Context::get().empty({kMelBins, mel_cap}, gpu::DType::F32);
    }
    if (mel_gen_done != mel_gen) {
      mel_buf_cur ^= 1;
      mel_gen_done = mel_gen;
    }
    gpu::Tensor &t = mel_buf[mel_buf_cur];
    float *dst = t.ptr<float>();
    for (int b = 0; b < kMelBins; ++b)
      memcpy(dst + (size_t)b * mel_cap, &mel_store[(size_t)b * mel_cap], (size_t)frames * sizeof(float));
    return t;
  }

  static int enc_rows_for(int frames) {
    if (frames <= 0) return 0;
    const int conv_rows = enc::conv_frames(std::min(kChunkFrames, frames));
    int total = 0;
    for (int f0 = 0; f0 < frames; f0 += kWindowFrames) {
      const int f1 = std::min(frames, f0 + kWindowFrames);
      for (int s = f0; s < f1; s += kChunkFrames) {
        const int len = std::min(kChunkFrames, f1 - s);
        total += std::min(enc::swift_token_count(len), conv_rows);
      }
    }
    return total;
  }

  // Encode mel frames [f0, f1) as the reference does: consecutive aligned 8-chunk windows,
  // concatenated. A range longer than one window must not become a single window.
  gpu::Tensor encode_range(const gpu::Tensor &mel, int f0, int f1, int longest) {
    std::vector<gpu::Tensor> parts;
    int rows = 0;
    for (int s = f0; s < f1; s += kWindowFrames) {
      const int e = std::min(f1, s + kWindowFrames);
      parts.push_back(enc.encode_window(mel, mel_cap, s, e, longest));
      rows += (int)parts.back().dim(0);
    }
    if (parts.size() == 1) return parts[0];
    gpu::Tensor out = gpu::Context::get().empty({rows, enc::kOutputDim}, gpu::DType::F32);
    float *dst = out.ptr<float>();
    for (const gpu::Tensor &t : parts) {
      memcpy(dst, t.host(), t.nbytes());
      dst += t.numel();
    }
    return out;
  }

  // ---- prompt ----
  void ensure_prompt(const host::Prompt &prompt, int n_audio) {
    aps = prompt.audio_pad_start;
    if (aps < 0) throw std::runtime_error("stream2: prompt has no audio pad");
    prefix_ids.assign(prompt.ids.begin(), prompt.ids.begin() + aps);
    suffix_ids.assign(prompt.ids.begin() + aps + n_audio, prompt.ids.end());
    ensure_emb(std::max(2048, aps + 64));
    dec.reserve_kv(caches, 2048);
    // The prompt prefix (everything up to <|audio_start|>) is identical for every session (the
    // language hint lives in the suffix), so its KV rows are kept across sessions and the prefix
    // forward is only run once per process. That removes a per-clip fixed cost from the first
    // update, which is where the earliest words' display lag comes from.
    if (!prefix_kv_ok) {
      gpu::Cmd c;
      dec.embed_rows_into(c, prefix_ids, emb, 0);
      c.commit_and_wait();
      dec.forward_rows(emb, aps, caches, 0, 1);
      prefix_kv_ok = true;
    }
    prompt_ready = true;
  }

  void ensure_emb(int rows) {
    if (emb_cap >= rows) return;
    emb_cap = std::max(rows, emb_cap * 2);
    emb = dec.embedding_buffer(emb_cap);
  }

  // ---- display ----
  // Char offset of every token prefix in the decoded text. The hypothesis prefix is stable
  // between updates, so only the changed tail is recomputed (Decode is O(n) per call).
  std::vector<int> ends_cache, ends_ids;
  void token_ends(const std::vector<int> &ids_in, std::vector<int> *ends) {
    size_t keep = 0;
    while (keep < ends_cache.size() && keep < ids_in.size() && ends_ids[keep] == ids_in[keep]) ++keep;
    ends_cache.resize(keep);
    ends_ids.resize(keep);
    std::vector<int> pre(ids_in.begin(), ids_in.begin() + keep);
    for (size_t i = keep; i < ids_in.size(); ++i) {
      pre.push_back(ids_in[i]);
      ends_cache.push_back((int)tok.Decode(pre, true).size());
      ends_ids.push_back(ids_in[i]);
    }
    *ends = ends_cache;
  }

  bool update_display() {
    const bool ch = update_display_impl();
    // Remember this update's hypothesis for the time-based confirm (after the checks, so a word is
    // never confirmed by the update that shows it).
    hist.push_back({cur_rel, cur_words});
    while (hist.size() > 1 && hist.front().first < cur_rel - 400) hist.pop_front();
    return ch;
  }

  // Peek at the display: run the display policy on the current hypothesis and hypothesis text
  // (`hyp`) without keeping any of its state changes. Returns the text the policy would show, or
  // "" when it would not change. The policy is deterministic in (hypothesis, margins, shown,
  // prev_words, hist, cur_rel, period), none of which the peek modifies, so the peek's decisions
  // for the words it shows are exactly the ones the update's own display call makes.
  std::string peek_display(const std::string &hyp, int ts, bool strict) {
    const std::vector<std::string> s_shown = shown, s_prev = prev_words, s_prev2 = prev2_words,
                                   s_cur = cur_words;
    const std::string s_text = shown_text, s_hyp = hyp_text;
    const int s_disagree = disagree, s_tstart = tstart;
    const bool s_strict = display_strict;
    display_strict = strict;
    hyp_text = hyp;
    tstart = ts;
    const bool ch = update_display_impl();
    std::string out = ch ? shown_text : std::string();
    shown = s_shown;
    prev_words = s_prev;
    prev2_words = s_prev2;
    cur_words = s_cur;
    shown_text = s_text;
    hyp_text = s_hyp;
    tstart = s_tstart;
    disagree = s_disagree;
    display_strict = s_strict;
    return out;
  }

  // Early partial emit: called after each decoded token of a preview update. Emits the display as
  // soon as it extends (a strict prefix relation, so a peek can never retract what is shown).
  // Returns the time of the emit, or -1.
  double try_early_emit(int frames) {
    if (!on_partial) return -1;
    // The draft rollback (ids.resize(acc)) can leave the hypothesis shorter than the transcript
    // start of the previous update; the display has no words then.
    if (tstart > (int)ids.size()) return -1;
    // The transcript start (after the model's "language X<asr_text>") is only located after the
    // decode loop, so locate it in the tokens decoded so far (the marker is at the very start) and
    // show only the transcript; while the marker is still being decoded, do not emit.
    int ts = tstart;
    if (popts.language.empty() && ts == 0) {
      if (asr_text_id < 0) return -1;
      int found = -1;
      for (int i = (int)ids.size() - 1; i >= 0; --i)
        if (ids[i] == asr_text_id) { found = i; break; }
      if (found < 0) return -1;
      ts = found + 1;
    }
    const std::string hyp = tok.Decode(std::vector<int>(ids.begin() + ts, ids.end()), true);
    // Hybrid: inside the fast window (the whole clip for short utterances) the last word may be
    // shown before a separator follows it; outside it (a long clip's later segments) only
    // provably complete words are shown, so the segment commits do not retract more.
    const bool strict = o.early_emit_hybrid && frames >= o.fast_frames;
    const std::string t = peek_display(hyp, ts, strict);
    if (t.size() <= shown_text.size() || t.compare(0, shown_text.size(), shown_text) != 0) return -1;
    if (t == early_text) return -1;
    early_text = t;
    const double now = gpu::now_ms();
    on_partial(t);
    return now;
  }

  // Confirmed: the word at index `i` also appears in every hypothesis of the last `confirm_ms` of
  // audio (time-based, cadence-independent), or in the previous `confirm` updates (count rule).
  // If no hypothesis lies inside the time window (the previous update is older than confirm_ms,
  // which happens at the 200 ms cadence whenever an update is skipped or the period is raised),
  // fall back to the count rule: the old 200 ms behaviour is one agreeing update, not a hold until
  // the word's margin rises.
  bool word_confirmed(size_t i, const std::string &w, int confirm_ms) {
    if (confirm_ms > 0) {
      const int win = std::max(1, confirm_ms / 10);  // mel frames (10 ms)
      bool any = false;
      for (auto it = hist.rbegin(); it != hist.rend(); ++it) {
        if (it->first < cur_rel - win) break;
        any = true;
        if (i >= it->second.size() || it->second[i] != w) return false;
      }
      if (any) return true;
      // No hypothesis inside the window: use the cadence-independent count rule below.
    }
    if (o.confirm <= 0) return false;
    if (i < prev_words.size() && prev_words[i] == w) {
      if (o.confirm == 1) return true;
      if (i < prev2_words.size() && prev2_words[i] == w) return true;
    }
    return false;
  }

  bool update_display_impl() {
    // Traditional-dominant hypothesis early in the segment: hold the display (script_fix
    // re-decodes it within an update or two; showing it first would be retracted).
    if (o.trad_hold && cur_rel * 10 <= o.trad_hold_early_ms && has_traditional(hyp_text)) return false;
    std::vector<Word> seg;
    segment_words(hyp_text, &seg);
    // Per-word display margins are filled first: `cur` is built from `seg` afterwards.
    if (!seg.empty()) {
      std::vector<int> dids(ids.begin() + tstart, ids.end());
      std::vector<float> dmargins(margins.begin() + tstart, margins.end());
      std::vector<int> ends;
      token_ends(dids, &ends);
      for (auto &wd : seg) {
        float m = INFINITY;
        for (size_t i = 0; i < ends.size(); ++i) {
          const int s = i ? ends[i - 1] : 0;
          const int e = ends[i];
          if (e > wd.begin && s < wd.end) m = std::min(m, dmargins[i]);
        }
        wd.margin = (m == INFINITY) ? 1e9f : (std::isfinite(m) ? m : 0.0f);
      }
    }
    std::vector<Word> cur;
    cur.reserve(committed_words.size() + seg.size());
    for (const std::string &w : committed_words) {
      Word c;
      c.margin = INFINITY;
      c.text = w;
      cur.push_back(c);
    }
    cur.insert(cur.end(), seg.begin(), seg.end());
    cur_words.clear();
    for (const auto &wd : cur) cur_words.push_back(wd.text);
    // Gate for this update: a stricter one for the first updates of the session, whose words
    // carry the least context and are the ones a later hypothesis retracts.
    const bool early = o.early_gate > 0.0f &&
                       (o.early_gate_ms > 0 ? cur_rel * 10 <= o.early_gate_ms
                                            : updates + 1 <= o.early_updates);
    // Cadence-aware display policy: inside the fast cadence the display needs a longer
    // time-based confirm (100 ms updates give half the agreement evidence per update) and can
    // afford a lower gate.
    const bool fast_now = o.fast_period_ms > 0 && period <= o.fast_period_ms;
    const float gate = early ? o.early_gate : (fast_now ? o.fast_margin_gate : o.margin_gate);
    const int cms = fast_now ? o.fast_confirm_ms : o.confirm_ms;
    size_t lcp = 0;
    while (lcp < shown.size() && lcp < cur.size() && shown[lcp] == cur[lcp].text) ++lcp;
    const size_t base = committed_words.size();
    if (lcp < shown.size()) {
      // A shown word changed. Truncating the display to the agreed prefix costs the whole tail
      // (it must be re-shown later, which shows up as lag); keeping the word count and letting
      // the text follow the newest hypothesis costs only the changed words (flicker).
      // tail_keep: a disagreement inside the last few shown words is dropped at once instead of
      // being rewritten in place (the newest words are the ones a near tie flips).
      if (o.tail_keep > 0 && shown.size() - lcp <= (size_t)o.tail_keep) {
        shown.resize(lcp);
        disagree = 0;
      } else if (++disagree >= std::max(1, o.resync)) {
        if (o.rewrite) {
          if (shown.size() > cur.size()) shown.resize(cur.size());
        } else {
          shown.resize(lcp);
        }
      }
    } else {
      disagree = 0;
    }
    if (o.rewrite || lcp == shown.size()) {
      for (size_t i = shown.size(); i < cur.size(); ++i) {
        bool show = cur[i].margin >= gate;
        if (!show && word_confirmed(i, cur[i].text, cms)) show = true;
        if (!show) break;
        // Peek (early emit): the last word of the hypothesis may still be extended by the next
        // token, so it is not shown early unless a separator already follows it.
        if (display_strict && i >= base && i + 1 == cur.size() &&
            (size_t)seg[i - base].end >= hyp_text.size())
          break;
        shown.push_back(cur[i].text);
      }
    }
    prev2_words = prev_words;
    prev_words.clear();
    for (const auto &wd : cur) prev_words.push_back(wd.text);
    if (shown.size() > cur.size()) return false;
    std::string text = committed_text;
    if (shown.size() > base) {
      const std::string seg_text = hyp_text.substr(0, seg[shown.size() - base - 1].end);
      // The new segment's hypothesis does not necessarily start with a space (after a commit the
      // model continues mid-sentence, e.g. "Sufferings."), so joining it straight onto the
      // committed text merges the last committed word with the first new one ("...danger and" +
      // "Sufferings" -> "...danger andSufferings"). The merge is invisible until the next commit
      // re-joins the two with a space, which then retracts the whole tail.
      if (!text.empty() && text.back() != ' ' && !seg_text.empty() && seg_text.front() != ' ') text += ' ';
      text += seg_text;
    }
    if (text == shown_text) return false;
    shown_text = text;
    return true;
  }

  // ---- segmentation (long utterances) ----
  // Quietest 200 ms point in [seg_f0+1200, seg_f0+1600) frames, as a frame index: a long
  // utterance is cut in a pause between 12 and 16 s.
  int find_cut(int frames) {
    const int lo = seg_f0 + 1200;
    const int hi = std::min(frames, seg_f0 + 1600);
    if (hi - lo < 40) return seg_f0 + 1400;
    const size_t n = published.load(std::memory_order_acquire);
    const int block = 320;  // 20 ms
    std::vector<double> e;
    for (int f = lo; f + 2 <= hi; f += 2) {
      const size_t s0 = (size_t)f * 160, s1 = std::min(n, s0 + block);
      double acc = 0;
      for (size_t i = s0; i < s1; ++i) acc += (double)audio[i] * audio[i];
      e.push_back(s1 > s0 ? acc / (double)(s1 - s0) : 0.0);
    }
    if (e.size() < 10) return seg_f0 + 1400;
    double best = INFINITY;
    size_t best_i = 0;
    for (size_t i = 0; i + 10 <= e.size(); ++i) {
      double sum = 0;
      for (size_t k = 0; k < 10; ++k) sum += e[i + k];
      if (sum < best) { best = sum; best_i = i; }
    }
    return lo + (int)(best_i + 5) * 2;  // middle of the quietest 200 ms
  }

  // Middle frame of the quietest 200 ms in the last `seg_pause_lookback` frames, when that window
  // is a pause; -1 otherwise. A pause is the quietest 200 ms being <= seg_pause_rel x the mean
  // 20 ms block energy of the last 2 s (measured: real pauses are < 0.01, speech is > 0.1).
  // Cutting as soon as the pause is seen keeps the display at most
  // `seg_pause_lookback` ahead of the cut, so the commit does not retract the words after it.
  int find_pause_cut(int frames) {
    const int lo = std::max(seg_f0 + 200, frames - o.seg_pause_lookback);
    const int hi = frames - 10;
    if (hi - lo < 40) return -1;
    const size_t n = published.load(std::memory_order_acquire);
    const int block = 320;  // 20 ms
    std::vector<double> e;
    for (int f = lo; f + 2 <= hi; f += 2) {
      const size_t s0 = (size_t)f * 160, s1 = std::min(n, s0 + block);
      double acc = 0;
      for (size_t i = s0; i < s1; ++i) acc += (double)audio[i] * audio[i];
      e.push_back(s1 > s0 ? acc / (double)(s1 - s0) : 0.0);
    }
    if (e.size() < 10) return -1;
    double best = INFINITY;
    size_t best_i = 0;
    for (size_t i = 0; i + 10 <= e.size(); ++i) {
      double sum = 0;
      for (size_t k = 0; k < 10; ++k) sum += e[i + k];
      if (sum < best) { best = sum; best_i = i; }
    }
    // Reference: mean 20 ms block energy of the last 2 s (speech level, so a pause reads ~0).
    double ref = 0;
    int nb = 0;
    for (int f = std::max(seg_f0, frames - 200); f + 2 <= frames; f += 2) {
      const size_t s0 = (size_t)f * 160, s1 = std::min(n, s0 + block);
      double acc = 0;
      for (size_t i = s0; i < s1; ++i) acc += (double)audio[i] * audio[i];
      ref += s1 > s0 ? acc / (double)(s1 - s0) : 0.0;
      ++nb;
    }
    ref = nb > 0 ? ref / nb : 0.0;
    if (best / 10.0 > (double)o.seg_pause_rel * ref) return -1;
    return lo + (int)(best_i + 5) * 2;  // middle of the quietest 200 ms
  }

  // Exact greedy decode of the current segment up to frame `cut`, then commit its text. Used by the
  // 16 s fallback, which cuts behind the display: the re-decoded text is a prefix of the display
  // (the words after the cut are dropped and re-decoded by the next segment). The pause commit
  // freezes instead.
  void commit_segment(int cut) {
    committing = true;
    update(true, cut);
    committing = false;
    host::SplitResult r = host::SplitOutput(tok, ids, popts.language);
    if (!r.text.empty()) {
      if (!committed_text.empty() && committed_text.back() != ' ') committed_text += ' ';
      committed_text += r.text;
    }
    committed_tokens += host::GeneratedTokenCount(ids, rules) - tstart;
    reset_segment();
  }

  // Commit the current hypothesis as it stands, without re-decoding the segment. The pause commit
  // uses this: the cut is the current audio end and the display already shows this text (the
  // display is a prefix of the hypothesis), so nothing is retracted.
  void commit_segment_freeze() {
    if (!hyp_text.empty()) {
      if (!committed_text.empty() && committed_text.back() != ' ') committed_text += ' ';
      committed_text += hyp_text;
    }
    committed_tokens += host::GeneratedTokenCount(ids, rules) - tstart;
    reset_segment();
  }

  // Reset the decoder/display state for the next segment (the committed text is kept).
  void reset_segment() {
    std::vector<Word> w;
    segment_words(committed_text, &w);
    committed_words.clear();
    for (const Word &x : w) committed_words.push_back(x.text);
    ids.clear();
    margins.clear();
    ids_prev.clear();
    ids_prev2.clear();
    tstart = 0;
    hyp_text.clear();
    finish = "length";
    stopped = false;
    shown = committed_words;
    shown_text = committed_text;
    prev_words = committed_words;
    prev2_words = committed_words;
    hist.clear();
    disagree = 0;
    prompt_ready = false;
    frozen_rows = 0;
    kv_f0 = 0;
    frozen_windows = 0;
    enc_dirty = true;
    loop_seen = false;  // the new segment's previews decide again
    ++segments;
  }

  // One update over all audio received so far. Returns true when the display text changed.
  bool update(bool is_final, int frame_limit = 0) {
    const double t_start = gpu::now_ms();
    double gpu = 0;
    int frames = ensure_mel();
    if (frames <= 0) {
      if (is_final && !committing) {
        finish = "stop";
        stopped = true;
      }
      return false;
    }
    if (frame_limit > 0 && frame_limit < frames) frames = frame_limit;
    if (!is_final && frames - seg_f0 >= kSegmentFrames) {
      const int cut = find_cut(frames);
      if (cut > seg_f0 + 200 && cut + 50 <= frames) {
        commit_segment(cut);
        seg_f0 = cut;
        gpu += last_gpu;
      }
    }
    const int rel = frames - seg_f0;
    cur_rel = rel;
    // A pause commit sets seg_f0 = frames, so the very next update - typically the utterance final
    // when the pause is at the end of the clip - has no audio of its own. The prompt would then
    // carry zero <|audio_pad|> rows and ensure_prompt() would throw. There is nothing new to
    // decode: keep the current hypothesis (after a commit that is the committed text).
    if (rel <= 0) {
      // For the real final the committed text is the transcript, so report a clean stop
      // (reset_segment left finish = "length", which would read as a truncated decode).
      if (is_final && !committing) {
        finish = "stop";
        stopped = true;
      }
      return false;
    }
    // Language hint: the previews may run without one (the model then detects the language
    // itself, which keeps it from translating a Chinese utterance into English on a short audio
    // prefix); the final uses the standard request hint so its transcript matches the reference.
    {
      const bool real_final = is_final && !committing;
      // Previews: once the model has told us the language itself, use it as the hint (the
      // reference server's dynamic language). The model then stops emitting the marker, which would
      // otherwise be frozen into the retained prefix, and the final needs no prompt switch when
      // the detected language already is the final one (cheap final, no end retraction).
      // A recent preview already told us the language: if it is the final's language, keep the
      // preview prompt for the final (no suffix switch, no re-prefill of the prefix, and the
      // final can reuse the preview's KV). Far from the preview (whole-clip input) keep the
      // final hint so the offline transcript is exact.
      const int grew = last_frames >= 0 ? frames - last_frames : INT_MAX;
      const bool near_preview = frames >= kFastFinalMinFrames && !enc_dirty && prompt_ready &&
                                !ids.empty() && grew >= 0 && grew <= kFastFinalFrames;
      std::string want = real_final ? final_lang : preview_lang;
      if (real_final && near_preview && popts.language.empty()) {
        host::SplitResult r = host::SplitOutput(tok, ids, popts.language);
        if (r.language.has_value() &&
            host::NormalizeLanguage(*r.language) == host::NormalizeLanguage(final_lang))
          want = popts.language;  // the preview already decoded in the final language
      }
      if (popts.language != want) {
        popts.language = want;
        prompt_ready = false;  // only the suffix (the language marker) changes
        if (real_final && tstart > 0) {
          // The final's prompt carries the language marker itself: drop the preview's marker
          // tokens so the previous transcript stays a valid draft (cheap final). The stable-prefix
          // reference (ids_prev/ids_prev2) must lose the same marker, not be cleared: clearing it
          // makes `stable` 0 and silently disables retained_final on every clip whose preview
          // language differs from the final hint (all zh/mixed clips: final_lang is English).
          const std::vector<int> marker(ids.begin(), ids.begin() + tstart);
          ids.erase(ids.begin(), ids.begin() + tstart);
          if ((int)margins.size() >= tstart) margins.erase(margins.begin(), margins.begin() + tstart);
          auto drop_marker = [&](std::vector<int> &v) {
            if ((int)v.size() >= tstart && std::equal(marker.begin(), marker.end(), v.begin()))
              v.erase(v.begin(), v.begin() + tstart);
            else
              v.clear();
          };
          drop_marker(ids_prev);
          drop_marker(ids_prev2);
          ids_hist.clear();  // entries still carry the marker; fall back to the marker-free ids_prev2
          tstart = 0;
        }
      }
    }
    // Fast final: "end" arrived with no new audio since the last update and an unchanged prompt,
    // so the KV cache and the hypothesis are current. Roll back to the end of the hypothesis and
    // only decode the remaining tokens (usually 0-3) instead of re-encoding and re-prefilling.
    // For a long utterance a preview that is at most kFastFinalFrames behind is just as good: the
    // new audio is at most one protocol chunk of the clip's tail and the LM continues the
    // hypothesis; this keeps the final's re-encode + re-prefill off the stop path.
    const int grew = last_frames >= 0 ? frames - last_frames : INT_MAX;
    const bool long_utt = frames >= kFastFinalMinFrames;
    const bool recent_preview = is_final && !loop_seen && !enc_dirty && prompt_ready && !ids.empty() &&
                                popts.language == last_lang &&
                                (frames == last_frames ||
                                 (long_utt && grew > 0 && grew <= kFastFinalFrames));
    if (long_utt && recent_preview && stopped) {
      // The preview already reached the stop token: the hypothesis is the complete transcript,
      // only the trailing silence (<= kFastFinalFrames) is new. Re-encoding it cannot add text
      // and would only re-verify a draft that is already final.
      const bool ch = update_display();
      ++updates;
      last_gpu = dec.last_gpu_ms;
      return ch;
    }
    if (recent_preview) {
      // decode from the last hypothesis token: the KV already holds the whole hypothesis
      float m = 0;
      finish = "length";
      stopped = false;
      int tok_next = 0;
      while ((int)ids.size() < kMaxNew) {
        tok_next = dec.step_margin(ids.back(), caches, &m);
        ids.push_back(tok_next);
        margins.push_back(m);
        if (host::IsStopToken(tok_next, rules)) { finish = "stop"; stopped = true; break; }
        if (rules.stop_on_token_loop && host::IsTokenLoop(ids)) { finish = "stop"; stopped = true; break; }
      }
      hyp_text = tok.Decode(std::vector<int>(ids.begin() + tstart, ids.end()), true);
      const bool ch = update_display();
      ++updates;
      last_gpu = dec.last_gpu_ms;
      return ch;
    }
    host::Prompt prompt = host::BuildPrompt(tok, rel, popts);
    const int n_audio = prompt.audio_token_count;
    if (!prompt_ready) {
      ensure_prompt(prompt, n_audio);
      gpu += dec.last_gpu_ms;
    }
    if ((int)prompt.ids.size() != aps + n_audio + (int)suffix_ids.size())
      throw std::runtime_error("stream2: prompt layout changed");

    // Which audio rows are already frozen in the KV cache, and which frame range must be
    // re-encoded?  frozen_rows is always a whole number of complete chunks.
    const int complete = rel / kChunkFrames;
    int n_frozen, f0;
    if (enc_dirty) {
      n_frozen = 0;
      f0 = 0;
    } else if (is_final && o.final_exact) {
      const int windows = rel / kWindowFrames;
      f0 = std::min(kWindowFrames * windows, kv_f0);  // kv_f0 = rows whose KV is already valid
      n_frozen = std::min(kChunkTokens * (f0 / kChunkFrames), n_audio);
    } else if (o.mutable_chunks >= 8 || unpaced || rel >= o.fast_frames) {  // exact: freeze only complete 8-chunk windows
      // Whole-clip input (set_unpaced) always takes the exact path, so its transcript is the
      // offline one. The shortcut is also limited to the fast-cadence region (rel < fast_frames):
      // only there does the cadence gain anything (the period is 200 ms from 8 s on anyway), and
      // the pause commit freezes the preview's text, so a stale-KV preview must not feed a
      // committed segment.
      const int windows = rel / kWindowFrames;
      // A window is only exact once it has been encoded with all 800 frames, so the update
      // that sees a window complete re-encodes it (and re-prefills it) before freezing it.
      f0 = kWindowFrames * (windows > frozen_windows ? frozen_windows : windows);
      n_frozen = std::min(kChunkTokens * (f0 / kChunkFrames), n_audio);
    } else {
      // Preview shortcut: re-encode only the last `mutable_chunks` seconds and keep the older
      // audio rows' KV from the previous update. Only rows whose KV the previous update actually
      // left in the cache can be frozen (prev_audio_rows); on the first update of a segment that
      // is 0, so the whole range is encoded exactly (whole-clip input arrives in the first update:
      // freezing rows with an empty KV would make the final garbage).
      const int fc = std::max(0, complete - o.mutable_chunks);
      const int valid = kChunkFrames * (prev_audio_rows / kChunkTokens);
      f0 = std::min(kChunkFrames * fc, valid);
      n_frozen = std::min(kChunkTokens * (f0 / kChunkFrames), n_audio);
    }
    if (f0 >= rel) f0 = std::max(0, rel - kChunkFrames);
    f0 = (f0 / kChunkFrames) * kChunkFrames;
    n_frozen = std::min((f0 / kChunkFrames) * kChunkTokens, n_audio);
    gpu::Tensor rows;
    {
      gpu::Tensor mel = mel_tensor(frames);
      enc.conv_cache_limit = o.mel_cache ? frames - 16 : (1 << 30);
      rows = encode_range(mel, seg_f0 + f0, frames, std::min(kChunkFrames, rel));
      gpu += enc.last_gpu_ms;
    }
    int nm2 = (int)rows.dim(0);
    if (nm2 <= 0 || n_frozen + nm2 > n_audio) {  // unexpected: redo the whole audio from scratch
      n_frozen = 0;
      f0 = 0;
      gpu::Tensor mel = mel_tensor(frames);
      rows = encode_range(mel, seg_f0, frames, std::min(kChunkFrames, rel));
      gpu += enc.last_gpu_ms;
      nm2 = (int)rows.dim(0);
    }
    // The prompt asks for SwiftTokenCount(frames) pads but the encoder produces fewer rows for
    // a partial chunk (min(swift_token_count(len), conv_frames)); the tail keeps the audio_pad
    // embedding, exactly like engine.mm's `fill = min(audio_token_count, enc_rows)`.
    const int n_pad = n_audio - n_frozen - nm2;
    const int nsuf = (int)suffix_ids.size();
    const int nd = (int)ids.size();
    // Retained prefix (as SGLang-Omni's realtime decoder does): force the previous hypothesis minus its last
    // `retained` tokens (teacher-forced, so the display can never retract older words) and only
    // verify/decode the tail. Trailing stop tokens are not part of the transcript; pop further
    // while the prefix text ends in U+FFFD (a truncated multi-byte character).
    int forced = 0;
    // The real final (and, with retained_final, the 16 s segment commit) teacher-forces the prefix
    // the display already shows, so an exact re-decode cannot rewrite an early word at the end and
    // retract the whole tail. After a preview loop the exact decode is used (local_repeat).
    if (o.retained > 0 && nd > 0 &&
        (!is_final || (o.retained_final && !loop_seen))) {
      int usable = nd;
      while (usable > 0 && host::IsStopToken(ids[usable - 1], rules)) --usable;
      // Only force the part of the hypothesis that was already stable in the previous update
      // (longest common prefix with the hypothesis two updates ago): a fresh, still-changing
      // guess (e.g. the model's first-language guess on a short audio prefix) is re-derived
      // instead of being frozen into the transcript.
      int stable = usable;
      if (o.retain_stable) {
        stable = 0;
        // Reference hypothesis: the update at least `retain_stable_ms` of audio older and at least
        // two updates back (the older of the two), so the evidence window does not shrink with the
        // cadence. Falls back to the old two-updates rule.
        const std::vector<int> *ref = &ids_prev2;
        if (o.retain_stable_ms > 0 && ids_hist.size() > 2) {
          const int win = o.retain_stable_ms / 10;  // frames
          // Newest entry that is at least two updates back AND at least `win` frames older.
          for (size_t k = ids_hist.size() - 1; k-- > 0;) {  // k = n-2, n-3, ...
            if (ids_hist[k].first <= cur_rel - win) { ref = &ids_hist[k].second; break; }
          }
        }
        const int m = std::min((int)ref->size(), usable);
        while (stable < m && ids[stable] == (*ref)[stable]) ++stable;
      }
      forced = std::max(0, std::min(usable - o.retained, stable));
      if (forced > tstart) {
        std::vector<int> ends;
        token_ends(std::vector<int>(ids.begin() + tstart, ids.begin() + forced), &ends);
        int e = ends[forced - tstart - 1];
        while (forced > tstart && e >= 3 &&
               hyp_text.compare((size_t)e - 3, 3, "\xEF\xBF\xBD") == 0) {
          --forced;
          e = forced > tstart ? ends[forced - tstart - 1] : 0;
        }
      }
    }
    const int n = nm2 + n_pad + nsuf + nd;
    ensure_emb(n + 8);

    // Batch = [encoder rows][audio_pad rows][suffix][draft] at positions aps+n_frozen..
    {
      gpu::Cmd c;
      dec.embed_rows_into(c, suffix_ids, emb, nm2 + n_pad);
      dec.splice_embeddings(c, emb, rows, 0, nm2);
      if (n_pad > 0) {
        pad_ids.assign((size_t)n_pad, audio_pad_id);
        dec.embed_rows_into(c, pad_ids, emb, nm2);
      }
      if (nd > 0) dec.embed_rows_into(c, ids, emb, nm2 + n_pad + nsuf);
      c.commit_and_wait();
    }
    const int k = std::min(nd - forced + 1, n);
    gpu::Tensor logits;
    std::vector<dec::Decoder::RowOut> rout =
        dec.forward_rows(emb, n, caches, aps + n_frozen, k, &logits);
    gpu += dec.last_gpu_ms;
    frozen_rows = n_frozen;
    kv_f0 = f0;
    prev_audio_rows = n_audio;
    enc_dirty = false;
    if (rel / kWindowFrames > frozen_windows) frozen_windows = rel / kWindowFrames;

    // Verify the draft (rout[i] predicts ids[i]) and continue greedily.
    // Draft hysteresis: a preview keeps its previous token while
    // it is within `hyst_delta` of the top1 logit, so a near-tie cannot flip the hypothesis every
    // update. The final (is_final) stays exact greedy.
    const float delta = is_final ? o.final_delta : o.hyst_delta;
    const int vocab = (int)logits.dim(1);
    const uint16_t *lg = delta > 0 ? logits.ptr<uint16_t>() : nullptr;
    int acc = forced;
    while (acc < nd) {
      const int r = acc - forced;
      if (rout[r].argmax == ids[acc]) {
        ++acc;
        continue;
      }
      if (!lg) break;
      if (o.exact_stop && !is_final && host::IsStopToken(ids[acc], rules)) break;
      const float l_top = gpu::bf16_to_f32(lg[(size_t)r * vocab + rout[r].argmax]);
      const float l_draft = gpu::bf16_to_f32(lg[(size_t)r * vocab + ids[acc]]);
      if (l_draft < l_top - delta) break;
      ++acc;
    }
    int next;
    float next_margin;
    if (acc < nd) {
      next = rout[acc - forced].argmax;
      next_margin = rout[acc - forced].margin;
    } else {
      next = rout[k - 1].argmax;
      next_margin = rout[k - 1].margin;
    }
    // Forced tokens are stable by construction: their display margin is infinite.
    for (int i = 0; i < forced; ++i) margins[i] = INFINITY;
    for (int i = forced; i < acc; ++i) margins[i] = rout[i - forced].margin;
    const bool draft_done = acc == nd && nd > 0 && host::IsStopToken(ids[nd - 1], rules);
    dec.rollback(caches, aps + n_frozen + nm2 + n_pad + nsuf + acc);
    ids.resize(acc);
    margins.resize(acc);

    const int hyp_cap = std::max(48, (int)(8.0 * (double)rel / 100.0));
    int new_count = 0;
    finish = draft_done ? "stop" : "length";
    stopped = draft_done;
    hyp_truncated = false;
    while (!draft_done && (int)ids.size() < kMaxNew) {
      ids.push_back(next);
      margins.push_back(next_margin);
      if (host::IsStopToken(next, rules)) {
        finish = "stop";
        stopped = true;
        break;
      }
      if (rules.stop_on_token_loop && host::IsTokenLoop(ids)) {
        finish = "stop";
        stopped = true;
        break;
      }
      ++new_count;
      if (!is_final && o.early_emit) try_early_emit(frames);
      if (!is_final && (new_count >= o.max_new_preview || (int)ids.size() >= hyp_cap)) {
        // The preview stopped at the per-update cap: the hypothesis does not cover the segment's
        // audio, so it must not be frozen as a committed segment (the pause commit would lose
        // every later word).
        hyp_truncated = true;
        break;
      }
      float m = 0;
      next = dec.step_margin(next, caches, &m);
      next_margin = m;
      gpu += dec.last_gpu_ms;
    }
    // Transcript start: with no requested language the model emits "language X<asr_text>" itself;
    // the display, the retained prefix and the token count work on the transcript tokens only.
    tstart = 0;
    if (popts.language.empty() && asr_text_id >= 0) {
      for (int i = 0; i < (int)ids.size(); ++i) {
        if (ids[i] == asr_text_id) {
          tstart = i + 1;
          break;
        }
      }
      if (tstart > 0) {
        // The model's own language detection ("language English" / "language Chinese").
        const std::string marker = tok.Decode(std::vector<int>(ids.begin(), ids.begin() + tstart), true);
        const size_t p = marker.find("language ");
        std::string lang = p == std::string::npos ? std::string() : marker.substr(p + 9);
        const size_t lt = lang.find('<');  // the <asr_text> marker is not skipped by Decode
        if (lt != std::string::npos) lang.resize(lt);
        while (!lang.empty() && (lang.back() == ' ' || lang.back() == '\n')) lang.pop_back();
        while (!lang.empty() && lang.front() == ' ') lang.erase(lang.begin());
        if (!lang.empty()) detected_lang = lang;
      }
    }
    hyp_text = tok.Decode(std::vector<int>(ids.begin() + tstart, ids.end()), true);
    if (!is_final && local_repeat(ids, tstart)) loop_seen = true;
    ids_prev2 = ids_prev;
    ids_prev = ids;
    ids_hist.push_back({cur_rel, ids});
    while (ids_hist.size() > 1 && ids_hist.front().first < cur_rel - 200) ids_hist.pop_front();
    // Script fix: the empty-hint preview can transcribe a Mandarin clip in Traditional script
    // (電視報道 instead of 电视报道), which the final then rewrites completely. When the hypothesis
    // keeps Traditional-only characters for two updates (a one-update flip such as 當天 -> 当天
    // corrects itself), switch the preview hint to English and drop the segment so the model
    // re-decodes it.
    if (o.script_fix && !is_final && !committing && preview_lang.empty() && !ids.empty() &&
        updates + 1 <= 12) {
      trad_streak = has_traditional(hyp_text) ? trad_streak + 1 : 0;
      if (trad_streak >= 2) {
        preview_lang = "English";
        popts.language = "English";
        prompt_ready = false;
        ids.clear();
        margins.clear();
        ids_prev.clear();
        ids_prev2.clear();
        tstart = 0;
        hyp_text.clear();
        finish = "length";
        stopped = false;
        shown = committed_words;
        shown_text = committed_text;
        prev_words = committed_words;
        prev2_words = committed_words;
        hist.clear();
        disagree = 0;
        trad_streak = 0;
      }
    }
    const bool changed = update_display();
    // Adopt the model's own language detection as the preview hint when it is English: then the
    // prompt does not change at an English final, the draft stays valid and the fast final can
    // fire. Only English is adopted: forcing "Chinese" on the previews changes the script of some
    // clips and the final's language.
    if (o.adopt_lang && !is_final && !committing && !lang_adopted && popts.language.empty() &&
        tstart > 0) {
      const std::string marker = tok.Decode(std::vector<int>(ids.begin(), ids.begin() + tstart), true);
      if (marker.find("English") != std::string::npos) {
        lang_adopted = true;
        preview_lang = "English";
        popts.language = "English";
        prompt_ready = false;  // the language marker moves from the model's output into the prompt
        ids.erase(ids.begin(), ids.begin() + tstart);
        margins.erase(margins.begin(), margins.begin() + tstart);
        tstart = 0;
        ids_prev.clear();
        ids_prev2.clear();
      }
    }
    // Pause-triggered segment commit (long-group flicker): the segment is long enough and the
    // audio is in a pause, so the current hypothesis is complete and the display already shows it.
    // Freeze it and start the next segment at the audio end: no re-decode, so nothing is retracted
    // (the 16 s fallback above still cuts at the quietest point of [12,16) s and re-decodes).
    // `hyp_truncated` guards whole-clip input, whose single preview update stops at
    // max_new_preview tokens: freezing that partial hypothesis would truncate the transcript.
    if (!is_final && o.seg_pause_commit && !unpaced && !hyp_truncated &&
        frames - seg_f0 >= o.seg_min_frames && find_pause_cut(frames) >= 0) {
      commit_segment_freeze();
      seg_f0 = frames;
    }
    ++updates;
    last_gpu = gpu;
    last_frames = frames;
    last_lang = popts.language;
    const double wall = gpu::now_ms() - t_start;
    // The engine must not saturate: if an update takes as long as the audio it consumes, the
    // input queue grows and every token's display lag grows with it. Keep ~45% slack, and also
    // respect the GPU-share target (the measured GPU time is ~1.2x the wall time here).
    // Skip the first two updates (Metal warm-up inflates the wall time by ~5x).
    if (updates > 2) ema_wall = ema_wall > 0 ? 0.7 * ema_wall + 0.3 * wall : wall;
    // The first two updates of a session carry the Metal warm-up and the first-call paths, so their
    // GPU time is not representative; letting it raise the period shifts every later display
    // boundary by 100 ms (the trigger quantises to the 100 ms audio chunks). Keep the floor there.
    // First-update timing: inside the fast region the first `first_fast_updates` updates may use
    // the fast period as their floor, so the first preview does not wait for min_period_ms.
    const bool first_fast = o.fast_period_ms > 0 && frames < o.fast_frames &&
                            updates < o.first_fast_updates;
    period = (int)std::min(400.0, std::max({(double)(first_fast
                                                         ? std::min(o.min_period_ms, o.fast_period_ms)
                                                         : o.min_period_ms),
                                            updates > 2 ? gpu / std::max(0.05f, o.target_share) : 0.0,
                                            ema_wall * o.period_ema}));
    // Cheap updates at the start of a segment can afford the 100 ms cadence (one audio chunk);
    // the GPU share of such an update is gpu/100 ms. The condition is the segment's AUDIO span
    // only (deterministic: the same audio gives the same cadence);
    // the GPU share is bounded by fast_frames, measured per group.
    // `frames` (the whole clip's audio), not `rel`: a multi-segment utterance gets the fast cadence
    // only in its first fast_frames of audio, so the later segments of a long utterance (and their
    // commits) keep the 200 ms cadence.
    if (o.fast_period_ms > 0 && frames < o.fast_frames && updates > 2)
      period = std::min(period, o.fast_period_ms);
    return changed;
  }
};

// ---------------------------------------------------------------------------
// public API

Stream::Stream(const std::string &model_dir) : impl_(new Impl(model_dir)) {}
Stream::~Stream() = default;

void Stream::set_options(const Options &o) {
  opts_ = o;
  impl_->o = o;
}
void Stream::set_unpaced(bool v) { impl_->unpaced = v; }
void Stream::set_final_language(const std::string *language) {
  impl_->final_lang = language ? *language : impl_->final_lang0;
}
void Stream::set_on_partial(std::function<void(const std::string &)> f) {
  impl_->on_partial = std::move(f);
}

void Stream::reset() {
  Impl &s = *impl_;
  // Keep the KV buffers and the prompt-prefix rows (identical in every session): truncate every
  // layer to the prefix length instead of freeing 235 MB and re-forwarding the prefix.
  if ((int)s.caches.size() == s.dec.config().n_layers && s.aps > 0 && s.prefix_kv_ok) {
    for (auto &c : s.caches) c.len = s.aps;
  } else {
    s.caches.clear();
    s.prefix_kv_ok = false;
  }
  s.published.store(0, std::memory_order_release);
  s.mel_frames = 0;
  s.clamp = -INFINITY;
  s.clamp_changes = 0;
  s.frozen_rows = 0;
  s.kv_f0 = 0;
  s.prev_audio_rows = 0;
  s.frozen_windows = 0;
  s.enc_dirty = true;
  s.early_text.clear();  // no early emit survives a session boundary
  ++s.mel_gen;  // a new session's mel is unrelated to the previous one's
  s.aps = -1;
  s.prefix_ids.clear();
  s.suffix_ids.clear();
  s.prompt_ready = false;
  s.ends_cache.clear();
  s.ends_ids.clear();
  s.committed_text.clear();
  s.committed_words.clear();
  s.committed_tokens = 0;
  s.seg_f0 = 0;
  s.segments = 0;
  s.unpaced = false;
  s.ids.clear();
  s.ids_prev.clear();
  s.ids_prev2.clear();
  s.ids_hist.clear();
  s.tstart = 0;
  s.margins.clear();
  s.hyp_text.clear();
  s.finish = "length";
  s.stopped = false;
  s.shown.clear();
  s.prev_words.clear();
  s.prev2_words.clear();
  s.hist.clear();
  s.cur_words.clear();
  s.cur_rel = 0;
  s.shown_text.clear();
  s.disagree = 0;
  s.last_gpu = 0;
  s.ema_wall = 0;
  s.period = (opts_.fast_period_ms > 0 && opts_.first_fast_updates > 0)
                 ? std::min(opts_.min_period_ms, opts_.fast_period_ms)
                 : opts_.min_period_ms;
  s.updates = 0;
  s.o = opts_;
  s.detected_lang.clear();
  s.loop_seen = false;
  // The preview hint is per session: script_fix/adopt_lang switch it mid-clip.
  s.preview_lang = s.preview_lang0;
  s.popts.language = s.preview_lang0;
  s.trad_streak = 0;
  s.lang_adopted = false;
}

void Stream::add(const float *samples, size_t n) {
  Impl &s = *impl_;
  const size_t n0 = s.published.load(std::memory_order_relaxed);
  const size_t take = std::min(n, s.audio.size() - n0);
  if (take == 0) return;
  memcpy(s.audio.data() + n0, samples, take * sizeof(float));
  s.published.store(n0 + take, std::memory_order_release);
}

size_t Stream::num_samples() const { return impl_->published.load(std::memory_order_acquire); }

bool Stream::update(bool is_final) { return impl_->update(is_final); }

const std::string &Stream::display() const { return impl_->shown_text; }
std::string Stream::final_text() {
  host::SplitResult r = host::SplitOutput(impl_->tok, impl_->ids, impl_->popts.language);
  if (impl_->committed_text.empty()) return r.text;
  if (r.text.empty()) return impl_->committed_text;
  return impl_->committed_text + " " + r.text;
}
int Stream::generated_token_count() {
  return impl_->committed_tokens + host::GeneratedTokenCount(impl_->ids, impl_->rules) - impl_->tstart;
}
std::string Stream::finish_reason() { return impl_->finish; }
std::string Stream::language() {
  host::SplitResult r = host::SplitOutput(impl_->tok, impl_->ids, impl_->popts.language);
  return r.language.value_or("");
}
double Stream::last_gpu_ms() const { return impl_->last_gpu; }
int Stream::period_ms() const { return impl_->period; }
bool Stream::finished() const { return impl_->stopped; }

}  // namespace s2
