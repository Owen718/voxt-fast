// Streaming ASR session: incremental log-mel, encoder-window cache, decoder KV reuse and a
// margin-gated display that rarely retracts words.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace s2 {

// Preview (live text) policy. The values are the tuned defaults; the final transcript does not
// depend on them except where noted.
struct Options {
  // Trailing complete 1 s chunks re-encoded on every preview (8 = the whole encoder window, exact).
  // With 2, older audio rows keep their KV from the previous update, so the per-update cost stays
  // flat inside an 8 s window instead of growing with the audio.
  int mutable_chunks = 2;
  // A word is shown once its smallest token margin (top-1 minus top-2 logit) reaches the gate, or
  // once the recent hypotheses agree on it (confirm rules below).
  float margin_gate = 2.0f;       // gate outside the fast cadence
  float fast_margin_gate = 2.0f;  // gate while the 100 ms cadence is active
  // Stricter gate for the first part of a session, whose words carry the least context.
  float early_gate = 6.0f;
  int early_gate_ms = 500;   // segment audio covered by the early gate (0 = use early_updates)
  int early_updates = 10;    // updates covered by the early gate when early_gate_ms is 0
  int max_new_preview = 32;  // tokens decoded per preview update at most
  // Draft hysteresis: the previous hypothesis token is kept while its logit is within this of
  // the top-1 logit, so a near tie does not flip the hypothesis on every update.
  float hyst_delta = 2.0f;   // previews
  float final_delta = 0.5f;  // final (0 = exact greedy)
  // A preview keeps the draft's stop token only when it is still the top-1 token (no
  // hysteresis): new audio must be able to extend a hypothesis that ended early. Used for
  // Qwen3-ASR 0.6B, whose guess from a segment's first half second can be a whole made-up
  // sentence that the hysteresis then keeps re-accepting, stop token included.
  bool exact_stop = false;
  // Confirm: a low-margin word is shown when every hypothesis of the last `confirm_ms` of audio
  // agrees on it (time-based, independent of the cadence); without hypotheses in that window,
  // when the last `confirm` updates agree.
  int confirm = 1;
  int confirm_ms = 200;
  int fast_confirm_ms = 300;  // confirm window while the 100 ms cadence is active
  int resync = 3;     // consecutive disagreements inside the shown text before it is rewritten
  int tail_keep = 1;  // a disagreement within the last shown word drops it at once
  bool rewrite = true;  // on a disagreement keep the word count and rewrite in place (else truncate)
  // Traditional script: when the empty-hint preview transcribes a Mandarin clip in Traditional
  // characters, switch the preview hint to English and re-decode (script_fix), and hold the
  // display meanwhile during the first trad_hold_early_ms of the segment (trad_hold).
  bool script_fix = true;
  bool trad_hold = true;
  int trad_hold_early_ms = 3000;
  // Adopt the model's own "English" detection as the preview hint, so an English final needs no
  // prompt switch (the preview's hypothesis stays a valid draft).
  bool adopt_lang = true;
  // Keep one mel buffer across updates so the encoder's per-chunk conv cache hits for the chunks
  // that are already final (exact).
  bool mel_cache = true;
  // Update cadence: period = max(min_period_ms, gpu_ms / target_share, ema(wall) * period_ema),
  // at most 400 ms. During the first fast_frames of audio a cheap update may run every
  // fast_period_ms (100 ms = one audio chunk), which halves the mean wait for new text.
  int min_period_ms = 200;
  int first_fast_updates = 0;  // the first N updates may use the fast period as their floor
  float target_share = 0.55f;
  float period_ema = 1.55f;
  int fast_period_ms = 100;
  int fast_frames = 800;  // 8 s
  bool final_exact = false;  // final: re-encode the whole tail window before decoding
  // Retained prefix: from the second decode of a segment on, teacher-force the previous
  // hypothesis minus its last `retained` tokens and decode only the tail, so older words are not
  // revised. retain_stable limits this to the prefix that already agreed with the hypothesis at
  // least retain_stable_ms (and two updates) earlier. retained_final applies it to the final.
  int retained = 5;
  bool retained_final = true;
  bool retain_stable = true;
  int retain_stable_ms = 400;
  // Pause commit for long utterances: once a segment is seg_min_frames long and the last
  // seg_pause_lookback frames contain a pause (quietest 200 ms <= seg_pause_rel x the mean 20 ms
  // energy of the last 2 s), freeze the hypothesis and start a new segment there. Speech without
  // a pause is cut at 16 s instead.
  bool seg_pause_commit = true;
  int seg_min_frames = 1200;
  int seg_pause_lookback = 60;
  float seg_pause_rel = 0.05f;
  // Early partial emit: re-evaluate the display after every decoded token and emit as soon as the
  // text extends, instead of after the whole update. Inside the fast window the last word may be
  // shown before a separator follows it (hybrid); outside it only complete words are shown early.
  bool early_emit = true;
  bool early_emit_hybrid = true;
};

class Stream {
 public:
  explicit Stream(const std::string &model_dir);
  ~Stream();
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;

  void set_options(const Options &o);
  // Called from inside update() as soon as the display text extends (early partial emit), before
  // the update's remaining decode steps. The text is a prefix of what update() will display.
  void set_on_partial(std::function<void(const std::string &)> f);
  const Options &options() const { return opts_; }
  // The caller feeds audio much faster than real time (a whole recording at once). Preview-only
  // machinery that freezes a hypothesis instead of re-decoding it is then off: the preview stops
  // at max_new_preview tokens, so a frozen segment would lose the rest. Sticky until reset().
  void set_unpaced(bool v);
  // Language hint of the final for the next utterance (Voxt sends its own per request): a name or
  // code, "" lets the model detect it, nullptr restores the default (English). Call after reset().
  void set_final_language(const std::string *language);

  void reset();
  // Append audio (mono float32, 16 kHz). Thread-safe against update().
  void add(const float *samples, size_t n);
  size_t num_samples() const;

  // One update over all audio received so far. Returns true when the display text changed.
  bool update(bool is_final);

  const std::string &display() const;
  std::string final_text();
  int generated_token_count();
  std::string finish_reason();
  std::string language();

  double last_gpu_ms() const;   // GPU ms of the last update
  int period_ms() const;        // adaptive period the next update should wait for
  bool finished() const;        // hypothesis ended on a stop token

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Options opts_;
};

}  // namespace s2
