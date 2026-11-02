// asr_engine: streaming Qwen3-ASR over a JSON-lines protocol on stdin/stdout.
//
// stdin:  {"op":"start","session":N[,"language":L]}   L: the final's hint; "" lets the model detect
//         {"op":"audio","session":N,"pcm16":"<base64 16 kHz mono s16le>"}
//         {"op":"end","session":N}
//         {"op":"exit"}
// stdout: {"event":"ready"}
//         {"event":"partial","session":N,"text":...}          live text while audio arrives
//         {"event":"final","session":N,"text":...,"language":...,"generated_token_count":K,
//          "finish_reason":"stop"|"length"}
//
// A reader thread queues the input so audio is always read promptly; the main thread runs the
// updates. A "start" ends the previous session.
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gpu.h"
#include "json.h"
#include "stream2.h"

namespace {

// Previews never use more than this fraction of the session's wall time on the GPU.
constexpr double kPreviewBudget = 0.52;

std::mutex out_mu;
void emit(const std::string &line) {
  std::lock_guard<std::mutex> g(out_mu);
  fwrite(line.data(), 1, line.size(), stdout);
  fputc('\n', stdout);
  fflush(stdout);
}

std::vector<float> b64_pcm16(const std::string &s) {
  static int8_t T[256];
  static bool init = false;
  if (!init) {
    memset(T, -1, sizeof T);
    const char *a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) T[(unsigned char)a[i]] = (int8_t)i;
    init = true;
  }
  std::vector<uint8_t> bytes;
  bytes.reserve(s.size() * 3 / 4);
  uint32_t acc = 0;
  int bits = 0;
  for (unsigned char c : s) {
    int8_t v = T[c];
    if (v < 0) continue;
    acc = (acc << 6) | (uint32_t)v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      bytes.push_back((uint8_t)(acc >> bits));
    }
  }
  const size_t n = bytes.size() / 2;
  std::vector<float> pcm(n);
  const int16_t *src = (const int16_t *)bytes.data();
  for (size_t i = 0; i < n; ++i) pcm[i] = src[i] / 32768.0f;
  return pcm;
}

struct Event {
  enum Kind { Start, Audio, End, Exit } kind;
  int64_t session = 0;
  std::vector<float> pcm;
  bool has_language = false;  // start: the final's language hint was given ("" = detect)
  std::string language;
};

std::mutex q_mu;
std::condition_variable q_cv;
std::deque<Event> q;
std::atomic<bool> end_pending{false};  // an "end" op is already queued: skip preview work

void reader() {
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    Event ev;
    try {
      json::Value v = json::parse(line);
      const std::string &op = v["op"].as_str();
      if (op == "start") ev.kind = Event::Start;
      else if (op == "audio") ev.kind = Event::Audio, ev.pcm = b64_pcm16(v["pcm16"].as_str());
      else if (op == "end") ev.kind = Event::End;
      else if (op == "exit") ev.kind = Event::Exit;
      else continue;
      if (auto *s = v.find("session")) ev.session = s->as_int();
      if (ev.kind == Event::Start)
        if (auto *l = v.find("language")) ev.has_language = true, ev.language = l->as_str();
    } catch (const std::exception &e) {
      fprintf(stderr, "bad input line: %s\n", e.what());
      continue;
    }
    bool stop = ev.kind == Event::Exit;
    if (ev.kind == Event::End) end_pending.store(true, std::memory_order_release);
    {
      std::lock_guard<std::mutex> g(q_mu);
      q.push_back(std::move(ev));
    }
    q_cv.notify_one();
    if (stop) return;
  }
  Event ev;
  ev.kind = Event::Exit;
  std::lock_guard<std::mutex> g(q_mu);
  q.push_back(std::move(ev));
  q_cv.notify_one();
}

std::string partial_line(long long session, const std::string &text) {
  return "{\"event\": \"partial\", \"session\": " + std::to_string(session) +
         ", \"text\": " + json::quote(text) + "}";
}

}  // namespace

int main(int argc, char **argv) {
  std::string model_path;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model-path" && i + 1 < argc) model_path = argv[++i];
  }
  if (model_path.empty()) {
    fprintf(stderr, "usage: asr_engine --model-path <dir>\n");
    return 2;
  }
  gpu::Context::get();

  s2::Stream stream(model_path);
  {
    // Qwen3-ASR 0.6B (text hidden size 1024; 1.7B is 2048): its language guess on a segment's
    // first fraction of a second is often "English" for Chinese speech, and its early guesses can
    // be whole made-up sentences. Keep the preview hint automatic (no adopt_lang), do not let the
    // draft hysteresis keep an early stop token, and do not teacher-force a retained prefix. The
    // 1.7B options are unchanged.
    s2::Options o;
    const json::Value cfg = json::parse(json::read_file(model_path + "/config.json"));
    if (cfg["thinker_config"]["text_config"]["hidden_size"].as_int() < 2048) {
      // A made-up sentence from a segment's first half second can stay identical for seconds and
      // would be teacher-forced as the retained prefix, ending in a stop token, so new speech was
      // not decoded (0348_en_long: 5 s). Re-verify the whole draft on every preview instead.
      o.adopt_lang = false;
      o.exact_stop = true;
      o.retained = 0;
      fprintf(stderr, "Qwen3-ASR 0.6B options: adopt_lang off, exact_stop on, retained 0\n");
    }
    stream.set_options(o);
  }
  // Warm up the Metal pipelines and the first-update paths on synthetic audio before the first
  // session. Without it the first update of every session is ~2x slower, and that update sets the
  // update cadence. Two updates cover the two encoder shapes a session starts with: a 0.3 s
  // partial chunk and a full 1 s chunk.
  try {
    std::vector<float> warm(16000, 0.0f);  // 1.0 s of low noise
    unsigned seed = 12345u;
    for (float &v : warm) {
      seed = seed * 1103515245u + 12345u;
      v = 0.02f * ((float)((seed >> 16) & 0x7fff) / 32768.0f - 0.5f);
    }
    stream.reset();
    stream.add(warm.data(), 16000 * 3 / 10);
    @autoreleasepool { stream.update(false); }
    stream.add(warm.data() + 16000 * 3 / 10, 16000 * 7 / 10);
    @autoreleasepool { stream.update(false); }
    stream.reset();
  } catch (const std::exception &e) {
    fprintf(stderr, "warm-up failed (ignored): %s\n", e.what());
    stream.reset();
  }
  emit("{\"event\": \"ready\"}");
  std::thread rd(reader);

  size_t last_update_samples = 0;
  // The stream emits partials from inside update() as soon as the text extends (early emit);
  // `last_partial` keeps the post-update emit from sending the same text twice.
  std::string last_partial;
  long long cur_session = 0;
  stream.set_on_partial([&](const std::string &text) {
    last_partial = text;
    emit(partial_line(cur_session, text));
  });
  double session_t0 = gpu::now_ms();
  double gpu_spent_ms = 0;
  while (true) {
    Event ev;
    {
      std::unique_lock<std::mutex> g(q_mu);
      q_cv.wait(g, [] { return !q.empty(); });
      ev = std::move(q.front());
      q.pop_front();
    }
    if (ev.kind == Event::Exit) break;
    if (ev.kind == Event::Start) {
      stream.reset();
      stream.set_final_language(ev.has_language ? &ev.language : nullptr);
      cur_session = ev.session;
      last_partial.clear();
      last_update_samples = 0;
      session_t0 = gpu::now_ms();
      gpu_spent_ms = 0;
      end_pending.store(false, std::memory_order_release);
      continue;
    }
    if (ev.kind == Event::Audio) {
      stream.add(ev.pcm.data(), ev.pcm.size());
      const size_t have = stream.num_samples();
      const size_t need = (size_t)stream.period_ms() * 16;
      // Audio arriving much faster than real time (16 samples per ms) is a recording sent at
      // once, not a microphone: previews would only cost time, the final does the work.
      const double wall_ms = gpu::now_ms() - session_t0;
      if ((double)have / 16.0 > 2.0 * wall_ms + 500.0) stream.set_unpaced(true);
      // Budget guard: skip the preview while previews have used more than kPreviewBudget of the
      // session's wall time. Whole-recording input has almost no wall time, so it goes straight
      // to the final.
      if (gpu_spent_ms > kPreviewBudget * wall_ms) continue;
      if (end_pending.load(std::memory_order_acquire)) continue;  // the final will do the work
      if (have - last_update_samples >= need) {
        bool changed = false;
        @autoreleasepool { changed = stream.update(false); }
        last_update_samples = have;
        gpu_spent_ms += stream.last_gpu_ms();
        if (changed && stream.display() != last_partial) {
          last_partial = stream.display();
          emit(partial_line(ev.session, stream.display()));
        }
      }
      continue;
    }
    if (ev.kind == Event::End) {
      bool changed = false;
      @autoreleasepool { changed = stream.update(true); }
      const std::string text = stream.final_text();
      if (changed && text != stream.display()) emit(partial_line(ev.session, stream.display()));
      emit("{\"event\": \"final\", \"session\": " + std::to_string(ev.session) + ", \"text\": " +
           json::quote(text) + ", \"generated_token_count\": " +
           std::to_string(stream.generated_token_count()) + ", \"finish_reason\": " +
           json::quote(stream.finish_reason()) + ", \"language\": " + json::quote(stream.language()) +
           "}");
      continue;
    }
  }
  rd.detach();
  return 0;
}
