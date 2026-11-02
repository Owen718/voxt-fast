# SoL-Voxt

SoL-Voxt（原名 Voxt Fast）是 macOS 听写应用 [Voxt](https://github.com/sgl-project/sglang-omni/tree/main/Voxt) 的一个改版。它把 Voxt 里的两个本地模型换成了加速引擎，模型本身没有换：

- **语音识别**：Qwen3-ASR 1.7B（6 bit）的流式引擎。说话时，屏幕上的文字跟得更紧。
- **文字润色**：Qwen3.5-4B-OptiQ（4 bit）的推理引擎。说完后，去口癖、改标点这一步更快完成。

这是非官方改版，与 Voxt 和 SGLang-Omni 官方无关。

两个引擎都是用 SoL-Pi-Swarm（多个 AI agent 协作优化代码的系统）在一台 M4 Pro 上自动优化出来的。优化过程中，模型权重、提示词和解码规则都没有改动。评分器会检查每个候选版本的输出是否与原版一致，不一致的版本记零分。

## 效果

测试机是 MacBook Pro（M4 Pro，48 GB），表中是中位数。两行数字都是本仓库的引擎在测试机空闲时，用各自优化任务的评分器测得的。

| 环节 | 原版 | SoL-Voxt |
|---|---|---|
| 说话时屏幕出字的延迟（短 / 中 / 长句） | 319 / 397 / 388 ms | 64 / 83 / 98 ms |
| 说完后润色完成的耗时（短 / 中 / 长段） | 1.57 / 1.97 / 3.83 s | 0.20 / 0.27 / 0.70 s |

两个“原版”的含义：

- 语音识别的原版是 SGLang-Omni 的实时识别服务，每 500 ms 刷新一次。在 GPU 占用不超过一半的前提下，500 ms 是它能做到的最短刷新间隔。
- 润色的原版是 Voxt 现在的做法：每个请求新建一个 mlx-swift-lm 会话。

**出字延迟**的含义：某个字从它能被识别出来的那一刻，到它稳定显示在屏幕上，中间隔了多久。

**正确性**：

- **语音识别**：392 条中英文和中英混合的测试录音，整体错误率 4.12%，原版为 4.42%。说完后插入的最终文字走引擎的离线精确路径，和预览文字无关。
- **润色**：384 条测试输入里，378 条输出与 Voxt 原版逐字相同。其余几条只在模型几乎打平的位置选了另一个词，评分器校准过，这种差异属于正常的浮点误差。

**内存**：润色引擎的峰值约 5.9 GB，原版为 5.6 GB。识别引擎约 2.6 GB。

## 安装

1. 从 [Releases](../../releases) 下载 `SoL-Voxt.zip`（v0.1.2 及以前为 `Voxt-Fast.zip`），解压后把 `SoL-Voxt.app` 拖进“应用程序”。从 Voxt Fast 升级时，先退出并删除旧的 `Voxt Fast.app`；bundle id 没变，设置和已下载的模型都会保留。
2. 这个应用没有经过 Apple 公证，第一次打开前要在终端里去掉隔离标记：

   ```bash
   xattr -dr com.apple.quarantine "/Applications/SoL-Voxt.app"
   ```

3. 打开应用，按提示授予麦克风和辅助功能权限。辅助功能权限用于把文字输入到其他应用。

要求 Apple Silicon 的 Mac 和 macOS 15 以上，内存建议 16 GB 以上。目前只在 M4 Pro / macOS 26 上测过，其他芯片上的速度和稳定性还没有验证。

## 首次设置

两个加速引擎只在选中对应模型时才生效。从 v0.1.2 起，新安装默认选中的就是这两个模型，首次引导里也排在最前面：

- **听写模型**：**Qwen3-ASR 1.7B (6bit)**，约 2.0 GB。第一次使用前，在首次引导或模型设置里下载它。
- **润色模型**：在 AI 增强里选本地模型 **Qwen3.5-4B-OptiQ-4bit** 并下载。润色参数要保持默认，即温度、重复惩罚等都不改，思考保持关闭。改了这些参数的请求会自动改走 Voxt 原来的路径。

选其他听写模型也能用，例如 Qwen3-ASR 0.6B 或 1.7B (8bit)。这些模型走 SGLang-Omni 原来的识别服务，速度与原版 Voxt 相同。如果你装过 v0.1.1 并选了 0.6B，升级后设置会保留，请在模型设置里改选 1.7B (6bit)。v0.1.1 里选 0.6B 会报“network connection was lost”，原因是识别引擎不支持该模型而崩溃，v0.1.2 已修复。
- **快捷键**：如果要用自定义快捷键，先把快捷键预设改成“自定义”，否则启动时会被预设覆盖。

**蓝牙耳机**：从 v0.1.3 起，如果听写选中的麦克风是 AirPods 等蓝牙耳机，并且笔记本开着盖，听写会改用 MacBook 内置麦克风，声音照常从耳机播放。这是豆包输入法的做法。原因是蓝牙耳机的麦克风打开时要先切到通话模式，开头 0.3–2 秒的话会录不到。合盖时仍用耳机麦克风；USB、有线麦克风不受影响；会议录音不受影响。托盘的麦克风菜单里会显示这条说明。如果一定要用耳机麦克风听写：

```bash
defaults write com.voxt.Voxt.fast VoxtFastUseBluetoothMicForDictation -bool true
```

确认引擎是否在工作：

- 听写时，活动监视器里能看到 `asr_engine` 进程。
- 润色时能看到 `voxt-llm-engine` 进程。
- 润色引擎的日志在 `~/Library/Logs/SoL-Voxt/llm-engine.log`。

如果想临时关掉润色加速、改回原来的路径：

```bash
defaults write com.voxt.Voxt.fast VoxtFastLLMDisabled -bool true
```

## 工作原理

### 语音识别（`engines/asr`）

这是一个用 Objective-C++ 和 Metal 从头写的 Qwen3-ASR 推理引擎，数值计算与 MLX 保持一致。速度主要来自以下几点：

- **增量计算**：每次刷新只重新编码最后 2 秒音频。更早音频的解码器 KV 缓存直接复用。
- **前 8 秒每 100 ms 刷新一次**：刷新间隔按每次刷新的 GPU 耗时动态调整，同时限制预览占用的 GPU 时间不超过一半。
- **按置信度决定何时显示**：一个词的置信度足够高，或者最近几次识别结果一致，才显示出来。已经显示的前缀会被固定，后续刷新只解码尾部，所以屏幕上的字很少回退。
- **提前显示**：每解出一个 token 就检查一次能否显示，不用等整次刷新结束。

### 文字润色（`engines/llm`）

这是 Voxt 原来的 mlx-swift-lm 推理路径，外加以下改动：

- **跨请求缓存系统提示词**：Voxt 每次请求都会重新预填充约 950 个 token 的系统提示词。现在只算一次，之后的请求从缓存的状态接着算。
- **精确的投机解码**：润色结果大部分是照抄原文，所以直接从转写原文里复制一段作为草稿，让模型一次前向验证。只有与贪心解码结果相同的 token 才会被接受，所以输出与逐个 token 解码完全一样。Qwen3.5 的 Gated DeltaNet 层在回滚时会重算循环状态。
- **内核改动**：
  - 编译后的验证前向；
  - 每次处理 4 行的 GDN Metal 内核；
  - 9 行和 12 到 16 行的量化矩阵乘法使用 16 行分块；
  - 预填充阶段只对最后一行算 logits。

对 mlx-swift 和 mlx-swift-lm 的改动都在 `engines/llm/patches/` 里。

### 接入 Voxt

- **语音识别**：Voxt 通过 OmniRuntime 里的 `qwen3_asr_server` 访问本地识别服务（HTTP 加 WebSocket）。`adapter/main.swift` 用 Swift 实现了同样的接口，背后调用 `asr_engine`。Silero VAD 等其他模型仍交给 SGLang-Omni 原来的服务。
- **润色**：`Voxt/Voxt/Core/Models/FastLLMEngine.swift` 管理 `voxt-llm-engine` 进程。满足条件的本地润色请求会交给它，条件是模型为 Qwen3.5-4B-OptiQ、没有多轮历史、解码参数为默认值。其他请求仍走 Voxt 原来的路径。空闲卸载和切换模型时，引擎进程会一起停掉。

## 从源码构建

需要 Apple Silicon、Xcode（含 Metal 工具链）和 [uv](https://docs.astral.sh/uv/)。第一次构建需要联网，用来拉取依赖和 SGLang-Omni。

```bash
scripts/build-app.sh
```

脚本按以下顺序工作：

1. 编译两个引擎和转接程序。
2. 按 Voxt 所在的 commit 编译 SGLang-Omni 的原生运行时。
3. 编译 Voxt。
4. 把引擎放进 `SoL-Voxt.app/Contents/Resources`，对整个应用做本机签名。

结果在 `dist/SoL-Voxt.app` 和 `dist/SoL-Voxt.zip`。也可以单独编译两个引擎：

```bash
engines/asr/build.sh          # -> engines/asr/build/bin/asr_engine
engines/llm/scripts/build.sh  # -> engines/llm/build/bin/voxt-llm-engine
```

两个引擎都用 JSON lines 协议通过 stdin/stdout 通信，协议写在各自 `main` 文件的开头。

## 目录结构

| 路径 | 内容 |
|---|---|
| `Voxt/` | Voxt 源码，来自 sglang-omni 8b81cda。改动见下一节 |
| `engines/asr/` | Qwen3-ASR 流式引擎（Objective-C++ / Metal） |
| `engines/llm/` | Qwen3.5-4B 润色引擎（Swift / MLX），以及对 mlx-swift、mlx-swift-lm 的补丁 |
| `adapter/` | Voxt 的本地识别服务接口，背后是 `asr_engine` |
| `scripts/build-app.sh` | 编译并打包整个应用 |
| `docs/` | 两次优化的结果卡片（PDF） |

## 与 Voxt 原版的差异

Voxt 源码以单独的提交导入，之后的改动都在后续提交里。改动只有以下几处：

- `Voxt/Core/Models/FastLLMEngine.swift`（新增）和 `CustomLLMModelManager.swift`：把满足条件的本地润色请求交给加速引擎。
- `Voxt/Config/Fast.xcconfig`（新增）：应用名改为“SoL-Voxt”（原名 Voxt Fast），bundle id 改为 `com.voxt.Voxt.fast`，不启用沙盒，使用本机签名。
- `AppUpdateManager.swift`：对 `.fast` 关闭 Sparkle 自动更新，避免被官方版本覆盖。

## 已知问题

- 只在 M4 Pro 上测过。其他芯片上内核分块的速度收益可能不同，但结果应该不变。
- 应用没有公证，需要手动去掉隔离标记。每次重新签名后，macOS 可能要求重新授予权限。
- 两个模型同时加载时，内存占用约 8.5 GB。

## 许可

本仓库以 Apache-2.0 发布。第三方代码：

- Voxt 和 SGLang-Omni：Apache-2.0。
- MLX、mlx-swift、mlx-swift-lm：MIT。

详见 `LICENSE`、`NOTICE` 和各目录下的 `NOTICE`。
