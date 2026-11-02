// FastLLMEngine.swift
// Runs local Qwen3.5-4B-OptiQ requests on the bundled voxt-llm-engine process.

import Foundation
import MLXLMCommon

/// The bundled `voxt-llm-engine` (Contents/Resources/FastLLM): the same model, prompt and
/// greedy decoding as the in-process MLX path, with the system prompt's state kept between
/// requests and exact speculative decoding. One request runs at a time; the process keeps the
/// model loaded until `stop()`.
actor FastLLMEngine {
    nonisolated static let supportedRepo = "mlx-community/Qwen3.5-4B-OptiQ-4bit"
    /// Set to true (defaults write com.voxt.Voxt.fast VoxtFastLLMDisabled -bool true) to use
    /// the in-process MLX path for every request.
    nonisolated static let disabledDefaultsKey = "VoxtFastLLMDisabled"

    nonisolated static var executableURL: URL? {
        Bundle.main.resourceURL?.appendingPathComponent("FastLLM/voxt-llm-engine")
    }

    nonisolated static var isAvailable: Bool {
        guard !UserDefaults.standard.bool(forKey: disabledDefaultsKey),
              let url = executableURL else { return false }
        return FileManager.default.isExecutableFile(atPath: url.path)
    }

    enum EngineError: LocalizedError {
        case notBundled
        case exited(String)
        case failed(String)

        var errorDescription: String? {
            switch self {
            case .notBundled: return "The fast LLM engine is not bundled with this build."
            case .exited(let detail): return "The fast LLM engine stopped. \(detail)"
            case .failed(let message): return "The fast LLM engine failed: \(message)"
            }
        }
    }

    private var process: Process?
    private var input: FileHandle?
    private var events: AsyncLineSequence<FileHandle.AsyncBytes>.AsyncIterator?
    private var modelDirectory: URL?
    private var nextRequestID = 1
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    var isRunning: Bool { process?.isRunning == true }

    /// Starts the engine for `modelDirectory` if it is not already serving it.
    func prepare(modelDirectory: URL) async throws {
        await acquire()
        defer { release() }
        try await start(modelDirectory: modelDirectory)
    }

    /// One cleanup request; `onChunk` receives the text as it is generated. Returns the full text.
    func generate(
        modelDirectory: URL,
        system: String,
        user: String,
        maxTokens: Int,
        onChunk: @Sendable (String) -> Void
    ) async throws -> String {
        await acquire()
        defer { release() }
        try await start(modelDirectory: modelDirectory)
        let id = nextRequestID
        nextRequestID += 1
        try send(["op": "generate", "id": id, "system": system, "user": user, "max_tokens": maxTokens])
        while let event = try await nextEvent() {
            // Events of an earlier, cancelled request are skipped.
            guard event["id"] as? Int == id else { continue }
            switch event["event"] as? String {
            case "delta":
                if let text = event["text"] as? String, !Task.isCancelled { onChunk(text) }
            case "done":
                return event["text"] as? String ?? ""
            case "error":
                throw EngineError.failed(event["message"] as? String ?? "unknown error")
            default:
                continue
            }
        }
        stopProcess()
        throw EngineError.exited("It ended during a request.")
    }

    /// Ends the process and releases the model's memory.
    func stop() async {
        await acquire()
        defer { release() }
        stopProcess()
    }

    // MARK: - Process

    private func start(modelDirectory: URL) async throws {
        if isRunning, self.modelDirectory == modelDirectory { return }
        stopProcess()
        guard let executable = Self.executableURL else { throw EngineError.notBundled }
        let process = Process()
        process.executableURL = executable
        process.arguments = ["--model", modelDirectory.path, "--repo", Self.supportedRepo]
        let stdin = Pipe()
        let stdout = Pipe()
        _ = fcntl(stdin.fileHandleForWriting.fileDescriptor, F_SETNOSIGPIPE, 1)
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = Self.logHandle()
        try process.run()
        self.process = process
        self.input = stdin.fileHandleForWriting
        self.events = stdout.fileHandleForReading.bytes.lines.makeAsyncIterator()
        self.modelDirectory = modelDirectory
        VoxtLog.modelInfo("Fast LLM engine starting. pid=\(process.processIdentifier), model=\(modelDirectory.lastPathComponent)")
        while let event = try await nextEvent() {
            if event["event"] as? String == "ready" {
                VoxtLog.modelInfo("Fast LLM engine ready. load_ms=\(event["load_ms"] ?? "?"), warmup_ms=\(event["warmup_ms"] ?? "?")")
                return
            }
        }
        stopProcess()
        throw EngineError.exited("It did not start; see ~/Library/Logs/SoL-Voxt/llm-engine.log.")
    }

    private func stopProcess() {
        guard let process else { return }
        if process.isRunning {
            try? input?.write(contentsOf: Data("{\"op\": \"exit\"}\n".utf8))
        }
        try? input?.close()
        if process.isRunning {
            let pid = process.processIdentifier
            DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
                if kill(pid, 0) == 0 { kill(pid, SIGTERM) }
            }
        }
        self.process = nil
        input = nil
        events = nil
        modelDirectory = nil
        VoxtLog.modelInfo("Fast LLM engine stopped.")
    }

    private func send(_ object: [String: Any]) throws {
        guard let input else { throw EngineError.exited("It is not running.") }
        var data = try JSONSerialization.data(withJSONObject: object)
        data.append(0x0A)
        try input.write(contentsOf: data)
    }

    /// The next JSON event from the engine's stdout, or nil when it has exited. Only called by
    /// the request holding the turn, so the iterator is not used concurrently.
    private func nextEvent() async throws -> [String: Any]? {
        guard var iterator = events else { return nil }
        defer { if events != nil { events = iterator } }
        while let line = try await iterator.next() {
            if let event = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any] {
                return event
            }
        }
        return nil
    }

    private static func logHandle() -> FileHandle {
        let directory = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/SoL-Voxt", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("llm-engine.log")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        return (try? FileHandle(forWritingTo: url)) ?? FileHandle.nullDevice
    }

    // MARK: - One request at a time

    private func acquire() async {
        if busy {
            await withCheckedContinuation { waiters.append($0) }
        } else {
            busy = true
        }
    }

    private func release() {
        if waiters.isEmpty {
            busy = false
        } else {
            waiters.removeFirst().resume()  // hand the turn over; `busy` stays true
        }
    }
}

extension CustomLLMModelManager {
    /// Whether a request with these resolved settings decodes exactly like the engine:
    /// greedy, Voxt's repetition penalty, the default prefill chunking and thinking off.
    static func fastEngineSupports(
        repo: String,
        parameters: GenerateParameters,
        settings: LLMGenerationSettings,
        tuning: CustomLLMGenerationTuning
    ) -> Bool {
        repo == FastLLMEngine.supportedRepo
            && FastLLMEngine.isAvailable
            && settings.thinking.mode != .on
            && settings.stop.isEmpty
            && tuning.prefillStepSizeOverride == nil
            && parameters.maxTokens != nil
            && parameters.temperature == 0
            && parameters.topP == 1
            && parameters.topK == 0
            && parameters.minP == 0
            && parameters.repetitionPenalty == 1.05
            && parameters.repetitionContextSize == 32
            && parameters.presencePenalty == nil
            && parameters.frequencyPenalty == nil
            && parameters.kvBits == nil
    }
}
