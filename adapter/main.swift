// qwen3_asr_server: Voxt's local Qwen3-ASR server interface in front of asr_engine.
//
// Voxt starts `qwen3_asr_server --supervised --model-kind K --model-directory D ...` from its
// OmniRuntime and talks to it over loopback HTTP. For K = qwen3_asr this program serves:
//   - start-up: `{"event":"ready","host":..,"port":..,"model_name":..,"server_pid":..}` on stdout;
//     `{"command":"shutdown"}` or the end of stdin stops it;
//   - GET /health, GET /v1/models;
//   - POST /v1/audio/transcriptions (multipart WAV) -> SSE `transcript.text.done` + `[DONE]`:
//     the final transcript, which Voxt requests for the whole recording when the user stops;
//   - GET /v1/realtime (WebSocket): the live preview while the user speaks. Each engine partial
//     becomes one `transcription.segment` (segment 0) that replaces the previous one.
// Every other model kind (Silero VAD, Sortformer, ...) is handed to SGLang-Omni's own server,
// installed next to this program as qwen3_asr_server.sglang.
//
// asr_engine runs one utterance at a time and a `start` op resets it, so a new session takes
// over the engine: Voxt cancels the live session when the user stops and then sends the final.

import CryptoKit
import Foundation
import Network

let here = Bundle.main.executableURL!.resolvingSymlinksInPath().deletingLastPathComponent()
let queue = DispatchQueue(label: "qwen3_asr_server")  // every piece of state lives on this queue

func log(_ message: String) {
    FileHandle.standardError.write(Data("qwen3_asr_server: \(message)\n".utf8))
}

func jsonLine(_ object: [String: Any]) -> Data {
    var data = (try? JSONSerialization.data(withJSONObject: object, options: [.withoutEscapingSlashes])) ?? Data()
    data.append(0x0A)
    return data
}

func printEvent(_ object: [String: Any]) {
    FileHandle.standardOutput.write(jsonLine(object))
}

// MARK: - Arguments

var supervised = false
var modelKind = "qwen3_asr"
var modelDirectory: String?
var startupTimeout = 120.0
var host = "127.0.0.1"
var port: UInt16 = 0
do {
    var args = CommandLine.arguments.dropFirst().makeIterator()
    while let arg = args.next() {
        switch arg {
        case "--supervised": supervised = true
        case "--model-kind": modelKind = args.next() ?? modelKind
        case "--model-directory", "--model-path": modelDirectory = args.next()
        case "--startup-timeout-s": startupTimeout = Double(args.next() ?? "") ?? startupTimeout
        case "--host": host = args.next() ?? host
        case "--port": port = UInt16(args.next() ?? "") ?? port
        default: break  // options of the other model kinds
        }
    }
}

// asr_engine runs Qwen3-ASR 1.7B (6-bit) and 0.6B (4-bit); other Qwen3-ASR checkpoints (1.7B
// 8-bit, ...) and every other model kind go to SGLang-Omni's own server at the original speed.
let fastModelDirectories = ["qwen3-asr-1.7b-6bit", "qwen3-asr-0.6b-4bit"]
let servesFastModel = modelKind == "qwen3_asr" &&
    (modelDirectory.map { dir in
        let name = URL(fileURLWithPath: dir).lastPathComponent.lowercased()
        return fastModelDirectories.contains { name.contains($0) }
    } ?? false)
if !servesFastModel {
    let server = here.appendingPathComponent("qwen3_asr_server.sglang").path
    var argv: [UnsafeMutablePointer<CChar>?] = [strdup(server)]
    argv += CommandLine.arguments.dropFirst().map { strdup($0) }
    argv.append(nil)
    execv(server, argv)
    log("cannot run \(server): \(String(cString: strerror(errno)))")
    exit(1)
}

func fail(_ reason: String) -> Never {
    log(reason)
    if supervised { printEvent(["event": "failed", "reason": reason]) }
    exit(1)
}

guard let modelPath = modelDirectory else { fail("--model-directory is required") }
let modelName = URL(fileURLWithPath: modelPath).lastPathComponent

// MARK: - Engine

/// The asr_engine process. Events are routed to the session that owns them.
final class Engine {
    private let process = Process()
    private let input = Pipe()
    private let output = Pipe()
    private var handlers: [Int: ([String: Any]) -> Void] = [:]
    private var nextSession = 1
    private(set) var current = 0  // the newest session; older ones stop sending
    var busy = 0                  // sessions in flight, reported by /health

    func start(modelPath: String, timeout: Double) throws {
        process.executableURL = here.appendingPathComponent("asr_engine")
        process.arguments = ["--model-path", modelPath]
        process.standardInput = input
        process.standardOutput = output
        process.standardError = FileHandle.standardError
        _ = fcntl(input.fileHandleForWriting.fileDescriptor, F_SETNOSIGPIPE, 1)
        try process.run()
        let ready = DispatchSemaphore(value: 0)
        Thread.detachNewThread { [self] in
            var buffer = Data()
            var isReady = false
            while true {
                let chunk = output.fileHandleForReading.availableData
                if chunk.isEmpty { break }
                buffer.append(chunk)
                while let newline = buffer.firstIndex(of: 0x0A) {
                    let line = buffer[buffer.startIndex ..< newline]
                    buffer.removeSubrange(buffer.startIndex ... newline)
                    guard let event = (try? JSONSerialization.jsonObject(with: line)) as? [String: Any] else {
                        continue
                    }
                    if !isReady {
                        if event["event"] as? String == "ready" {
                            isReady = true
                            ready.signal()
                        }
                        continue
                    }
                    queue.async { self.route(event) }
                }
            }
            if !isReady { ready.signal() }
            log("asr_engine exited")
            exit(1)  // Voxt notices the exit and starts a new server on next use
        }
        if ready.wait(timeout: .now() + timeout) == .timedOut {
            process.terminate()
            throw NSError(domain: "qwen3_asr_server", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "asr_engine was not ready within \(Int(timeout)) s"])
        }
    }

    private func route(_ event: [String: Any]) {
        guard let session = event["session"] as? Int else { return }
        handlers[session]?(event)
    }

    private func send(_ op: [String: Any]) {
        input.fileHandleForWriting.write(jsonLine(op))
    }

    /// Opens a session; `language` is the final's hint as Voxt sent it ("" lets the model detect).
    func open(language: String, handler: @escaping ([String: Any]) -> Void) -> Int {
        let session = nextSession
        nextSession += 1
        handlers[session] = handler
        current = session
        send(["op": "start", "session": session, "language": language])
        return session
    }

    func audio(_ session: Int, pcm16Base64: String) {
        send(["op": "audio", "session": session, "pcm16": pcm16Base64])
    }

    func end(_ session: Int) {
        send(["op": "end", "session": session])
    }

    func close(_ session: Int) {
        handlers[session] = nil
    }

    func stop() {
        guard process.isRunning else { return }
        send(["op": "exit"])
        process.terminate()
    }
}

// MARK: - Audio

/// Voxt sends float32 WAV (format 3); 16-bit PCM is accepted too. Returns 16 kHz mono s16le.
func pcm16(fromWAV data: Data) throws -> Data {
    func wavError(_ message: String) -> Error {
        NSError(domain: "wav", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
    let bytes = [UInt8](data)
    func u16(_ i: Int) -> Int { Int(bytes[i]) | Int(bytes[i + 1]) << 8 }
    func u32(_ i: Int) -> Int { u16(i) | u16(i + 2) << 16 }
    guard bytes.count >= 12, bytes[0 ..< 4] == [0x52, 0x49, 0x46, 0x46], bytes[8 ..< 12] == [0x57, 0x41, 0x56, 0x45]
    else { throw wavError("not a WAV file") }
    var pos = 12
    var format = 0, channels = 0, rate = 0, bits = 0
    var payload: Range<Int>?
    while pos + 8 <= bytes.count {
        let id = String(bytes: bytes[pos ..< pos + 4], encoding: .ascii) ?? ""
        let size = u32(pos + 4)
        let body = pos + 8
        if id == "fmt " {
            format = u16(body)
            channels = u16(body + 2)
            rate = u32(body + 4)
            bits = u16(body + 14)
        } else if id == "data" {
            payload = body ..< min(bytes.count, body + size)
        }
        pos = body + size + (size & 1)
    }
    guard let payload, format != 0 else { throw wavError("WAV without fmt or data chunk") }
    guard rate == 16000, channels == 1 else { throw wavError("expected 16 kHz mono, got \(rate) Hz x\(channels)") }
    if format == 1, bits == 16 { return Data(bytes[payload]) }
    guard format == 3, bits == 32 else { throw wavError("unsupported WAV format \(format)/\(bits) bit") }
    let count = payload.count / 4
    var out = Data(count: count * 2)
    data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            let samples = dst.bindMemory(to: Int16.self)
            for i in 0 ..< count {
                let bitPattern = raw.loadUnaligned(fromByteOffset: payload.lowerBound + 4 * i, as: UInt32.self)
                // x * 32767 in float32, rounded half to even and clipped (as numpy.rint + clip).
                let x = (Float(bitPattern: UInt32(littleEndian: bitPattern)) * 32767).rounded(.toNearestOrEven)
                samples[i] = Int16(littleEndian: x.isNaN ? 0 : Int16(max(-32768, min(32767, x))))
            }
        }
    }
    return out
}

// MARK: - HTTP

struct Request {
    var method = ""
    var path = ""
    var headers: [String: String] = [:]  // lowercased names
    var body = Data()
}

/// Splits a multipart/form-data body into its named parts.
func multipartParts(_ body: Data, contentType: String) -> [String: Data] {
    guard let range = contentType.range(of: "boundary=") else { return [:] }
    var boundary = String(contentType[range.upperBound...])
    if let semicolon = boundary.firstIndex(of: ";") { boundary = String(boundary[..<semicolon]) }
    boundary = boundary.trimmingCharacters(in: CharacterSet(charactersIn: "\" "))
    let delimiter = Data("--\(boundary)".utf8)
    let headerEnd = Data("\r\n\r\n".utf8)
    var parts: [String: Data] = [:]
    var cursor = body.startIndex
    while let start = body.range(of: delimiter, in: cursor ..< body.endIndex) {
        let partStart = start.upperBound
        guard let next = body.range(of: delimiter, in: partStart ..< body.endIndex) else { break }
        cursor = next.lowerBound
        guard let headersEnd = body.range(of: headerEnd, in: partStart ..< next.lowerBound) else { continue }
        let header = String(decoding: body[partStart ..< headersEnd.lowerBound], as: UTF8.self)
        guard let nameRange = header.range(of: "name=\"") else { continue }
        let name = header[nameRange.upperBound...].prefix { $0 != "\"" }
        var contentEnd = next.lowerBound
        if contentEnd - headersEnd.upperBound >= 2 { contentEnd -= 2 }  // the CRLF before the delimiter
        parts[String(name)] = body.subdata(in: headersEnd.upperBound ..< contentEnd)
    }
    return parts
}

/// One TCP connection: an HTTP/1.1 request, answered and closed, or a WebSocket.
final class Connection {
    private let connection: NWConnection
    private let engine: Engine
    private var buffer = Data()
    private var request: Request?
    private var bodyLength = 0
    // WebSocket state
    private var isWebSocket = false
    private var fragments = Data()
    private var session = 0
    private var eventIndex = 0
    private var closed = false
    var onClose: (() -> Void)?

    init(_ connection: NWConnection, engine: Engine) {
        self.connection = connection
        self.engine = engine
    }

    func start() {
        connection.stateUpdateHandler = { [self] state in
            switch state {
            case .failed, .cancelled:
                finishSession()
                onClose?()
                onClose = nil
            default:
                break
            }
        }
        connection.start(queue: queue)
        receive()
    }

    private func receive() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [self] data, _, complete, error in
            if let data { buffer.append(data) }
            if isWebSocket { readFrames() } else { readRequest() }
            if complete || error != nil {
                connection.cancel()
            } else if !closed {
                receive()
            }
        }
    }

    // MARK: HTTP request

    private func readRequest() {
        if request == nil {
            guard let end = buffer.range(of: Data("\r\n\r\n".utf8)) else { return }
            let lines = String(decoding: buffer[buffer.startIndex ..< end.lowerBound], as: UTF8.self)
                .components(separatedBy: "\r\n")
            buffer.removeSubrange(buffer.startIndex ..< end.upperBound)
            var req = Request()
            let parts = lines[0].split(separator: " ")
            guard parts.count >= 2 else { return respond(400, json: ["error": "bad request line"]) }
            req.method = String(parts[0])
            req.path = String(parts[1].split(separator: "?", maxSplits: 1)[0])
            for line in lines.dropFirst() {
                guard let colon = line.firstIndex(of: ":") else { continue }
                req.headers[line[..<colon].lowercased()] =
                    line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            }
            bodyLength = Int(req.headers["content-length"] ?? "") ?? 0
            if req.headers["expect"]?.lowercased() == "100-continue" {
                connection.send(content: Data("HTTP/1.1 100 Continue\r\n\r\n".utf8), completion: .idempotent)
            }
            request = req
        }
        guard var req = request, buffer.count >= bodyLength else { return }
        req.body = buffer.prefix(bodyLength)
        buffer.removeFirst(bodyLength)
        handle(req)
    }

    private func handle(_ req: Request) {
        switch (req.method, req.path) {
        case ("GET", "/health"):
            respond(200, json: ["status": "ok", "model": modelName,
                                "request_states": ["running": engine.busy, "waiting": 0]])
        case ("GET", "/v1/models"):
            respond(200, json: ["object": "list", "data": [["id": modelName, "object": "model"]]])
        case ("POST", "/v1/audio/transcriptions"):
            transcribe(req)
        case ("GET", "/v1/realtime") where req.headers["upgrade"]?.lowercased() == "websocket":
            upgrade(req)
        default:
            respond(404, json: ["error": ["message": "not found", "type": "invalid_request_error"]])
        }
    }

    private func respond(_ status: Int, json object: [String: Any]) {
        let body = (try? JSONSerialization.data(withJSONObject: object)) ?? Data()
        let reason = [200: "OK", 400: "Bad Request", 404: "Not Found"][status] ?? "Error"
        var head = "HTTP/1.1 \(status) \(reason)\r\nContent-Type: application/json\r\n"
        head += "Content-Length: \(body.count)\r\nConnection: close\r\n\r\n"
        connection.send(content: Data(head.utf8) + body, isComplete: true, completion: .contentProcessed { [self] _ in
            connection.cancel()
        })
        closed = true
    }

    /// The final: the whole recording in one audio op, so the engine sees it far faster than
    /// real time and takes its exact offline path.
    private func transcribe(_ req: Request) {
        let parts = multipartParts(req.body, contentType: req.headers["content-type"] ?? "")
        func field(_ name: String) -> String { parts[name].map { String(decoding: $0, as: UTF8.self) } ?? "" }
        guard let file = parts["file"] else {
            return respond(400, json: ["error": ["message": "missing file", "type": "invalid_request_error"]])
        }
        let pcm: Data
        do {
            pcm = try pcm16(fromWAV: file)
        } catch {
            return respond(400, json: ["error": ["message": error.localizedDescription, "type": "invalid_request_error"]])
        }
        let stream = field("stream") == "true"
        let withMetadata = field("include_generation_metadata") == "true"
        engine.busy += 1
        session = engine.open(language: field("language")) { [self] event in
            guard event["event"] as? String == "final" else { return }
            engine.close(session)
            engine.busy -= 1
            session = 0
            var done: [String: Any] = ["type": "transcript.text.done", "text": event["text"] as? String ?? ""]
            if withMetadata {
                let language = (event["language"] as? String).flatMap { $0.isEmpty ? nil : $0 }
                done["generation_metadata"] = [
                    "generated_token_count": event["generated_token_count"] as? Int ?? 0,
                    "finish_reason": event["finish_reason"] as? String ?? "stop",
                    "language": language as Any? ?? NSNull(),
                ]
            }
            if !stream { return respond(200, json: ["text": done["text"]!]) }
            var payload = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
            payload += "Connection: close\r\n\r\n"
            let event = String(decoding: try! JSONSerialization.data(withJSONObject: done, options: [.withoutEscapingSlashes]), as: UTF8.self)
            payload += "data: \(event)\n\ndata: [DONE]\n\n"
            connection.send(content: Data(payload.utf8), isComplete: true, completion: .contentProcessed { [self] _ in
                connection.cancel()
            })
            closed = true
        }
        engine.audio(session, pcm16Base64: pcm.base64EncodedString())
        engine.end(session)
    }

    // MARK: WebSocket (live preview)

    private func upgrade(_ req: Request) {
        guard let key = req.headers["sec-websocket-key"] else {
            return respond(400, json: ["error": "missing Sec-WebSocket-Key"])
        }
        let accept = Data(Insecure.SHA1.hash(data: Data((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").utf8)))
            .base64EncodedString()
        let head = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            + "Sec-WebSocket-Accept: \(accept)\r\n\r\n"
        connection.send(content: Data(head.utf8), completion: .idempotent)
        isWebSocket = true
        engine.busy += 1
        session = engine.open(language: "") { [self] event in
            switch event["event"] as? String {
            case "partial":
                eventIndex += 1
                sendText(["type": "transcription.segment", "event_index": eventIndex, "segment_id": 0,
                          "text": event["text"] as? String ?? "", "is_final": false])
            case "final":
                sendText(["type": "transcription.completed", "text": event["text"] as? String ?? ""])
                closeWebSocket()
            default:
                break
            }
        }
        readFrames()
    }

    private func readFrames() {
        while !closed, buffer.count >= 2 {
            let b = [UInt8](buffer.prefix(14))
            let fin = b[0] & 0x80 != 0
            let opcode = b[0] & 0x0F
            let masked = b[1] & 0x80 != 0
            var length = Int(b[1] & 0x7F)
            var offset = 2
            if length == 126 {
                guard b.count >= 4 else { return }
                length = Int(b[2]) << 8 | Int(b[3])
                offset = 4
            } else if length == 127 {
                guard b.count >= 10 else { return }
                length = (2 ..< 10).reduce(0) { $0 << 8 | Int(b[$1]) }
                offset = 10
            }
            let maskOffset = offset
            if masked { offset += 4 }
            guard buffer.count >= offset + length else { return }
            var payload = [UInt8](buffer.dropFirst(offset).prefix(length))
            if masked {
                let mask = [UInt8](buffer.dropFirst(maskOffset).prefix(4))
                for i in payload.indices { payload[i] ^= mask[i % 4] }
            }
            buffer.removeFirst(offset + length)
            switch opcode {
            case 0x0, 0x1, 0x2:
                fragments.append(contentsOf: payload)
                if fin {
                    if opcode != 0x2 { handleMessage(fragments) }
                    fragments.removeAll()
                }
            case 0x8:
                closeWebSocket()
            case 0x9:
                sendFrame(opcode: 0xA, payload: Data(payload))
            default:
                break
            }
        }
    }

    private func handleMessage(_ data: Data) {
        guard session != 0, let message = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return
        }
        if engine.current != session {
            return closeWebSocket()  // a newer session (usually Voxt's final request) took the engine
        }
        switch message["type"] as? String {
        case "input_audio_buffer.append":
            engine.audio(session, pcm16Base64: message["audio"] as? String ?? "")
        case "transcription.done":
            engine.end(session)
        default:
            break  // session.update and input_audio_buffer.commit need no action
        }
    }

    private func sendText(_ object: [String: Any]) {
        guard !closed, let data = try? JSONSerialization.data(withJSONObject: object, options: [.withoutEscapingSlashes])
        else { return }
        sendFrame(opcode: 0x1, payload: data)
    }

    private func sendFrame(opcode: UInt8, payload: Data) {
        var frame = Data([0x80 | opcode])
        if payload.count < 126 {
            frame.append(UInt8(payload.count))
        } else if payload.count < 65536 {
            frame.append(contentsOf: [126, UInt8(payload.count >> 8), UInt8(payload.count & 0xFF)])
        } else {
            frame.append(127)
            frame.append(contentsOf: (0 ..< 8).reversed().map { UInt8((payload.count >> (8 * $0)) & 0xFF) })
        }
        frame.append(payload)
        connection.send(content: frame, completion: .idempotent)
    }

    private func closeWebSocket() {
        guard !closed else { return }
        sendFrame(opcode: 0x8, payload: Data([0x03, 0xE8]))  // 1000: normal closure
        closed = true
        finishSession()
        connection.send(content: nil, isComplete: true, completion: .contentProcessed { [self] _ in
            connection.cancel()
        })
    }

    /// Releases the engine session of this connection, if it still holds one.
    private func finishSession() {
        guard session != 0 else { return }
        engine.close(session)
        engine.busy -= 1
        session = 0
    }
}

// MARK: - Main

let engine = Engine()
do {
    try engine.start(modelPath: modelPath, timeout: startupTimeout)
} catch {
    fail("asr_engine: \(error.localizedDescription)")
}

var connections: [ObjectIdentifier: Connection] = [:]
let parameters = NWParameters.tcp
parameters.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(host), port: NWEndpoint.Port(rawValue: port) ?? .any)
let listener: NWListener
do {
    listener = try NWListener(using: parameters)
} catch {
    fail("cannot listen on \(host):\(port): \(error)")
}
listener.newConnectionHandler = { nw in
    let connection = Connection(nw, engine: engine)
    let id = ObjectIdentifier(connection)
    connections[id] = connection
    connection.onClose = { connections[id] = nil }
    connection.start()
}
listener.stateUpdateHandler = { state in
    switch state {
    case .ready:
        printEvent(["event": "ready", "host": host, "port": Int(listener.port?.rawValue ?? 0),
                    "model_name": modelName, "server_pid": Int(getpid())])
    case .failed(let error):
        fail("listener failed: \(error)")
    default:
        break
    }
}
listener.start(queue: queue)

func shutdown() -> Never {
    if supervised { printEvent(["event": "stopped"]) }
    listener.cancel()
    engine.stop()
    exit(0)
}

var signalSources: [DispatchSourceSignal] = []
for sig in [SIGTERM, SIGINT, SIGHUP] {
    signal(sig, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: sig, queue: queue)
    source.setEventHandler { shutdown() }
    source.resume()
    signalSources.append(source)
}

if supervised {
    // Voxt's control channel: a shutdown command, or the end of stdin when Voxt quits.
    Thread.detachNewThread {
        while let line = readLine() {
            if let command = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any],
               command["command"] as? String == "shutdown"
            {
                break
            }
        }
        queue.async { shutdown() }
    }
}

dispatchMain()
