import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/chat_message.dart';

class OllamaModel {
  final String name;
  final int size;
  const OllamaModel({required this.name, required this.size});
}

/// One SSE event parsed off the wire: either a `token` piece of text, a
/// `done` signal, or an `error` message. This mirrors exactly what
/// frontend/static/js/app.js does in Module 04 — same backend, same wire
/// format, a different language reading it.
class ChatEvent {
  final String? token;
  final bool done;
  final String? error;
  const ChatEvent.token(this.token) : done = false, error = null;
  const ChatEvent.done() : token = null, done = true, error = null;
  const ChatEvent.error(this.error) : token = null, done = false;
}

class ApiService {
  final String baseUrl; // e.g. http://10.0.2.2:8000 or your Render/Fly URL
  final http.Client _client = http.Client();

  ApiService(this.baseUrl);

  Future<List<OllamaModel>> listModels() async {
    final res = await _client.get(Uri.parse("$baseUrl/api/models"));
    if (res.statusCode != 200) {
      throw Exception("GET /api/models failed: HTTP ${res.statusCode}");
    }
    final data = jsonDecode(res.body) as Map<String, dynamic>;
    return (data["models"] as List)
        .map((m) => OllamaModel(name: m["name"] as String, size: (m["size"] ?? 0) as int))
        .toList();
  }

  /// Streams the assistant's reply token-by-token. Ollama's chat completion
  /// arrives at our FastAPI backend as newline-delimited JSON, which the
  /// backend re-emits as Server-Sent Events. `http.Client.send` gives us
  /// the raw byte stream; we decode it incrementally and split on the
  /// blank-line SSE frame boundary, exactly like the JS version does with
  /// `ReadableStream` — the underlying problem (network chunks don't
  /// respect your message boundaries) is identical on every platform.
  Stream<ChatEvent> streamChat({
    required String model,
    required List<ChatMessage> messages,
  }) async* {
    final request = http.Request("POST", Uri.parse("$baseUrl/api/chat"))
      ..headers["Content-Type"] = "application/json"
      ..body = jsonEncode({
        "model": model,
        "messages": messages.map((m) => m.toJson()).toList(),
      });

    final streamedResponse = await _client.send(request);
    if (streamedResponse.statusCode != 200) {
      final body = await streamedResponse.stream.bytesToString();
      yield ChatEvent.error("HTTP ${streamedResponse.statusCode}: $body");
      return;
    }

    var buffer = "";
    await for (final chunk in streamedResponse.stream.transform(utf8.decoder)) {
      buffer += chunk;
      final frames = buffer.split("\n\n");
      buffer = frames.removeLast(); // last piece may be incomplete — keep for next chunk

      for (final frame in frames) {
        if (frame.trim().isEmpty) continue;

        String event = "message";
        String data = "";
        for (final line in frame.split("\n")) {
          if (line.startsWith("event:")) {
            event = line.substring(6).trim();
          } else if (line.startsWith("data:")) {
            data = line.substring(5).trim();
          }
        }
        if (data.isEmpty) continue;

        if (event == "done") {
          yield const ChatEvent.done();
          return;
        } else if (event == "error") {
          final parsed = jsonDecode(data) as Map<String, dynamic>;
          yield ChatEvent.error(parsed["error"] as String? ?? "unknown error");
        } else {
          final parsed = jsonDecode(data) as Map<String, dynamic>;
          yield ChatEvent.token(parsed["token"] as String? ?? "");
        }
      }
    }
  }

  void dispose() => _client.close();
}
