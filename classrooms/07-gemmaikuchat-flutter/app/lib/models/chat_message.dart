/// A single turn in a conversation. Mirrors the {role, content} shape the
/// FastAPI backend from Module 04 expects and returns — the same contract,
/// just consumed by a different client.
class ChatMessage {
  final String role; // "user" | "assistant"
  final String content;

  const ChatMessage({required this.role, required this.content});

  Map<String, dynamic> toJson() => {"role": role, "content": content};

  factory ChatMessage.fromJson(Map<String, dynamic> json) => ChatMessage(
        role: json["role"] as String,
        content: json["content"] as String,
      );

  ChatMessage copyWith({String? content}) =>
      ChatMessage(role: role, content: content ?? this.content);
}

class Conversation {
  final String id;
  String title;
  final List<ChatMessage> messages;

  Conversation({required this.id, required this.title, List<ChatMessage>? messages})
      : messages = messages ?? [];

  Map<String, dynamic> toJson() => {
        "id": id,
        "title": title,
        "messages": messages.map((m) => m.toJson()).toList(),
      };

  factory Conversation.fromJson(Map<String, dynamic> json) => Conversation(
        id: json["id"] as String,
        title: json["title"] as String,
        messages: (json["messages"] as List)
            .map((m) => ChatMessage.fromJson(m as Map<String, dynamic>))
            .toList(),
      );
}
