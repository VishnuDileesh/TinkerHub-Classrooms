import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

import '../models/chat_message.dart';

/// Persists conversations locally, the mobile equivalent of Module 04's
/// `localStorage` calls in app.js. Same idea — save-on-every-message,
/// load-on-launch — a different platform API underneath.
class HistoryStore {
  static const _key = "gemmaikuchat.conversations";

  Future<List<Conversation>> load() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_key);
    if (raw == null) return [];
    final list = jsonDecode(raw) as List;
    return list.map((c) => Conversation.fromJson(c as Map<String, dynamic>)).toList();
  }

  Future<void> save(List<Conversation> conversations) async {
    final prefs = await SharedPreferences.getInstance();
    final raw = jsonEncode(conversations.map((c) => c.toJson()).toList());
    await prefs.setString(_key, raw);
  }
}
