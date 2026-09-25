import 'package:flutter/material.dart';

import 'screens/chat_screen.dart';

/// Point this at your Module 04 backend.
///
/// - Android emulator reaching your laptop's localhost: http://10.0.2.2:8000
/// - iOS simulator reaching your laptop's localhost:    http://localhost:8000
/// - A physical device on the same Wi-Fi:                http://<your-laptop-lan-ip>:8000
/// - Your Module 05/06 deployment:                       https://your-backend.onrender.com
const String kBackendBaseUrl = "http://10.0.2.2:8000";

void main() {
  runApp(const GemmaikuChatApp());
}

class GemmaikuChatApp extends StatelessWidget {
  const GemmaikuChatApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "GemmaikuChat",
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF6C8CFF), brightness: Brightness.dark),
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF6C8CFF), brightness: Brightness.dark),
        useMaterial3: true,
      ),
      home: const ChatScreen(baseUrl: kBackendBaseUrl),
    );
  }
}
