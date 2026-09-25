import 'package:flutter/material.dart';

import '../models/chat_message.dart';
import '../services/api_service.dart';
import '../services/history_store.dart';
import '../widgets/message_bubble.dart';

class ChatScreen extends StatefulWidget {
  final String baseUrl;

  const ChatScreen({super.key, required this.baseUrl});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  late final ApiService _api;
  final HistoryStore _store = HistoryStore();
  final TextEditingController _inputController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  List<Conversation> _conversations = [];
  Conversation? _active;
  List<OllamaModel> _models = [];
  String? _selectedModel;
  String? _modelError;
  bool _isStreaming = false;

  @override
  void initState() {
    super.initState();
    _api = ApiService(widget.baseUrl);
    _loadHistory();
    _loadModels();
  }

  @override
  void dispose() {
    _api.dispose();
    _inputController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _loadHistory() async {
    final loaded = await _store.load();
    setState(() {
      _conversations = loaded;
      _active = loaded.isNotEmpty ? loaded.first : null;
    });
  }

  Future<void> _loadModels() async {
    try {
      final models = await _api.listModels();
      setState(() {
        _models = models;
        final gemmaiku = models.where((m) => m.name.toLowerCase().startsWith("gemmaiku"));
        _selectedModel = gemmaiku.isNotEmpty ? gemmaiku.first.name : models.firstOrNull?.name;
        _modelError = models.isEmpty ? "No models pulled yet on the backend." : null;
      });
    } catch (e) {
      setState(() => _modelError = "Could not reach $baseUrlLabel — is the backend running?");
    }
  }

  String get baseUrlLabel => widget.baseUrl;

  void _startNewConversation() {
    setState(() => _active = null);
  }

  Future<void> _send() async {
    final text = _inputController.text.trim();
    if (text.isEmpty || _isStreaming || _selectedModel == null) return;

    setState(() {
      _active ??= Conversation(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        title: text.length > 40 ? text.substring(0, 40) : text,
      );
      if (!_conversations.contains(_active)) {
        _conversations.insert(0, _active!);
      }
      _active!.messages.add(ChatMessage(role: "user", content: text));
      _isStreaming = true;
    });
    _inputController.clear();
    await _store.save(_conversations);
    _scrollToBottom();

    final assistantMessage = ChatMessage(role: "assistant", content: "");
    setState(() => _active!.messages.add(assistantMessage));

    var fullText = "";
    await for (final event in _api.streamChat(model: _selectedModel!, messages: _active!.messages)) {
      if (event.error != null) {
        fullText += "\n\n[error: ${event.error}]";
      } else if (event.done) {
        break;
      } else {
        fullText += event.token ?? "";
      }
      setState(() {
        _active!.messages[_active!.messages.length - 1] =
            assistantMessage.copyWith(content: fullText);
      });
      _scrollToBottom();
    }

    setState(() => _isStreaming = false);
    await _store.save(_conversations);
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final messages = _active?.messages ?? [];

    return Scaffold(
      appBar: AppBar(
        title: const Text("GemmaikuChat"),
        actions: [
          IconButton(
            icon: const Icon(Icons.add_comment_outlined),
            tooltip: "New chat",
            onPressed: _startNewConversation,
          ),
        ],
      ),
      drawer: Drawer(
        child: SafeArea(
          child: ListView(
            children: [
              const Padding(
                padding: EdgeInsets.all(16),
                child: Text("History", style: TextStyle(fontWeight: FontWeight.bold)),
              ),
              for (final convo in _conversations)
                ListTile(
                  title: Text(convo.title, maxLines: 1, overflow: TextOverflow.ellipsis),
                  selected: convo.id == _active?.id,
                  onTap: () {
                    setState(() => _active = convo);
                    Navigator.pop(context);
                  },
                ),
            ],
          ),
        ),
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: Row(
              children: [
                const Text("Model:"),
                const SizedBox(width: 8),
                Expanded(
                  child: DropdownButton<String>(
                    isExpanded: true,
                    value: _selectedModel,
                    hint: Text(_modelError ?? "Loading models…"),
                    items: _models
                        .map((m) => DropdownMenuItem(value: m.name, child: Text(m.name)))
                        .toList(),
                    onChanged: (v) => setState(() => _selectedModel = v),
                  ),
                ),
              ],
            ),
          ),
          if (_modelError != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              child: Text(_modelError!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
            ),
          Expanded(
            child: messages.isEmpty
                ? const Center(
                    child: Padding(
                      padding: EdgeInsets.all(24),
                      child: Text(
                        "Talk to whatever model you've pulled into Ollama.\n"
                        "Try gemmaiku for a model that only answers in haiku.",
                        textAlign: TextAlign.center,
                      ),
                    ),
                  )
                : ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    itemCount: messages.length,
                    itemBuilder: (context, i) {
                      final isLast = i == messages.length - 1;
                      return MessageBubble(
                        message: messages[i],
                        isStreaming: isLast && _isStreaming && messages[i].role == "assistant",
                      );
                    },
                  ),
          ),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _inputController,
                      minLines: 1,
                      maxLines: 5,
                      textInputAction: TextInputAction.send,
                      onSubmitted: (_) => _send(),
                      decoration: const InputDecoration(
                        hintText: "Message GemmaikuChat...",
                        border: OutlineInputBorder(borderRadius: BorderRadius.all(Radius.circular(20))),
                        contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  IconButton.filled(
                    icon: const Icon(Icons.arrow_upward),
                    onPressed: _isStreaming ? null : _send,
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

extension _FirstOrNull<T> on Iterable<T> {
  T? get firstOrNull => isEmpty ? null : first;
}
