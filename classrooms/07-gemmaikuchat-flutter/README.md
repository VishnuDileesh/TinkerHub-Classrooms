# GemmaikuChat, in your pocket — a Flutter mobile client

Every module so far has put GemmaikuChat in a browser tab. [Module 04](../04-gemmaikuchat-fastapi) built the backend contract — `GET /api/models`, `POST /api/chat` streaming Server-Sent Events — and that contract doesn't care what's on the other end of it. This module proves that by building a second, completely independent client: a real Android/iOS app in [Flutter](https://flutter.dev), talking to the exact same FastAPI backend, with zero changes to Module 04's code.

That's the actual lesson here, more than "learn Flutter": once you've built a backend around a clean HTTP contract, *any* client can sit in front of it — a browser, a phone, a CLI, a Slack bot. This module is proof.

## What you'll build

```
07-gemmaikuchat-flutter/
└── app/
    ├── pubspec.yaml
    └── lib/
        ├── main.dart                  # entry point, backend URL config
        ├── models/chat_message.dart   # ChatMessage + Conversation
        ├── services/
        │   ├── api_service.dart       # talks to /api/models and /api/chat
        │   └── history_store.dart     # local persistence (shared_preferences)
        ├── screens/chat_screen.dart   # the whole UI: message list, composer, model picker, history drawer
        └── widgets/message_bubble.dart
```

A single-screen chat app: a drawer for conversation history, a model picker pulled live from your backend, a message list, and a composer — streaming tokens in exactly like the web version did.

## Prerequisites

- **A running GemmaikuChat backend** — either Module 04 running locally (`uvicorn main:app`), or a deployed one from [Module 05](../05-deploying-gemmaikuchat)/[06](../06-gemmaiku-inference-engine). You'll point the app at whichever one you're using.
- **Flutter SDK installed** — `flutter doctor` should run clean (or close to it — a missing Xcode is fine if you're only targeting Android, and vice versa). Install via [flutter.dev/docs/get-started/install](https://flutter.dev/docs/get-started/install).
- **Either** Android Studio (for an Android emulator) **or** Xcode (for an iOS simulator) — you only need one to follow this module, though the code targets both.

## Concepts, before you read the code

**Why Flutter, and what's actually different from the web version?**
Flutter compiles one Dart codebase to native Android and iOS apps (and web, and desktop) — no WebView, no JavaScript bridge, real native widgets under the hood. The interesting constraint this module hits, that Module 04 never had to think about, is: **a mobile app isn't served from the same origin as your backend.** In the browser, `fetch("/api/chat")` "just worked" because the backend served the frontend itself. A phone has no such thing as "the same origin" — every request is fully qualified (`http://10.0.2.2:8000/api/chat`), which means from day one this app has to think about the thing Module 05 only introduced when splitting the frontend onto Vercel: an explicit backend base URL, and — because it's now a *native* app instead of a browser page — a platform-level policy about plaintext HTTP that didn't exist in a browser tab either. More on both below.

**Why a `Stream<ChatEvent>` instead of a callback?**
Dart's `Stream` is the async-sequence-of-values primitive — the same conceptual shape as JavaScript's `ReadableStream` from Module 04's `app.js`, just Dart's native version of it. `api_service.dart` reads raw bytes off the HTTP response with `http.Client.send()`, decodes them, and does the exact same "buffer partial chunks, split on the blank-line SSE boundary" dance the JS frontend does — network chunks don't respect message boundaries on any platform, mobile included. If you've read Module 04's `app.js` closely, `streamChat()` here will look almost line-for-line familiar; that's intentional.

**Why `setState` instead of Provider/Riverpod/Bloc?**
Real Flutter apps usually reach for a state management package once the widget tree grows. This one deliberately doesn't, for the same reason Module 04's frontend skipped React: one screen, one piece of mutable state (`_conversations`, `_active`, `_isStreaming`), and `setState` makes every place that state changes visible in a five-second `Ctrl+F` for `setState(`. Reach for something heavier once you've actually felt the pain of not having it — not before.

## Step-by-step walkthrough

### 1. Turn this into a real Flutter project

The `lib/` and `pubspec.yaml` here are the app's source — but Flutter also needs the generated native scaffolding (`android/`, `ios/`, platform boilerplate) that `flutter create` produces, which isn't checked in since it's regenerable and platform/tooling-version-specific. From `app/`:

```bash
cd classrooms/07-gemmaikuchat-flutter/app
flutter create . --project-name gemmaikuchat_mobile --org com.tinkerhub
```

This scaffolds `android/`, `ios/`, etc. **around** the existing `lib/` and `pubspec.yaml` without touching them. Then fetch dependencies:

```bash
flutter pub get
```

### 2. Point the app at your backend

Open `lib/main.dart` — `kBackendBaseUrl` is the one thing you configure per-run, same role as Module 05's `API_BASE` constant in the Vercel-split frontend:

```dart
const String kBackendBaseUrl = "http://10.0.2.2:8000";
```

- **Android emulator** reaching a backend running on your own laptop: use `10.0.2.2` — the emulator's special alias for "the host machine's localhost." (`localhost` from inside the emulator means the emulator itself, not your laptop — a very common first mistake.)
- **iOS simulator**: `localhost` works as-is, since the simulator shares your Mac's network stack.
- **A physical phone** on the same Wi-Fi as your laptop: use your laptop's LAN IP (`ipconfig getifaddr en0` on macOS), not `localhost` or `10.0.2.2`.
- **A deployed backend** from Module 05/06: just use its real `https://` URL — and skip the cleartext config in step 3 entirely, since HTTPS has no such restriction.

### 3. Allow plaintext HTTP for local development

This is the platform-level thing mentioned above: both Android and iOS block plaintext (`http://`) network requests by default on real network calls — a security default, not a bug — which only matters here because you're hitting a local `http://` backend instead of a deployed `https://` one. Skip this step entirely once you're pointed at a Module 05/06 HTTPS deployment.

**Android** — create `android/app/src/main/res/xml/network_security_config.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <domain-config cleartextTrafficPermitted="true">
        <domain includeSubdomains="true">10.0.2.2</domain>
    </domain-config>
</network-security-config>
```

then reference it from `android/app/src/main/AndroidManifest.xml`, inside the `<application>` tag:

```xml
<application
    android:networkSecurityConfig="@xml/network_security_config"
    ...>
```

**iOS** — add an App Transport Security exception to `ios/Runner/Info.plist`:

```xml
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsLocalNetworking</key>
    <true/>
</dict>
```

### 4. Read the app in the order it actually runs

1. **`lib/models/chat_message.dart`** — `ChatMessage` and `Conversation`, the same two shapes Module 04's `app.js` keeps in `localStorage`, here with explicit `toJson`/`fromJson` because Dart doesn't have JavaScript's implicit object-literal-to-JSON convenience.
2. **`lib/services/api_service.dart`** — `listModels()` is a plain `GET`. `streamChat()` is the one worth reading slowly: it opens a streamed `POST`, decodes bytes as they arrive, and yields `ChatEvent`s (`token`, `done`, `error`) — parsed from the identical `event:`/`data:` SSE frames Module 04's backend already emits. **Nothing about the backend changed to support this client.**
3. **`lib/services/history_store.dart`** — `shared_preferences` is Flutter's simplest persistent key-value store, the mobile analogue of `localStorage`. Same load-on-launch, save-on-every-message pattern as the web version.
4. **`lib/screens/chat_screen.dart`** — the whole UI. `_send()` is the mobile version of `app.js`'s `onSubmit`/`streamAssistantReply`: push the user message, append an empty assistant message, then update that same message's content on every `ChatEvent` as it streams in, calling `setState` each time so the `ListView` redraws with the latest partial text.

### 5. Run it

```bash
# with an Android emulator or iOS simulator already booted:
flutter run
```

Hot reload (`r` in the terminal, or your IDE's hot-reload button) applies most UI edits in under a second without losing app state — the single biggest quality-of-life difference from live-reloading a web page.

## Checkpoint

You know this worked when:
1. The model dropdown populates with whatever your backend's Ollama has pulled — same list `curl http://localhost:8000/api/models` would show you.
2. Sending a message streams the reply in progressively (a small spinner next to the in-progress bubble), not all at once.
3. Backgrounding the app and reopening it still shows your conversation history in the drawer.
4. Turning off Wi-Fi and sending a message shows an in-chat `[error: ...]` message, not a frozen UI or a crash.

## Common pitfalls

- **`Connection refused` on Android** — almost always `localhost` instead of `10.0.2.2` in `kBackendBaseUrl`. This is the single most common Flutter-networking mistake and it's worth internalizing why, not just fixing it once.
- **`CLEARTEXT communication not permitted`** — step 3's Android config is missing, or the domain listed doesn't match the host you're actually hitting (it must match `kBackendBaseUrl`'s host exactly).
- **CORS-looking errors that aren't CORS** — CORS is a *browser* enforcement mechanism; native mobile HTTP clients don't have it. If a request fails from the app, the cause is elsewhere (network reachability, the cleartext policy above, or the backend actually being down) — don't waste time hunting for a CORS header that was never relevant here.
- **Physical device can't reach your laptop** — confirm both are on the same Wi-Fi network (not one on Wi-Fi and one on cellular), and that your laptop's firewall isn't blocking inbound connections on the backend's port.

## Things to try next

- **Point it at your Module 06 Fly.io or Hugging Face deployment** instead of a local backend — no code changes needed beyond `kBackendBaseUrl`, which is exactly the point of having built a clean HTTP contract in Module 04.
- **Build for web too**: `flutter build web` compiles this same codebase to a static site — a genuinely different mechanism from Module 04's hand-written HTML/CSS/JS frontend, worth comparing side by side now that you've built both.
- **Add push notifications** for when a long generation finishes while the app is backgrounded.
- **Swap `shared_preferences` for `sqflite`** once conversation history outgrows a single JSON blob, and notice exactly where that migration touches the code (only `history_store.dart` — nothing in `chat_screen.dart` should need to change if the interface stays the same).
