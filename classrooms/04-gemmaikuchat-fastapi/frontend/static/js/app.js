// GemmaikuChat frontend. No framework, no build step — just fetch(),
// SSE parsing, and DOM updates. Everything here is meant to be readable
// top to bottom.

const el = {
  messages: document.getElementById("messages"),
  emptyState: document.getElementById("empty-state"),
  form: document.getElementById("chat-form"),
  input: document.getElementById("chat-input"),
  sendBtn: document.getElementById("send-btn"),
  modelSelect: document.getElementById("model-select"),
  modelStatus: document.getElementById("model-status"),
  newChatBtn: document.getElementById("new-chat-btn"),
  historyList: document.getElementById("history-list"),
};

const STORAGE_KEY = "gemmaikuchat.conversations";

/** @type {{id: string, title: string, messages: {role: string, content: string}[]}[]} */
let conversations = loadConversations();
let activeId = conversations[0]?.id ?? null;
let isStreaming = false;

init();

function init() {
  loadModels();
  renderHistory();
  renderActiveConversation();

  el.form.addEventListener("submit", onSubmit);
  el.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      el.form.requestSubmit();
    }
  });
  el.input.addEventListener("input", autoGrow);
  el.newChatBtn.addEventListener("click", startNewConversation);
}

function autoGrow() {
  el.input.style.height = "auto";
  el.input.style.height = Math.min(el.input.scrollHeight, 200) + "px";
}

// ---------- Models ----------

async function loadModels() {
  try {
    const res = await fetch("/api/models");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    el.modelSelect.innerHTML = "";
    if (data.models.length === 0) {
      el.modelStatus.textContent = "No models pulled yet. Try: ollama pull gemma3:1b";
      el.modelStatus.classList.add("error");
      return;
    }
    for (const m of data.models) {
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = m.name;
      el.modelSelect.appendChild(opt);
    }
    // Prefer a model literally named/prefixed "gemmaiku" if it's installed.
    const gemmaiku = data.models.find((m) => m.name.toLowerCase().startsWith("gemmaiku"));
    if (gemmaiku) el.modelSelect.value = gemmaiku.name;

    el.modelStatus.textContent = `${data.models.length} model(s) available`;
    el.modelStatus.classList.remove("error");
  } catch (err) {
    el.modelStatus.textContent = "Could not reach Ollama. Is `ollama serve` running?";
    el.modelStatus.classList.add("error");
  }
}

// ---------- Conversation state ----------

function loadConversations() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveConversations() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
  } catch {
    // localStorage full or unavailable — conversation just won't persist across reloads.
  }
}

function getActiveConversation() {
  return conversations.find((c) => c.id === activeId) ?? null;
}

function startNewConversation() {
  activeId = null;
  renderActiveConversation();
  renderHistory();
  el.input.focus();
}

function renderHistory() {
  el.historyList.innerHTML = "";
  for (const convo of conversations) {
    const li = document.createElement("li");
    li.textContent = convo.title || "New chat";
    li.className = convo.id === activeId ? "active" : "";
    li.addEventListener("click", () => {
      activeId = convo.id;
      renderActiveConversation();
      renderHistory();
    });
    el.historyList.appendChild(li);
  }
}

function renderActiveConversation() {
  el.messages.innerHTML = "";
  const convo = getActiveConversation();
  if (!convo || convo.messages.length === 0) {
    el.messages.appendChild(el.emptyState);
    return;
  }
  for (const msg of convo.messages) {
    appendBubble(msg.role, msg.content);
  }
  scrollToBottom();
}

// ---------- Sending messages ----------

async function onSubmit(e) {
  e.preventDefault();
  const text = el.input.value.trim();
  if (!text || isStreaming) return;

  let convo = getActiveConversation();
  if (!convo) {
    convo = {
      id: crypto.randomUUID(),
      title: text.slice(0, 40),
      messages: [],
    };
    conversations.unshift(convo);
    activeId = convo.id;
    renderHistory();
  }

  if (el.messages.contains(el.emptyState)) {
    el.messages.removeChild(el.emptyState);
  }

  convo.messages.push({ role: "user", content: text });
  appendBubble("user", text);
  saveConversations();

  el.input.value = "";
  autoGrow();

  const model = el.modelSelect.value;
  if (!model) {
    appendBubble("assistant", "No model selected. Pull one with `ollama pull <model>` first.", true);
    return;
  }

  await streamAssistantReply(convo, model);
}

async function streamAssistantReply(convo, model) {
  isStreaming = true;
  el.sendBtn.disabled = true;

  const bubble = appendBubble("assistant", "");
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  bubble.appendChild(cursor);

  let fullText = "";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: convo.messages.map(({ role, content }) => ({ role, content })),
      }),
    });

    if (!res.ok || !res.body) {
      throw new Error(`HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop(); // last chunk may be incomplete, keep it for next read

      for (const frame of frames) {
        const { event, data } = parseSseFrame(frame);
        if (!data) continue;

        if (event === "error") {
          const parsed = JSON.parse(data);
          fullText += `\n\n[error: ${parsed.error}]`;
          bubble.classList.add("error-bubble");
        } else if (event === "done") {
          // stream finished cleanly
        } else {
          const parsed = JSON.parse(data);
          fullText += parsed.token ?? "";
        }
        bubble.textContent = fullText;
        bubble.appendChild(cursor);
        scrollToBottom();
      }
    }
  } catch (err) {
    fullText += `\n\n[connection error: ${err.message}]`;
    bubble.textContent = fullText;
    bubble.classList.add("error-bubble");
  } finally {
    cursor.remove();
    isStreaming = false;
    el.sendBtn.disabled = false;
    convo.messages.push({ role: "assistant", content: fullText });
    saveConversations();
  }
}

/** Parse one `event: x\ndata: y` SSE frame into { event, data }. */
function parseSseFrame(frame) {
  let event = "message";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data = line.slice(5).trim();
  }
  return { event, data };
}

// ---------- DOM helpers ----------

function appendBubble(role, text, isError = false) {
  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble" + (isError ? " error-bubble" : "");
  bubble.textContent = text;

  row.appendChild(bubble);
  el.messages.appendChild(row);
  scrollToBottom();
  return bubble;
}

function scrollToBottom() {
  el.messages.scrollTop = el.messages.scrollHeight;
}
