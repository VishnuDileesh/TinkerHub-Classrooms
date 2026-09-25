"""
GemmaikuChat backend.

A thin FastAPI layer between a browser and a local Ollama server. It does
three things, on purpose, and nothing more:

1. GET  /api/models  -> ask Ollama what models are installed, so the
                         frontend can populate a dropdown.
2. POST /api/chat    -> forward a conversation to Ollama's /api/chat
                         endpoint and stream the response back to the
                         browser token-by-token over Server-Sent Events.
3. Serve the static frontend (frontend/index.html + static/) so the whole
   app is just `uvicorn main:app` and a browser tab.

No database, no auth, no framework-provided chat widget. The point of this
module is that you can read every line of it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

OLLAMA_HOST = "http://localhost:11434"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="GemmaikuChat")


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]


@app.get("/api/models")
async def list_models() -> dict:
    """Return the models Ollama currently has pulled, so the UI can list them."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=10.0)
            resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Could not reach Ollama at "
                    f"{OLLAMA_HOST}. Is `ollama serve` running?"
                ),
            ) from exc
    data = resp.json()
    models = [
        {"name": m["name"], "size": m.get("size", 0)}
        for m in data.get("models", [])
    ]
    return {"models": models}


async def _stream_ollama_chat(model: str, messages: list[dict]) -> AsyncIterator[bytes]:
    """
    Open a streaming POST to Ollama's /api/chat and re-yield each token as
    a Server-Sent Event. Ollama streams newline-delimited JSON objects like:

        {"message": {"role": "assistant", "content": "Pa"}, "done": false}
        {"message": {"role": "assistant", "content": "ris"}, "done": false}
        ...
        {"done": true, ...stats}

    We just forward the "content" piece of each line as an SSE `data:` frame,
    and send a final `event: done` frame so the frontend knows to stop the
    typing indicator.
    """
    payload = {"model": model, "messages": messages, "stream": True}

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream(
                "POST", f"{OLLAMA_HOST}/api/chat", json=payload
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    error = body.decode(errors="replace")
                    yield f"event: error\ndata: {json.dumps({'error': error})}\n\n".encode()
                    return

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("done"):
                        yield b"event: done\ndata: {}\n\n"
                        return
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n".encode()
        except httpx.ConnectError:
            yield (
                "event: error\ndata: "
                + json.dumps({"error": f"Could not reach Ollama at {OLLAMA_HOST}."})
                + "\n\n"
            ).encode()


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    messages = [m.model_dump() for m in req.messages]
    return StreamingResponse(
        _stream_ollama_chat(req.model, messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering, keep tokens flowing
        },
    )


# Serve the frontend last, so it doesn't shadow the /api/* routes above.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
