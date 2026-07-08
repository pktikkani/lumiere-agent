"""Local demo server for Lumiere.

Serves:
  - GET  /                the video-player + mic client (client/index.html)
  - /client/*, /videos/*  static assets and the demo MP4s
  - POST /api/offer       WebRTC offer -> spawns a bot session on the connection
  - PATCH /api/offer      trickle ICE candidates

Everything runs on http://localhost:7860 — one process, no external services
besides the cloud STT/LLM/TTS APIs the bot itself calls.
"""

from __future__ import annotations

import cv2_stub  # noqa: F401  MUST be first: registers a cv2 stub before pipecat imports the real one (see cv2_stub.py)

import os

from log_config import setup_logging

setup_logging()  # route all logs to files, keep console quiet — BEFORE importing bot

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from bot import run_bot

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
CLIENT_DIR = os.path.join(HERE, "client")
VIDEOS_DIR = os.path.join(HERE, "videos")

REQUIRED_KEYS = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]

app = FastAPI(title="Lumiere Voice Video Agent")
os.makedirs(VIDEOS_DIR, exist_ok=True)
app.mount("/client", StaticFiles(directory=CLIENT_DIR), name="client")
app.mount("/videos", StaticFiles(directory=VIDEOS_DIR), name="videos")

_handler = SmallWebRTCRequestHandler()


@app.on_event("startup")
async def _warn_missing_keys():
    missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
    if missing:
        logger.warning(f"Missing env keys (bot will fail to connect): {', '.join(missing)}")
    else:
        logger.info("All API keys present.")


@app.get("/")
async def index():
    return FileResponse(os.path.join(CLIENT_DIR, "index.html"))


@app.get("/api/videos")
async def api_videos():
    """The client fetches this to render the playlist UI. Same source as the LLM enum."""
    from videos_catalog import load_catalog

    catalog = load_catalog()
    return {
        "videos": [
            {"id": v.video_id, "filename": v.filename, "description": v.description}
            for v in catalog.videos
        ]
    }


@app.post("/api/log")
async def api_log(entry: dict):
    """Browser forwards its logs here so they land in the same file (UI stays clean)."""
    msg = str(entry.get("msg", ""))
    level = str(entry.get("level", "info")).upper()
    logger.log("ERROR" if level == "ERR" else "DEBUG", f"[browser] {msg}")
    return {"ok": True}


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    webrtc_request = SmallWebRTCRequest.from_dict(request)

    async def on_connection(connection: SmallWebRTCConnection):
        logger.info(f"New WebRTC connection {connection.pc_id} — starting Lumiere bot.")
        background_tasks.add_task(run_bot, connection)

    answer = await _handler.handle_web_request(
        request=webrtc_request,
        webrtc_connection_callback=on_connection,
    )
    return answer


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    await _handler.handle_patch_request(request)
    return {"status": "success"}


def main():
    import uvicorn

    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", "7860"))
    logger.info(f"Lumiere server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
