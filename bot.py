"""Lumiere — voice-controlled video presentation agent.

Pipecat pipeline:

    transport.input()  (SmallWebRTC, Silero VAD for turn-taking / barge-in)
      -> ElevenLabsRealtimeSTTService (scribe_v2_realtime, streaming WebSocket)
      -> context aggregator (user)
      -> AnthropicLLMService (claude-haiku-4-5, function calling for video control)
      -> transport.output()  (no TTS — silent, action-only agent)
      -> context aggregator (assistant)

Video control is done via LLM function calling. Each tool handler pushes a
`video_control` app-message down the same WebRTC data channel the mic uses, and
`client/app.js` applies it to the <video> element. No extra server, no polling.
"""

from __future__ import annotations

import cv2_stub  # noqa: F401  MUST be first: registers a cv2 stub before pipecat imports the real one (see cv2_stub.py)
import anthropic_role_patch  # noqa: F401  fixes a Pipecat 1.5.0 KeyError 'role' that broke follow-up commands

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.elevenlabs.stt import (
    CommitStrategy,
    ElevenLabsRealtimeSTTService,
    ElevenLabsSTTService,
)
from pipecat.services.llm_service import FunctionCallParams, FunctionCallResultProperties
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from videos_catalog import load_catalog

load_dotenv()

# --- Model choices (see VOICE_AGENT_DEMO_PLAN.md). Swap to haiku if latency > 1.5s. ---
# whisper-1 works on all OpenAI projects. gpt-4o-transcribe requires special project
# access (this key's project got a 403 for it). Override via STT_MODEL if yours has it.
STT_MODEL = os.getenv("STT_MODEL", "whisper-1")
# Haiku is plenty for tool-routing and roughly halves the LLM turn latency vs sonnet.
# Override with LLM_MODEL=claude-sonnet-5 if you want the smarter (slower) model.
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5")


def build_system_prompt(catalog) -> str:
    """System prompt with the live video list injected at startup."""
    lines = "\n".join(f'  - {v.video_id}: {v.description}' for v in catalog.videos)
    return (
        "You are Lumiere, a voice-controlled video presentation assistant. "
        "Your ONLY job is to control video playback by calling your tools. "
        "When the presenter asks to start, pause, resume, restart, or switch videos, "
        "immediately call the matching tool. Do NOT reply with text — just call the tool. "
        "Match loose phrasing to the right video from the list below "
        "(e.g. 'show the intro' -> play_video intro, 'the meeting agent demo' -> "
        "play_video meeting-intelligence). If a request is ambiguous, pick the closest video. "
        "Ignore chatter that isn't a playback command.\n\n"
        "Available videos (id: description):\n"
        f"{lines}"
    )


def build_tools(catalog) -> ToolsSchema:
    """The exact, small tool set from the plan. video_id is an enum of real files."""
    video_ids = [v.video_id for v in catalog.videos]
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="play_video",
                description="Play a specific video by its id. Use this to start or switch videos.",
                properties={
                    "video_id": {
                        "type": "string",
                        "enum": video_ids,
                        "description": "Which video to play.",
                    }
                },
                required=["video_id"],
            ),
            FunctionSchema(
                name="pause_video",
                description="Pause the currently playing video.",
                properties={},
                required=[],
            ),
            FunctionSchema(
                name="resume_video",
                description="Resume the currently paused video.",
                properties={},
                required=[],
            ),
            FunctionSchema(
                name="next_video",
                description="Play the next video in the playlist order.",
                properties={},
                required=[],
            ),
            FunctionSchema(
                name="previous_video",
                description="Play the previous video in the playlist order.",
                properties={},
                required=[],
            ),
            FunctionSchema(
                name="restart_video",
                description="Restart the current video from the beginning.",
                properties={},
                required=[],
            ),
        ]
    )


import re

# Words we ignore when matching a spoken phrase to a video (filler, not content).
_STOPWORDS = {
    "the", "a", "an", "to", "please", "video", "now", "this", "that", "one",
    "demo", "lumiere", "hey", "ok", "okay", "can", "you", "up", "me", "my",
}


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def _match_video(text_tokens: list[str], catalog) -> str | None:
    """Pick the video whose id/description shares the most words with the phrase."""
    tset = set(text_tokens)
    best, best_score = None, 0
    for v in catalog.videos:
        kw = (set(_tokens(v.video_id)) | set(_tokens(v.description))) - _STOPWORDS
        score = len(kw & tset)
        if score > best_score:
            best, best_score = v.video_id, score
    return best if best_score > 0 else None


def parse_intent(text: str, catalog):
    """Resolve a spoken command to (action, video_id|None) locally, no LLM.

    Returns None when the phrase is ambiguous (e.g. 'play the ...' with no
    recognizable video) so it falls through to the LLM, which is better at fuzzy
    name matching. Order matters: 'start over' must beat the 'start' play-verb.
    """
    toks = _tokens(text)
    if not toks:
        return None
    tset = set(toks)
    joined = " ".join(toks)

    if tset & {"restart", "replay"} or "over" in tset or "beginning" in tset:
        return ("restart", None)
    if tset & {"next", "skip", "forward"}:
        return ("next", None)
    if tset & {"previous", "prev", "back"}:
        return ("previous", None)
    if tset & {"pause", "hold", "freeze", "stop", "halt", "wait"}:
        return ("pause", None)
    if tset & {"resume", "continue", "unpause", "unfreeze"} or "keep going" in joined or "go on" in joined:
        return ("resume", None)
    if tset & {"play", "show", "start", "open", "switch", "watch", "begin"}:
        vid = _match_video(toks, catalog)
        if vid:
            return ("play", vid)
        if tset & {"play", "start", "begin"}:   # bare "play" = resume current
            return ("resume", None)
        return None  # ambiguous "show/open ..." -> let the LLM figure it out
    return None


class FastCommandRouter(FrameProcessor):
    """Intercept final transcripts and execute simple commands WITHOUT the LLM.

    Sits before the context aggregator. If `parse_intent` resolves the phrase, we
    fire the browser command directly and swallow the frame so the LLM turn is
    skipped entirely (instant, and no context pollution). Anything unresolved
    passes through to Claude as before.
    """

    def __init__(self, catalog, send_command):
        super().__init__()
        self._catalog = catalog
        self._send = send_command

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            intent = parse_intent(frame.text, self._catalog)
            if intent:
                action, video_id = intent
                if action != "play" or video_id in self._catalog.ids:
                    logger.info(f"fast-path: {action} {video_id or ''}".strip())
                    await self._send(action, video_id)
                    return  # swallow — LLM never runs for this command
        await self.push_frame(frame, direction)


def make_send_command(rtvi: RTVIProcessor):
    """Return an async fn that pushes a video_control command to the browser.

    Shared by the LLM tool handlers AND the local fast-path router so both routes
    emit identical RTVI server messages. The official Pipecat JS client receives
    them via `onServerMessage`.
    """

    async def send_command(action: str, video_id: str | None = None) -> None:
        message = {"type": "video_control", "action": action}
        if video_id is not None:
            message["video_id"] = video_id
        logger.info(f"→ browser: {message}")
        await rtvi.send_server_message(message)

    return send_command


def register_video_tools(llm: AnthropicLLMService, catalog, context, base_messages, send_command):
    """Wire each LLM tool to a browser command via the shared `send_command`.

    Only reached for commands the local fast-path can't resolve (mainly fuzzy
    'play <video name>' phrasing).
    """

    # run_llm=False: after the tool runs we do NOT want Claude to generate a spoken
    # reply. This also avoids a Pipecat adapter bug that raises "Unknown error: 'role'"
    # on the post-tool turn, which was corrupting the context and breaking later commands.
    NO_REPLY = FunctionCallResultProperties(run_llm=False)

    def reset_context():
        """Each voice command is independent — this is an action-only agent with no need
        for conversation memory. Resetting to just the system prompt after every command
        keeps latency and cost flat no matter how long the demo runs, and prevents stale
        turns from causing duplicate/confused tool calls."""
        context.set_messages(list(base_messages))

    async def play_video(params: FunctionCallParams):
        video_id = params.arguments.get("video_id")
        title = catalog.title_for(video_id)
        if video_id not in catalog.ids:
            await params.result_callback(
                {"ok": False, "error": f"No video named {video_id!r}. Available: {', '.join(catalog.ids)}"},
                properties=NO_REPLY,
            )
            return
        await send_command("play", video_id)
        await params.result_callback({"ok": True, "now_playing": title}, properties=NO_REPLY)
        reset_context()

    async def pause_video(params: FunctionCallParams):
        await send_command("pause")
        await params.result_callback({"ok": True, "state": "paused"}, properties=NO_REPLY)
        reset_context()

    async def resume_video(params: FunctionCallParams):
        await send_command("resume")
        await params.result_callback({"ok": True, "state": "playing"}, properties=NO_REPLY)
        reset_context()

    async def next_video(params: FunctionCallParams):
        await send_command("next")
        await params.result_callback({"ok": True, "note": "advanced to next video"}, properties=NO_REPLY)
        reset_context()

    async def previous_video(params: FunctionCallParams):
        await send_command("previous")
        await params.result_callback({"ok": True, "note": "went to previous video"}, properties=NO_REPLY)
        reset_context()

    async def restart_video(params: FunctionCallParams):
        await send_command("restart")
        await params.result_callback({"ok": True, "note": "restarted from the beginning"}, properties=NO_REPLY)
        reset_context()

    llm.register_function("play_video", play_video)
    llm.register_function("pause_video", pause_video)
    llm.register_function("resume_video", resume_video)
    llm.register_function("next_video", next_video)
    llm.register_function("previous_video", previous_video)
    llm.register_function("restart_video", restart_video)


class TranscriptLog(FrameProcessor):
    """Logs each final transcript so the demo operator can see what the agent heard."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            logger.info(f"heard: {frame.text!r}")
        await self.push_frame(frame, direction)


def make_stt(catalog, aiohttp_session):
    """Speech-to-text.

    Default (STT_PROVIDER=elevenlabs): ElevenLabs Scribe *Realtime* — a streaming
    WebSocket STT (scribe_v2_realtime). It transcribes WHILE you speak, so at
    turn-end the transcript is already ready — no post-utterance file upload. This
    is what keeps voice commands feeling instant (P99 ~0.4s vs the seconds-long
    round-trip of the file-based API). CommitStrategy.MANUAL lets Pipecat's own VAD
    decide the turn boundary, staying consistent with the rest of the pipeline.

    Fallbacks:
      STT_PROVIDER=elevenlabs-http  -> the old file-based Scribe (accurate but slow).
      STT_PROVIDER=openai           -> Whisper.

    keyterms bias transcription toward our actual command words + video ids.
    """
    provider = os.getenv("STT_PROVIDER", "elevenlabs").lower()
    keyterms = [
        "play", "pause", "resume", "stop", "next", "previous", "back", "restart",
        "intro", "portfolio", "meeting intelligence", "Lumiere",
        *catalog.ids,
    ]
    if provider == "openai":
        prompt = (
            "Voice commands for a video player. Likely phrases: play the intro, "
            "play the portfolio, play the meeting intelligence demo, pause, resume, "
            "next, previous, go back, restart. Video names: " + ", ".join(catalog.ids) + "."
        )
        logger.info("STT provider: OpenAI Whisper")
        return OpenAISTTService(model=STT_MODEL, api_key=os.getenv("OPENAI_API_KEY"), prompt=prompt)

    if provider == "elevenlabs-http":
        logger.info("STT provider: ElevenLabs Scribe (file-based, higher latency)")
        return ElevenLabsSTTService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            aiohttp_session=aiohttp_session,
            settings=ElevenLabsSTTService.Settings(keyterms=keyterms),
        )

    logger.info("STT provider: ElevenLabs Scribe Realtime (streaming)")
    return ElevenLabsRealtimeSTTService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        commit_strategy=CommitStrategy.MANUAL,
        settings=ElevenLabsRealtimeSTTService.Settings(keyterms=keyterms),
    )


def build_pipeline_task(transport: SmallWebRTCTransport, aiohttp_session) -> PipelineTask:
    """Assemble the full Pipecat pipeline for one WebRTC session."""
    catalog = load_catalog()
    if not catalog.videos:
        logger.warning(
            "No .mp4 files found in videos/. The agent will still connect, but has "
            "no videos to control. Drop MP4s into videos/ with descriptive names."
        )

    stt = make_stt(catalog, aiohttp_session)

    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model=LLM_MODEL,
        params=AnthropicLLMService.InputParams(
            max_tokens=200,       # snappy replies; also caps output-token cost per turn
            # NOTE: claude-sonnet-5 rejects `temperature` (deprecated for this model), so we omit it.
            enable_prompt_caching=True,  # cache system prompt + tool schema → ~90% cheaper input tokens after turn 1
        ),
    )

    # RTVIProcessor carries our video_control messages to the browser (the official
    # JS client receives them via onServerMessage). It also emits the RTVI events the
    # client SDK needs to consider the session "ready".
    rtvi = RTVIProcessor(transport=transport)

    # VAD MUST be an explicit pipeline processor in Pipecat 1.5.0 — passing
    # vad_analyzer to TransportParams alone does NOT run turn detection. Without
    # this processor, VAD never fires and STT is never triggered (the bug we hit).
    vad = VADProcessor(vad_analyzer=make_vad_analyzer())

    messages = [{"role": "system", "content": build_system_prompt(catalog)}]
    context = LLMContext(messages=messages, tools=build_tools(catalog))
    aggregators = LLMContextAggregatorPair(context)

    send_command = make_send_command(rtvi)

    # Register tools after context exists so handlers can reset it after each command.
    register_video_tools(llm, catalog, context, messages, send_command)

    # Local fast-path: resolves plain verbs (pause/resume/next/prev/restart) and clear
    # 'play <video>' phrases WITHOUT the LLM, so the video reacts the instant STT lands.
    # Unresolved phrases fall through to Claude. Disable with FAST_PATH=0.
    fast_path = (
        [FastCommandRouter(catalog, send_command)]
        if os.getenv("FAST_PATH", "1") != "0"
        else []
    )

    # Silent, action-only pipeline: voice -> VAD -> STT -> [fast-path] -> Claude -> browser.
    # No TTS, but transport.output() MUST be present: RTVI server messages (our
    # video_control commands) travel DOWNSTREAM and are only delivered to the browser
    # by the output transport. Without it the commands are generated but never sent.
    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            vad,
            stt,
            TranscriptLog(),
            *fast_path,
            aggregators.user(),
            llm,
            aggregators.assistant(),
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,  # barge-in: presenter can talk over the bot
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected — listening for commands (silent, action-only).")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected — cancelling task.")
        await task.cancel()

    return task


def make_vad_analyzer() -> SileroVADAnalyzer:
    """Silero VAD for segmenting the presenter's speech into turns.

    Defaults (min_volume=0.6, confidence=0.7) are too strict for a room mic, so we
    lower them. This analyzer is consumed by the VADProcessor in build_pipeline_task
    (in Pipecat 1.5.0 VAD is a pipeline processor, not a transport auto-feature).
    """
    return SileroVADAnalyzer(
        params=VADParams(
            confidence=0.5,
            start_secs=0.2,
            # Trailing silence before we call end-of-turn. 0.6 felt laggy; 0.35 fires
            # much sooner while still riding out the tiny gaps inside a short command.
            # Pipecat's baseline is 0.2 — drop further if commands still feel delayed,
            # raise a touch if it starts cutting you off mid-sentence.
            stop_secs=0.35,
            min_volume=0.15,
        )
    )


def make_transport(webrtc_connection) -> SmallWebRTCTransport:
    """SmallWebRTC transport: mic in only, no audio out (action-only agent)."""
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=False,  # no TTS / no bot audio — silent, action-only
        ),
    )
    return transport


async def run_bot(webrtc_connection) -> None:
    """Entry point invoked by server.py for each new WebRTC session."""
    import aiohttp

    transport = make_transport(webrtc_connection)
    async with aiohttp.ClientSession() as session:
        task = build_pipeline_task(transport, session)
        runner = PipelineRunner(handle_sigint=False)
        await runner.run(task)
