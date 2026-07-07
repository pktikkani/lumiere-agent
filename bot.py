"""Lumiere — voice-controlled video presentation agent.

Pipecat pipeline:

    transport.input()  (SmallWebRTC, Silero VAD for turn-taking / barge-in)
      -> OpenAISTTService (gpt-4o-transcribe, streaming)
      -> context aggregator (user)
      -> AnthropicLLMService (claude-sonnet-5, function calling for video control)
      -> ElevenLabsTTSService (eleven_flash_v2_5, websocket streaming)
      -> transport.output()
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
from pipecat.services.elevenlabs.stt import ElevenLabsSTTService
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
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5")


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


def register_video_tools(llm: AnthropicLLMService, rtvi: RTVIProcessor, catalog, context, base_messages):
    """Wire each LLM tool to a browser command sent as an RTVI server message.

    The official Pipecat JS client receives these via `onServerMessage`.
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

    async def send_command(action: str, video_id: str | None = None) -> None:
        message = {"type": "video_control", "action": action}
        if video_id is not None:
            message["video_id"] = video_id
        logger.info(f"→ browser: {message}")
        await rtvi.send_server_message(message)

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
    """Speech-to-text. Default: ElevenLabs Scribe (accurate on short commands, less
    prone to silence hallucinations than Whisper). Set STT_PROVIDER=openai to use
    Whisper instead. keyterms bias Scribe toward our actual command words + video ids.
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

    logger.info("STT provider: ElevenLabs Scribe")
    return ElevenLabsSTTService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        aiohttp_session=aiohttp_session,
        settings=ElevenLabsSTTService.Settings(keyterms=keyterms),
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

    # Register tools after context exists so handlers can reset it after each command.
    register_video_tools(llm, rtvi, catalog, context, messages)

    # Silent, action-only pipeline: voice -> VAD -> STT -> Claude -> video tool -> browser.
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
            stop_secs=0.6,   # slightly long trailing pause so short commands aren't cut
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
