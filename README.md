# Lumiere — voice-controlled video presentation agent

Talk to **Lumiere** and it plays, pauses, and switches local MP4s by voice. Say
*"play the intro"*, *"pause"*, *"play the meeting intelligence demo"*, *"next"* —
the video reacts. Built for live presentations where the presenter drives the deck
hands-free.

```
Browser mic ─► Silero VAD ─► ElevenLabs Scribe (STT) ─► Claude (tools) ─► video_control ─► <video>
                                                                              (over the WebRTC data channel)
```

Built on [Pipecat](https://github.com/pipecat-ai/pipecat) with `SmallWebRTCTransport`
(browser mic ↔ Python, peer-to-peer on localhost) and the official
`@pipecat-ai/client-js` browser SDK. The Mac only runs Python orchestration + the
browser; STT and reasoning are cloud APIs. **Internet reliability matters, not RAM.**

---

## 1. Prerequisites

- **Python 3.12** (pinned in `.python-version`; [`uv`](https://docs.astral.sh/uv/) fetches it automatically)
- [`uv`](https://docs.astral.sh/uv/) for packaging — `brew install uv`
- A modern browser (**Chrome or Arc recommended**; Safari works but is stricter)

## 2. Install

```bash
uv sync          # installs everything from the committed uv.lock (deterministic)
```

## 3. Add your API keys

Copy the template and fill it in:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=sk-...          # (optional) only if you switch STT back to Whisper
ANTHROPIC_API_KEY=sk-ant-...   # the Claude brain (function calling)
ELEVENLABS_API_KEY=...         # ElevenLabs Scribe speech-to-text
ELEVENLABS_VOICE_ID=...        # not used in the action-only build, but harmless to set
```

> **Keys:** OpenAI → platform.openai.com/api-keys · Anthropic → console.anthropic.com ·
> ElevenLabs → elevenlabs.io (profile → API key).

## 4. Add your videos

Drop `.mp4` files into **`videos/`**. The **filename becomes the name Lumiere understands**:

```
videos/
├── intro.mp4                 → "play the intro"
├── portfolio.mp4             → "play the portfolio"
└── meeting-intelligence.mp4  → "play the meeting intelligence demo"
```

Optional: edit `videos/descriptions.json` to give Lumiere a one-line description of
each video (helps it map loose phrasing). Videos are git-ignored.

## 5. Run

```bash
uv run python server.py
```

Open **http://localhost:7860**, click **Connect mic**, allow the microphone, then talk:

- *"Lumiere, play the intro"*
- *"pause"* / *"resume"*
- *"play the meeting intelligence demo"*
- *"next"* / *"go back"* / *"restart"*

The first video plays **with sound, no clicking** — connecting the mic unlocks audio
for the session. The intro does **not** auto-play on connect.

---

## Voice commands

| Say something like… | Tool | Effect |
|---|---|---|
| "play the intro", "start the demo" | `play_video(video_id)` | loads & plays that MP4 |
| "pause" | `pause_video` | pauses |
| "resume", "keep going" | `resume_video` | resumes |
| "next one" | `next_video` | next in playlist |
| "go back" | `previous_video` | previous in playlist |
| "start over", "restart" | `restart_video` | back to 0:00 |

## How it works

| File | Purpose |
|---|---|
| `server.py` | FastAPI: serves the client + WebRTC signaling (`/api/offer`) |
| `bot.py` | Pipecat pipeline (VAD → STT → Claude tools) + the 6 video-control tools |
| `videos_catalog.py` | discovers `videos/*.mp4`, builds the LLM's video enum |
| `client/` | `index.html`, `app.js` (official Pipecat SDK), `style.css` — mic + `<video>` UI |
| `cv2_stub.py` | stubs OpenCV so it doesn't clash with PyAV's ffmpeg (audio decode) |
| `anthropic_role_patch.py` | patches a Pipecat 1.5.0 adapter crash on tool-call context |

**Action-only:** the agent controls video via Claude **function calling** and never
speaks back (no TTS). Each command resets the LLM context, so latency and cost stay
flat no matter how long the session runs. Prompt caching + `max_tokens=200` keep it
cheap and snappy.

## Configuration (env vars)

| Var | Default | Notes |
|---|---|---|
| `STT_PROVIDER` | `elevenlabs` | set to `openai` to use Whisper instead of ElevenLabs Scribe |
| `LLM_MODEL` | `claude-sonnet-5` | swap to `claude-haiku-4-5-20251001` for lower latency/cost |
| `STT_MODEL` | `whisper-1` | only used when `STT_PROVIDER=openai` |
| `HOST` / `PORT` | `localhost` / `7860` | server bind address |

## Demo-day checklist

- **Network is the #1 risk** — STT + reasoning are cloud APIs. Bring a phone hotspot; test on it.
- Check ElevenLabs + Anthropic accounts have credit.
- **Use a good mic.** A near-silent input (some Bluetooth mics) breaks recognition —
  the built-in / Studio Display mic works great. Watch the input level in
  System Settings → Sound → Input; it must move when you talk.
- Use **Chrome or Arc**. Pre-warm: run one conversation before walking in so DNS/connections are hot.
- Do Not Disturb on; close notifications.
- **Record a backup video of a good run.** Non-negotiable.

## Troubleshooting

- **Nothing happens when I talk** → check the mic input level (must be well above silence).
  The server logs `heard: '...'` for every transcript — if nothing appears, it's the mic.
- **Video plays but no sound** → click the video once to unmute (fallback). Normally the
  Connect-mic click unlocks audio for the whole session.
- **Wrong video / no action** → the server log shows `→ browser: {...}` when a command
  fires and `heard: '...'` for what STT understood.
