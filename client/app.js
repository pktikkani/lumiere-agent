// Lumiere client — uses Pipecat's OFFICIAL browser SDK for the mic/WebRTC.
//
// Why the SDK (and not hand-rolled WebRTC): SmallWebRTCTransport requires a specific
// transceiver order + sendrecv negotiation for the mic. Hand-rolling it makes the
// server never receive audio frames ("No audio frame received"). The SDK does it right.
//
// The bot sends video commands as RTVI *server messages*; we get them via onServerMessage:
//   { type: "video_control", action: "play", video_id: "intro" }

import { PipecatClient, RTVIEvent } from "https://esm.sh/@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "https://esm.sh/@pipecat-ai/small-webrtc-transport";

const els = {
  connectBtn: document.getElementById("connectBtn"),
  status: document.getElementById("status"),
  statusDot: document.getElementById("statusDot"),
  video: document.getElementById("video"),
  nowPlaying: document.getElementById("nowPlaying"),
  playlist: document.getElementById("playlist"),
  log: document.getElementById("log"),
};

let client = null;
let playlist = [];
let currentIndex = -1;
let videoPrimed = false;   // true once the <video> element is blessed for sound (on Connect)

function log(msg, cls = "") {
  const line = document.createElement("div");
  if (cls) line.className = cls;
  line.textContent = msg;
  els.log.prepend(line);
}
function setStatus(t) { els.status.textContent = t; }
function setDot(cls) { els.statusDot.className = "dot" + (cls ? " " + cls : ""); }

// ---------- playlist ----------

async function loadPlaylist() {
  try {
    const res = await fetch("/api/videos");
    playlist = (await res.json()).videos || [];
  } catch { playlist = []; }
  renderPlaylist();
  if (!playlist.length) log("No videos found. Drop .mp4 files into videos/ and reload.", "err");
}
function renderPlaylist() {
  els.playlist.innerHTML = "";
  playlist.forEach((v, i) => {
    const li = document.createElement("li");
    if (i === currentIndex) li.classList.add("active");
    li.innerHTML = `<div class="id">${v.id}</div><div class="desc">${v.description}</div>`;
    els.playlist.appendChild(li);
  });
}
const indexOfId = (id) => playlist.findIndex((v) => v.id === id);

// ---------- video control (from bot commands) ----------

async function playByIndex(i) {
  if (i < 0 || i >= playlist.length) { log(`No video at position ${i}`, "err"); return; }
  currentIndex = i;
  const v = playlist[i];
  const wantSrc = `/videos/${encodeURIComponent(v.filename)}`;
  els.video.style.opacity = "1";    // reveal (hidden during connect-time priming)
  els.video.muted = false;          // WITH sound (element was blessed on Connect)
  els.video.volume = 1.0;

  // Only (re)assign src if it's actually a different video. Re-assigning the same
  // src triggers an implicit load() that interrupts play() ("interrupted by a new
  // load request"). For the primed intro this avoids the race entirely.
  const curSrc = els.video.currentSrc || els.video.src;
  if (!curSrc.endsWith(wantSrc)) {
    els.video.src = wantSrc;
    try { await new Promise((res) => { els.video.oncanplay = res; els.video.load(); setTimeout(res, 800); }); } catch {}
  } else {
    els.video.currentTime = 0;
  }

  try {
    await els.video.play();
  } catch (e) {
    log("audio blocked (" + e.message + ") — playing muted; click video to unmute", "err");
    els.video.muted = true;
    els.video.play().catch(() => {});
  }
  els.nowPlaying.textContent = `▶ ${v.id} — ${v.description}`;
  renderPlaylist();
}

// The RTVI server message may arrive as the payload directly, or nested under
// .data / .d depending on SDK build. Dig out the video_control object from any shape.
function handleServerMessage(source, raw) {
  log(`${source}: ${JSON.stringify(raw)}`);
  const candidates = [raw, raw?.data, raw?.data?.data, raw?.d, raw?.data?.d];
  for (const c of candidates) {
    if (c && c.type === "video_control") { applyCommand(c); return; }
  }
  log(`(no video_control found in ${source} payload)`, "err");
}

function applyCommand(cmd) {
  log(`cmd: ${JSON.stringify(cmd)}`, "cmd");
  switch (cmd.action) {
    case "play": playByIndex(cmd.video_id ? indexOfId(cmd.video_id) : Math.max(currentIndex, 0)); break;
    case "pause": els.video.pause(); els.nowPlaying.textContent = els.nowPlaying.textContent.replace("▶", "⏸"); break;
    case "resume": els.video.play().catch(() => {}); els.nowPlaying.textContent = els.nowPlaying.textContent.replace("⏸", "▶"); break;
    case "next": playByIndex(currentIndex + 1); break;
    case "previous": playByIndex(currentIndex - 1); break;
    case "restart": els.video.currentTime = 0; els.video.play().catch(() => {}); break;
    default: log("unknown action: " + cmd.action, "err");
  }
}

// ---------- connect via Pipecat SDK ----------

async function connect() {
  els.connectBtn.disabled = true;
  setStatus("connecting…");

  // Bless the <video> element for sound inside this click (a user gesture): play it
  // muted for a beat, then pause. This authorizes later voice-triggered playback to
  // run WITH sound. We keep it hidden (opacity 0) until a real command so the intro's
  // first frame doesn't show on connect. We do NOT call load()/removeAttribute — that
  // cancels the in-flight play and breaks the next real play() request.
  try {
    if (playlist[0]) {
      els.video.style.opacity = "0";
      els.video.src = `/videos/${encodeURIComponent(playlist[0].filename)}`;
      els.video.muted = true;
      await els.video.play();
      els.video.pause();
      els.video.currentTime = 0;
      els.video.muted = false;   // ready to play WITH sound on the first real command
      videoPrimed = true;
      log("audio unlocked");
    }
  } catch (e) { log("audio unlock skipped: " + e.message); }

  client = new PipecatClient({
    transport: new SmallWebRTCTransport({ webrtcUrl: "/api/offer" }),
    enableMic: true,
    enableCam: false,
    callbacks: {
      onBotReady: () => { log("bot ready — say 'play the intro'"); setStatus("ready"); setDot("connected"); },
      onServerMessage: (data) => handleServerMessage("onServerMessage", data),
      onUserStartedSpeaking: () => setDot("speaking"),
      onUserStoppedSpeaking: () => setDot("connected"),
      onError: (e) => log("error: " + (e?.message || JSON.stringify(e)), "err"),
    },
  });

  client.on(RTVIEvent.Connected, () => { log("connected"); setDot("connected"); });
  client.on(RTVIEvent.Disconnected, () => { log("disconnected"); setDot(""); els.connectBtn.disabled = false; els.connectBtn.textContent = "Connect mic"; });
  // Also listen via the event bus in case the callback shape differs across SDK builds.
  try { client.on(RTVIEvent.ServerMessage, (data) => handleServerMessage("RTVIEvent.ServerMessage", data)); } catch {}
  client.on(RTVIEvent.UserTranscript, (t) => { if (t?.final) log("you: " + t.text); });

  try {
    await client.connect();
    try { client.enableMic(true); } catch {}  // guarantee mic is live
    els.connectBtn.textContent = "Connected";
  } catch (e) {
    log("connect failed: " + e.message, "err");
    setStatus("failed");
    els.connectBtn.disabled = false;
  }
}

// Manual fallback: click the video to unmute it if it ever plays silently.
els.video.addEventListener("click", () => {
  if (els.video.muted) { els.video.muted = false; els.video.volume = 1.0; log("unmuted"); }
});

els.connectBtn.addEventListener("click", connect);
loadPlaylist();
