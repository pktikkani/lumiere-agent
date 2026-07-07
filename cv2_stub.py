"""Stub that stands in for OpenCV (cv2) so we never load its bundled ffmpeg dylibs.

WHY THIS EXISTS
---------------
On Apple Silicon, `opencv-python` and `av` (PyAV, used by aiortc to decode the
browser's Opus mic audio) each bundle their OWN copy of ffmpeg's libav* dylibs,
at *different* major versions (cv2 → 61.x, av → 62.x). When both load into one
process, macOS logs:

    Class AVFAudioReceiver is implemented in both .../cv2/.dylibs/libavdevice...
    and .../av/.dylibs/libavdevice...  ... may cause ... mysterious crashes.

That collision corrupts av's audio decoder: aiortc's `track.recv()` then TIMES
OUT on every frame ("No audio frame received"), so VAD never fires and nothing
is ever transcribed — the exact bug we hit.

Pipecat's SmallWebRTC transport imports cv2 ONLY to color-convert incoming VIDEO
frames (`cv2.COLOR_YUV2RGB_*`, `cv2.cvtColor`). This demo sends NO video — only
mic audio — so that code path never runs. Registering this lightweight stub as
`cv2` in sys.modules (before pipecat imports it) satisfies the import without
ever loading the conflicting dylibs, and audio decoding works.

USAGE: `import cv2_stub  # noqa: F401` as the FIRST import in each entrypoint,
before anything that imports pipecat.
"""

import sys
import types

if "cv2" not in sys.modules:
    _cv2 = types.ModuleType("cv2")
    # Constants pipecat's transport references at import time.
    _cv2.COLOR_YUV2RGB_I420 = 0
    _cv2.COLOR_YUV2RGB_NV12 = 0
    _cv2.COLOR_GRAY2RGB = 0

    def _cvtColor(*args, **kwargs):  # only reached for video frames (never, here)
        raise RuntimeError(
            "cv2 stub: cvtColor called — this build handles audio only and should "
            "never process video frames."
        )

    _cv2.cvtColor = _cvtColor
    _cv2.__version__ = "stub-0"
    sys.modules["cv2"] = _cv2
