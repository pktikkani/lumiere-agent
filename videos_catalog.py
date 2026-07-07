"""Discover the demo videos and build the LLM's video enum + descriptions.

Drop .mp4 files into videos/ with descriptive filenames — the filename (minus
extension) becomes the video_id the LLM uses ("intro.mp4" -> "intro").

Optional: put a videos/descriptions.json alongside them to give the LLM a
one-line description per video, e.g.:

    {
      "intro": "Company introduction and mission",
      "product-demo": "Live walkthrough of the product",
      "customer-story": "Testimonial from Crowe Partners"
    }

Without descriptions.json, the filename (spaces/dashes cleaned up) is used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

VIDEOS_DIR = Path(__file__).parent / "videos"
DESCRIPTIONS_FILE = VIDEOS_DIR / "descriptions.json"


@dataclass(frozen=True)
class Video:
    video_id: str          # "intro"
    filename: str          # "intro.mp4"
    description: str        # human-readable, injected into the system prompt


@dataclass(frozen=True)
class Catalog:
    videos: list[Video]

    @property
    def ids(self) -> list[str]:
        return [v.video_id for v in self.videos]

    def title_for(self, video_id: str) -> str:
        for v in self.videos:
            if v.video_id == video_id:
                return v.description
        return video_id


def _pretty(video_id: str) -> str:
    return video_id.replace("-", " ").replace("_", " ").strip().capitalize()


def load_catalog() -> Catalog:
    """List videos/*.mp4 in sorted order; merge descriptions.json if present."""
    descriptions: dict[str, str] = {}
    if DESCRIPTIONS_FILE.exists():
        try:
            descriptions = json.loads(DESCRIPTIONS_FILE.read_text())
        except json.JSONDecodeError:
            descriptions = {}

    videos: list[Video] = []
    if VIDEOS_DIR.exists():
        for path in sorted(VIDEOS_DIR.glob("*.mp4")):
            video_id = path.stem
            videos.append(
                Video(
                    video_id=video_id,
                    filename=path.name,
                    description=descriptions.get(video_id, _pretty(video_id)),
                )
            )
    return Catalog(videos=videos)
