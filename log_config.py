"""Central logging config: send everything to files, keep the console quiet.

All logs (server + bot + forwarded browser logs) land in:
    /Users/pavan/Documents/applogs/lumiere-agent/lumiere.log   (rotated)

Console keeps only INFO+ so the terminal stays clean (no ICE/asyncio DEBUG noise);
the file keeps full DEBUG for when you actually need to dig.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

LOG_DIR = Path(
    os.getenv("LUMIERE_LOG_DIR", "/Users/pavan/Documents/applogs/lumiere-agent")
)

_configured = False


def setup_logging() -> Path:
    """Idempotent. Returns the log directory."""
    global _configured
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if _configured:
        return LOG_DIR

    logger.remove()  # drop loguru's default stderr sink

    # Quiet console: INFO and up only.
    logger.add(
        sys.stderr,
        level=os.getenv("CONSOLE_LOG_LEVEL", "INFO"),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    )

    # Full-detail file: everything, rotated.
    logger.add(
        LOG_DIR / "lumiere.log",
        level="DEBUG",
        rotation="10 MB",
        retention="10 days",
        enqueue=True,          # safe across the bot's async tasks
        backtrace=True,
        diagnose=False,
    )

    _configured = True
    logger.info(f"Logging to {LOG_DIR / 'lumiere.log'}")
    return LOG_DIR
