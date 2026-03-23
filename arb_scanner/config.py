"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


# Discord
DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")

# Kalshi (optional for Phase 1)
KALSHI_API_KEY: str = os.getenv("KALSHI_API_KEY", "")
KALSHI_API_SECRET: str = os.getenv("KALSHI_API_SECRET", "")

# Scanner settings
SCAN_INTERVAL: int = int(os.getenv("SCAN_INTERVAL", "60"))
MIN_SPREAD_THRESHOLD: float = float(os.getenv("MIN_SPREAD_THRESHOLD", "4.0"))
MIN_MATCH_CONFIDENCE: float = float(os.getenv("MIN_MATCH_CONFIDENCE", "70"))

# API timeouts
API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "30"))

# Platform fees (on 0-1 scale)
POLYMARKET_FEE: float = 0.0  # maker rebates
KALSHI_FEE: float = 0.07  # 7 cents per contract worst case

# Calibration threshold
CALIBRATION_DIVERGENCE_THRESHOLD: float = float(
    os.getenv("CALIBRATION_DIVERGENCE_THRESHOLD", "5.0")
)

# ---------------------------------------------------------------------------
# Paper Trading (Phase 2)
# ---------------------------------------------------------------------------
PAPER_TRADE_ENABLED: bool = os.getenv("PAPER_TRADE_ENABLED", "true").lower() == "true"
INITIAL_BALANCE: float = float(os.getenv("INITIAL_BALANCE", "10000.0"))
MAX_POSITION_PCT: float = float(os.getenv("MAX_POSITION_PCT", "0.05"))
SLIPPAGE_PCT: float = float(os.getenv("SLIPPAGE_PCT", "0.005"))
PNL_SUMMARY_INTERVAL: int = int(os.getenv("PNL_SUMMARY_INTERVAL", "10"))
DISCORD_ALERTS_WEBHOOK_URL: str = os.getenv("DISCORD_ALERTS_WEBHOOK_URL", "")

# Data persistence
DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
