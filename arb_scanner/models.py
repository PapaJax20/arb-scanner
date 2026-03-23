"""Shared data models for the arbitrage scanner."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Platform(str, Enum):
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"
    MANIFOLD = "manifold"
    METACULUS = "metaculus"


class PlatformType(str, Enum):
    REAL_MONEY = "real_money"
    PLAY_MONEY = "play_money"
    REPUTATION = "reputation"


PLATFORM_TYPES = {
    Platform.POLYMARKET: PlatformType.REAL_MONEY,
    Platform.KALSHI: PlatformType.REAL_MONEY,
    Platform.MANIFOLD: PlatformType.PLAY_MONEY,
    Platform.METACULUS: PlatformType.REPUTATION,
}


class Market(BaseModel):
    """A single binary market from any platform."""

    platform: Platform
    market_id: str
    title: str
    url: Optional[str] = None
    yes_price: float = Field(ge=0.0, le=1.0, description="YES price normalized to 0-1")
    no_price: float = Field(ge=0.0, le=1.0, description="NO price normalized to 0-1")
    active: bool = True
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def platform_type(self) -> PlatformType:
        return PLATFORM_TYPES[self.platform]


class MatchedPair(BaseModel):
    """Two markets from different platforms matched as the same event."""

    market_a: Market
    market_b: Market
    similarity_score: float = Field(ge=0.0, le=100.0)
    normalized_title: str = ""


class SpreadResult(BaseModel):
    """Result of spread analysis on a matched pair."""

    pair: MatchedPair
    spread_pct: float
    fees_pct: float
    net_spread_pct: float
    direction: str = ""  # e.g. "Buy YES on polymarket, Buy NO on kalshi"
    is_actionable: bool = False
    is_calibration: bool = False  # True for play-money/reputation pairs
    deviation_pct: float = 0.0  # For calibration pairs


# ---------------------------------------------------------------------------
# Phase 2: Paper Trading Models
# ---------------------------------------------------------------------------


class TradeStatus(str, Enum):
    OPEN = "open"
    SETTLED = "settled"


class PaperTrade(BaseModel):
    """A single simulated paper trade."""

    trade_id: str
    platform: Platform
    market_id: str
    market_title: str
    direction: str  # "YES" or "NO"
    entry_price: float = Field(ge=0.0, le=1.0)
    size: float = Field(gt=0.0, description="Dollar amount wagered")
    simulated_cost: float = Field(description="Cost after slippage")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: TradeStatus = TradeStatus.OPEN
    settlement_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    resolved_at: Optional[datetime] = None

    # Metadata linking to the arb opportunity
    linked_trade_id: Optional[str] = None  # The other leg of the arb


class TradeResult(BaseModel):
    """Settlement result for a resolved trade."""

    trade_id: str
    settlement_price: float = Field(ge=0.0, le=1.0)
    pnl: float
    resolved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Portfolio(BaseModel):
    """Full portfolio state for paper trading."""

    balances: Dict[str, float] = Field(
        default_factory=lambda: {"polymarket": 10000.0, "kalshi": 10000.0}
    )
    open_trades: List[PaperTrade] = Field(default_factory=list)
    settled_trades: List[PaperTrade] = Field(default_factory=list)
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
