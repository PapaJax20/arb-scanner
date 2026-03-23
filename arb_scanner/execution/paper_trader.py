"""Core paper trading engine — simulated trade execution.

ALL TRADES ARE SIMULATED. No real money is ever at risk.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from arb_scanner import config
from arb_scanner.constraints.engine import ConstraintViolation
from arb_scanner.constraints.sizing import BregmanProjection, TradeRecommendation
from arb_scanner.models import (
    Market,
    MatchedPair,
    PaperTrade,
    Platform,
    Portfolio,
    SpreadResult,
    TradeStatus,
)

logger = logging.getLogger(__name__)


def _generate_trade_id() -> str:
    """Generate a unique trade ID."""
    return f"PT-{uuid.uuid4().hex[:12].upper()}"


class PaperTrader:
    """Simulated paper trading engine.

    Executes both legs of cross-platform arb trades and intra-platform
    constraint arb trades using simulated balances. No real money involved.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        slippage_pct: float | None = None,
        max_position_pct: float | None = None,
    ):
        self.portfolio = portfolio
        self.slippage_pct = slippage_pct if slippage_pct is not None else config.SLIPPAGE_PCT
        self.max_position_pct = (
            max_position_pct if max_position_pct is not None else config.MAX_POSITION_PCT
        )
        self._bregman = BregmanProjection(max_edge_fraction=0.5)  # half-Kelly

    def _max_trade_size(self, platform: str) -> float:
        """Max position size for a single trade on this platform."""
        balance = self.portfolio.balances.get(platform, 0.0)
        return balance * self.max_position_pct

    def _apply_slippage(self, price: float, direction: str) -> float:
        """Apply slippage — worse fill for the trader.

        Buying: price goes up. Selling: price goes down.
        """
        if direction == "YES":
            return min(price * (1.0 + self.slippage_pct), 0.99)
        else:
            # Buying NO is equivalent to selling YES, price goes up
            return min(price * (1.0 + self.slippage_pct), 0.99)

    def _compute_kelly_size(
        self,
        edge: float,
        price: float,
        platform: str,
    ) -> float:
        """Half-Kelly position sizing.

        Kelly fraction = edge / odds, then halved.
        Size in dollars = fraction * bankroll, capped by max_position_pct.
        """
        if edge <= 0 or price <= 0 or price >= 1:
            return 0.0

        # odds = payout / cost = (1 - price) / price for YES bets
        odds = (1.0 - price) / price
        if odds <= 0:
            return 0.0

        kelly_fraction = edge / odds
        half_kelly = kelly_fraction * 0.5

        balance = self.portfolio.balances.get(platform, 0.0)
        size = balance * max(0.0, min(half_kelly, 1.0))

        # Cap at max position size
        cap = self._max_trade_size(platform)
        return min(size, cap)

    def execute_arb(
        self,
        pair: MatchedPair,
        spread: SpreadResult,
    ) -> list[PaperTrade]:
        """Execute both legs of a cross-platform arb trade.

        Returns list of executed PaperTrade objects (0 if rejected, 2 if filled).
        """
        if not spread.is_actionable:
            return []

        # Parse the direction string to determine which leg is which
        # Direction format: "Buy YES on polymarket @ 0.40, Buy NO on kalshi @ 0.55"
        trades: list[PaperTrade] = []
        link_id = _generate_trade_id()

        legs = self._parse_spread_direction(spread, pair)
        if not legs:
            logger.warning("Could not parse spread direction: %s", spread.direction)
            return []

        for leg in legs:
            platform_str = leg["platform"]
            market = leg["market"]
            direction = leg["direction"]
            raw_price = leg["price"]

            fill_price = self._apply_slippage(raw_price, direction)
            edge = spread.net_spread_pct / 100.0 / 2.0  # Split edge across both legs
            size = self._compute_kelly_size(edge, fill_price, platform_str)

            if size <= 0:
                logger.info("[PAPER] Skipping leg — zero size for %s on %s", direction, platform_str)
                continue

            balance = self.portfolio.balances.get(platform_str, 0.0)
            cost = size * fill_price
            if cost > balance:
                logger.info("[PAPER] Insufficient balance on %s: need $%.2f, have $%.2f",
                            platform_str, cost, balance)
                continue

            trade = PaperTrade(
                trade_id=_generate_trade_id(),
                platform=Platform(platform_str),
                market_id=market.market_id,
                market_title=market.title,
                direction=direction,
                entry_price=fill_price,
                size=size,
                simulated_cost=cost,
                linked_trade_id=link_id,
            )

            self.portfolio.balances[platform_str] -= cost
            self.portfolio.open_trades.append(trade)
            trades.append(trade)

            logger.info(
                "[PAPER] FILLED %s %s on %s @ %.4f | size=$%.2f | cost=$%.2f",
                direction, market.title[:50], platform_str, fill_price, size, cost,
            )

        return trades

    def execute_constraint_arb(
        self,
        violation: ConstraintViolation,
        trade_rec: TradeRecommendation,
    ) -> list[PaperTrade]:
        """Execute intra-platform constraint arb trades.

        Returns list of executed PaperTrade objects.
        """
        trades: list[PaperTrade] = []
        link_id = _generate_trade_id()

        for i, market in enumerate(violation.markets):
            direction_val = trade_rec.trade_directions[i]
            if abs(direction_val) < 1e-6:
                continue

            direction = "YES" if direction_val > 0 else "NO"
            platform_str = market.platform.value
            raw_price = market.yes_price if direction == "YES" else market.no_price
            fill_price = self._apply_slippage(raw_price, direction)

            edge = trade_rec.edge_per_market[i]
            size = self._compute_kelly_size(edge, fill_price, platform_str)

            if size <= 0:
                continue

            balance = self.portfolio.balances.get(platform_str, 0.0)
            cost = size * fill_price
            if cost > balance:
                logger.info("[PAPER] Insufficient balance on %s for constraint arb", platform_str)
                continue

            trade = PaperTrade(
                trade_id=_generate_trade_id(),
                platform=market.platform,
                market_id=market.market_id,
                market_title=market.title,
                direction=direction,
                entry_price=fill_price,
                size=size,
                simulated_cost=cost,
                linked_trade_id=link_id,
            )

            self.portfolio.balances[platform_str] -= cost
            self.portfolio.open_trades.append(trade)
            trades.append(trade)

            logger.info(
                "[PAPER] CONSTRAINT FILL %s %s @ %.4f | size=$%.2f",
                direction, market.title[:50], fill_price, size,
            )

        return trades

    def _parse_spread_direction(
        self,
        spread: SpreadResult,
        pair: MatchedPair,
    ) -> list[dict]:
        """Parse the spread direction string into structured leg info.

        Direction format: "Buy YES on polymarket @ 0.40, Buy NO on kalshi @ 0.55"
        """
        legs = []
        parts = spread.direction.split(", ")

        for part in parts:
            part_lower = part.lower()
            if "buy yes" in part_lower:
                direction = "YES"
            elif "buy no" in part_lower:
                direction = "NO"
            else:
                continue

            # Determine which market this leg refers to
            a = pair.market_a
            b = pair.market_b

            if a.platform.value in part_lower:
                market = a
                platform_str = a.platform.value
            elif b.platform.value in part_lower:
                market = b
                platform_str = b.platform.value
            else:
                continue

            price = market.yes_price if direction == "YES" else market.no_price

            legs.append({
                "platform": platform_str,
                "market": market,
                "direction": direction,
                "price": price,
            })

        return legs
