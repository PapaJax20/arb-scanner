"""Position tracking and P&L management with JSON persistence."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from arb_scanner import config
from arb_scanner.models import (
    PaperTrade,
    Portfolio,
    TradeResult,
    TradeStatus,
)

logger = logging.getLogger(__name__)

_DEFAULT_PATH = config.DATA_DIR / "positions.json"


class PositionTracker:
    """Tracks paper trading positions, P&L, and persists state to disk."""

    def __init__(self, portfolio: Portfolio, persist_path: Path | None = None):
        self.portfolio = portfolio
        self.persist_path = persist_path or _DEFAULT_PATH

    # ------------------------------------------------------------------
    # Mark-to-market
    # ------------------------------------------------------------------

    def mark_to_market(self, price_map: dict[str, float]) -> float:
        """Update unrealized P&L for all open positions.

        Args:
            price_map: mapping of market_id -> current YES price.

        Returns:
            Total unrealized P&L across all open positions.
        """
        total_unrealized = 0.0

        for trade in self.portfolio.open_trades:
            current_price = price_map.get(trade.market_id)
            if current_price is None:
                continue

            if trade.direction == "YES":
                # Bought YES: profit if price goes up
                unrealized = (current_price - trade.entry_price) * trade.size
            else:
                # Bought NO: profit if YES price goes down
                unrealized = (trade.entry_price - current_price) * trade.size

            total_unrealized += unrealized

        self.portfolio.total_unrealized_pnl = total_unrealized
        return total_unrealized

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle_trade(self, trade_id: str, settlement_price: float) -> Optional[TradeResult]:
        """Settle an open trade when its market resolves.

        Args:
            trade_id: ID of the trade to settle.
            settlement_price: Final resolved price (0.0 or 1.0 typically).

        Returns:
            TradeResult if settled, None if trade not found.
        """
        trade = self._find_open_trade(trade_id)
        if trade is None:
            logger.warning("Trade %s not found in open positions", trade_id)
            return None

        if trade.status == TradeStatus.SETTLED:
            logger.warning("Trade %s already settled", trade_id)
            return None

        # Calculate P&L
        if trade.direction == "YES":
            pnl = (settlement_price - trade.entry_price) * trade.size
        else:
            pnl = (trade.entry_price - settlement_price) * trade.size

        now = datetime.now(timezone.utc)

        # Update trade
        trade.status = TradeStatus.SETTLED
        trade.settlement_price = settlement_price
        trade.realized_pnl = pnl
        trade.resolved_at = now

        # Move from open to settled
        self.portfolio.open_trades = [
            t for t in self.portfolio.open_trades if t.trade_id != trade_id
        ]
        self.portfolio.settled_trades.append(trade)
        self.portfolio.total_realized_pnl += pnl

        # Return capital + P&L to balance
        platform_str = trade.platform.value
        self.portfolio.balances[platform_str] = (
            self.portfolio.balances.get(platform_str, 0.0) + trade.simulated_cost + pnl
        )

        logger.info(
            "[PAPER] SETTLED %s | pnl=$%.2f | settlement=%.2f",
            trade_id, pnl, settlement_price,
        )

        return TradeResult(
            trade_id=trade_id,
            settlement_price=settlement_price,
            pnl=pnl,
            resolved_at=now,
        )

    # ------------------------------------------------------------------
    # Portfolio summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Generate portfolio summary."""
        total_balance = sum(self.portfolio.balances.values())
        settled = self.portfolio.settled_trades
        wins = [t for t in settled if (t.realized_pnl or 0) > 0]
        win_rate = len(wins) / len(settled) if settled else 0.0

        return {
            "total_balance": total_balance,
            "balances": dict(self.portfolio.balances),
            "open_positions": len(self.portfolio.open_trades),
            "settled_positions": len(settled),
            "unrealized_pnl": self.portfolio.total_unrealized_pnl,
            "realized_pnl": self.portfolio.total_realized_pnl,
            "win_rate": win_rate,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist portfolio state to JSON."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.portfolio.model_dump(mode="json")
        self.persist_path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("[PAPER] Portfolio state saved to %s", self.persist_path)

    @classmethod
    def load(cls, persist_path: Path | None = None) -> "PositionTracker":
        """Load portfolio state from JSON, or create fresh if none exists."""
        path = persist_path or _DEFAULT_PATH

        if path.exists():
            try:
                data = json.loads(path.read_text())
                portfolio = Portfolio.model_validate(data)
                logger.info(
                    "[PAPER] Loaded portfolio: %d open, %d settled positions",
                    len(portfolio.open_trades),
                    len(portfolio.settled_trades),
                )
                return cls(portfolio=portfolio, persist_path=path)
            except Exception as e:
                logger.error("Failed to load portfolio from %s: %s", path, e)

        # Fresh portfolio
        portfolio = Portfolio(
            balances={
                "polymarket": config.INITIAL_BALANCE,
                "kalshi": config.INITIAL_BALANCE,
            }
        )
        logger.info("[PAPER] Created fresh portfolio with $%.0f per platform", config.INITIAL_BALANCE)
        return cls(portfolio=portfolio, persist_path=path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_open_trade(self, trade_id: str) -> Optional[PaperTrade]:
        for trade in self.portfolio.open_trades:
            if trade.trade_id == trade_id:
                return trade
        return None
