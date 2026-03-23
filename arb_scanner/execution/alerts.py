"""Discord alerting for paper trades.

All alerts are clearly labeled [PAPER TRADE] to avoid any confusion
with real trading activity.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import aiohttp

from arb_scanner import config
from arb_scanner.models import PaperTrade, TradeResult

logger = logging.getLogger(__name__)


def _color_for_pnl(pnl: float) -> int:
    """Green for profit, red for loss."""
    return 0x2ECC71 if pnl >= 0 else 0xE74C3C


class PaperTradeAlerter:
    """Sends paper trade alerts to Discord."""

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or config.DISCORD_ALERTS_WEBHOOK_URL

    async def send_fill_alert(self, trade: PaperTrade) -> None:
        """Post to Discord when a paper trade fills."""
        if not self.webhook_url:
            return

        embed = {
            "title": f"\U0001f4dd [PAPER TRADE] Fill — {trade.direction} {trade.market_title[:80]}",
            "color": 0x3498DB,  # blue
            "fields": [
                {"name": "Trade ID", "value": trade.trade_id, "inline": True},
                {"name": "Platform", "value": trade.platform.value, "inline": True},
                {"name": "Direction", "value": trade.direction, "inline": True},
                {"name": "Entry Price", "value": f"{trade.entry_price:.4f}", "inline": True},
                {"name": "Size", "value": f"${trade.size:.2f}", "inline": True},
                {"name": "Cost", "value": f"${trade.simulated_cost:.2f}", "inline": True},
            ],
            "footer": {"text": "\u26a0\ufe0f SIMULATED — No real money"},
            "timestamp": trade.timestamp.isoformat(),
        }

        await self._post(embed)

    async def send_pnl_summary(self, summary: dict) -> None:
        """Post periodic P&L summary."""
        if not self.webhook_url:
            return

        pnl = summary.get("realized_pnl", 0.0) + summary.get("unrealized_pnl", 0.0)
        embed = {
            "title": "\U0001f4ca [PAPER TRADE] Portfolio Summary",
            "color": _color_for_pnl(pnl),
            "fields": [
                {"name": "Total Balance", "value": f"${summary['total_balance']:.2f}", "inline": True},
                {"name": "Open Positions", "value": str(summary["open_positions"]), "inline": True},
                {"name": "Win Rate", "value": f"{summary['win_rate']:.0%}", "inline": True},
                {"name": "Realized P&L", "value": f"${summary['realized_pnl']:.2f}", "inline": True},
                {"name": "Unrealized P&L", "value": f"${summary['unrealized_pnl']:.2f}", "inline": True},
            ],
            "footer": {"text": "\u26a0\ufe0f SIMULATED — No real money"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        for platform, balance in summary.get("balances", {}).items():
            embed["fields"].append(
                {"name": f"{platform} balance", "value": f"${balance:.2f}", "inline": True}
            )

        await self._post(embed)

    async def send_settlement_alert(self, trade: PaperTrade, result: TradeResult) -> None:
        """Post when a position resolves."""
        if not self.webhook_url:
            return

        embed = {
            "title": f"\U0001f3c1 [PAPER TRADE] Settled — {trade.market_title[:80]}",
            "color": _color_for_pnl(result.pnl),
            "fields": [
                {"name": "Trade ID", "value": result.trade_id, "inline": True},
                {"name": "Direction", "value": trade.direction, "inline": True},
                {"name": "Entry Price", "value": f"{trade.entry_price:.4f}", "inline": True},
                {"name": "Settlement", "value": f"{result.settlement_price:.4f}", "inline": True},
                {"name": "P&L", "value": f"${result.pnl:+.2f}", "inline": True},
                {"name": "Platform", "value": trade.platform.value, "inline": True},
            ],
            "footer": {"text": "\u26a0\ufe0f SIMULATED — No real money"},
            "timestamp": result.resolved_at.isoformat(),
        }

        await self._post(embed)

    async def _post(self, embed: dict) -> None:
        """Post a single embed to the webhook."""
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.webhook_url, json={"embeds": [embed]}
                ) as resp:
                    if resp.status == 429:
                        logger.warning("[PAPER] Discord rate limited")
                    resp.raise_for_status()
        except Exception as e:
            logger.warning("[PAPER] Failed to send Discord alert: %s", e)
