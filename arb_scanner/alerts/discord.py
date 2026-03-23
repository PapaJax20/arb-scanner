"""Discord webhook alerting."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List

import aiohttp

from arb_scanner import config
from arb_scanner.models import SpreadResult

logger = logging.getLogger(__name__)


def _embed_color(net_spread_pct: float, is_calibration: bool) -> int:
    """Return Discord embed color based on spread magnitude."""
    if is_calibration:
        return 0x3498DB  # blue for calibration
    if net_spread_pct >= 8.0:
        return 0xE74C3C  # red — exceptional
    if net_spread_pct >= 4.0:
        return 0x2ECC71  # green — actionable
    return 0xF39C12  # yellow — watch


def _build_embed(result: SpreadResult) -> dict:
    pair = result.pair
    a = pair.market_a
    b = pair.market_b

    if result.is_calibration:
        title = f"📊 Calibration Divergence: {result.deviation_pct:.1f}%"
    else:
        title = f"💰 Arb Spread: {result.net_spread_pct:.1f}% net"

    fields = [
        {
            "name": "Event",
            "value": a.title[:200],
            "inline": False,
        },
        {
            "name": f"{a.platform.value}",
            "value": f"YES: {a.yes_price:.3f} | NO: {a.no_price:.3f}",
            "inline": True,
        },
        {
            "name": f"{b.platform.value}",
            "value": f"YES: {b.yes_price:.3f} | NO: {b.no_price:.3f}",
            "inline": True,
        },
        {
            "name": "Match Confidence",
            "value": f"{pair.similarity_score:.0f}%",
            "inline": True,
        },
    ]

    if not result.is_calibration:
        fields.extend(
            [
                {
                    "name": "Gross Spread",
                    "value": f"{result.spread_pct:.2f}%",
                    "inline": True,
                },
                {
                    "name": "Fees",
                    "value": f"{result.fees_pct:.2f}%",
                    "inline": True,
                },
                {
                    "name": "Net Spread",
                    "value": f"**{result.net_spread_pct:.2f}%**",
                    "inline": True,
                },
                {
                    "name": "Direction",
                    "value": result.direction,
                    "inline": False,
                },
            ]
        )

    # Add market links
    links = []
    if a.url:
        links.append(f"[{a.platform.value}]({a.url})")
    if b.url:
        links.append(f"[{b.platform.value}]({b.url})")
    if links:
        fields.append(
            {"name": "Links", "value": " | ".join(links), "inline": False}
        )

    return {
        "title": title,
        "color": _embed_color(result.net_spread_pct, result.is_calibration),
        "fields": fields,
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {"text": "arb-scanner v0.1.0"},
    }


class DiscordAlerter:
    """Sends alerts via Discord webhook."""

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or config.DISCORD_WEBHOOK_URL

    async def send_alerts(self, results: List[SpreadResult]) -> int:
        """Send alerts for actionable results. Returns count of alerts sent."""
        if not self.webhook_url:
            logger.warning("Discord webhook URL not configured — skipping alerts")
            return 0

        sent = 0
        for result in results:
            if not result.is_actionable and not result.is_calibration:
                continue
            try:
                await self._post_embed(result)
                sent += 1
            except Exception as e:
                logger.warning("Failed to send Discord alert: %s", e)

        if sent:
            logger.info("Sent %d Discord alerts", sent)
        return sent

    async def _post_embed(self, result: SpreadResult) -> None:
        embed = _build_embed(result)
        payload = {"embeds": [embed]}

        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.webhook_url, json=payload) as resp:
                if resp.status == 429:
                    logger.warning("Discord rate limited")
                resp.raise_for_status()
