"""Kalshi public market data API client."""

from __future__ import annotations

import logging
from typing import List

from arb_scanner.clients.base import BaseClient
from arb_scanner.models import Market, Platform

logger = logging.getLogger(__name__)


class KalshiClient(BaseClient):
    platform = Platform.KALSHI
    base_url = "https://api.elections.kalshi.com/trade-api/v2"

    async def fetch_markets(self) -> List[Market]:
        markets: List[Market] = []
        cursor: str | None = None
        pages_fetched = 0
        max_pages = 5

        while pages_fetched < max_pages:
            params: dict = {"limit": "200"}
            if cursor:
                params["cursor"] = cursor

            try:
                data = await self._get(f"{self.base_url}/markets", params=params)
            except Exception:
                logger.warning("Kalshi: failed to fetch page %d", pages_fetched + 1)
                break

            if not isinstance(data, dict):
                break

            for item in data.get("markets", []):
                market = self._parse_market(item)
                if market:
                    markets.append(market)

            cursor = data.get("cursor")
            if not cursor:
                break
            pages_fetched += 1

        logger.info("Kalshi: fetched %d markets", len(markets))
        return markets

    def _parse_market(self, item: dict) -> Market | None:
        try:
            status = item.get("status", "")
            if status not in ("open", "active"):
                return None

            title = item.get("title", "")
            if not title:
                return None

            ticker = item.get("ticker", "")

            # Kalshi prices are in cents (0-100), normalize to 0-1
            yes_bid = item.get("yes_bid", 0) or 0
            yes_ask = item.get("yes_ask", 0) or 0
            no_bid = item.get("no_bid", 0) or 0
            no_ask = item.get("no_ask", 0) or 0

            # Use midpoint of bid/ask, fallback to last price
            if yes_bid and yes_ask:
                yes_price = (yes_bid + yes_ask) / 2 / 100
            elif item.get("last_price") is not None:
                yes_price = (item.get("last_price", 50) or 50) / 100
            else:
                yes_price = 0.5

            if no_bid and no_ask:
                no_price = (no_bid + no_ask) / 2 / 100
            else:
                no_price = 1.0 - yes_price

            yes_price = max(0.0, min(1.0, yes_price))
            no_price = max(0.0, min(1.0, no_price))

            return Market(
                platform=Platform.KALSHI,
                market_id=ticker,
                title=title,
                url=f"https://kalshi.com/markets/{ticker}",
                yes_price=yes_price,
                no_price=no_price,
                active=True,
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.debug("Kalshi: skip market: %s", e)
            return None
