"""Polymarket CLOB API client."""

from __future__ import annotations

import logging
from typing import List

from arb_scanner.clients.base import BaseClient
from arb_scanner.models import Market, Platform

logger = logging.getLogger(__name__)


class PolymarketClient(BaseClient):
    platform = Platform.POLYMARKET
    base_url = "https://clob.polymarket.com"

    async def fetch_markets(self) -> List[Market]:
        markets: List[Market] = []
        next_cursor = None
        pages_fetched = 0
        max_pages = 5  # Limit to avoid excessive API calls

        while pages_fetched < max_pages:
            params = {}
            if next_cursor:
                params["next_cursor"] = next_cursor

            try:
                data = await self._get(f"{self.base_url}/markets", params=params)
            except Exception:
                logger.warning("Polymarket: failed to fetch page %d", pages_fetched + 1)
                break

            if not isinstance(data, dict):
                break

            for item in data.get("data", data.get("markets", [])):
                market = self._parse_market(item)
                if market:
                    markets.append(market)

            next_cursor = data.get("next_cursor")
            if not next_cursor or next_cursor == "LTE=":
                break
            pages_fetched += 1

        logger.info("Polymarket: fetched %d markets", len(markets))
        return markets

    def _parse_market(self, item: dict) -> Market | None:
        try:
            if not item.get("active", True) and item.get("closed", False):
                return None

            question = item.get("question", "")
            if not question:
                return None

            condition_id = item.get("condition_id", "")

            # Extract prices from tokens
            tokens = item.get("tokens", [])
            yes_price = 0.5
            no_price = 0.5

            for token in tokens:
                outcome = token.get("outcome", "").upper()
                price = float(token.get("price", 0.5))
                if outcome == "YES":
                    yes_price = price
                elif outcome == "NO":
                    no_price = price

            # Clamp prices
            yes_price = max(0.0, min(1.0, yes_price))
            no_price = max(0.0, min(1.0, no_price))

            return Market(
                platform=Platform.POLYMARKET,
                market_id=condition_id or item.get("id", ""),
                title=question,
                url=f"https://polymarket.com/event/{item.get('slug', condition_id)}",
                yes_price=yes_price,
                no_price=no_price,
                active=True,
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.debug("Polymarket: skip market: %s", e)
            return None
