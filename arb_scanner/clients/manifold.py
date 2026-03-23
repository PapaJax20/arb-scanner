"""Manifold Markets API client."""

from __future__ import annotations

import logging
from typing import List

from arb_scanner.models import Market, Platform

from .base import BaseClient

logger = logging.getLogger(__name__)


class ManifoldClient(BaseClient):
    platform = Platform.MANIFOLD
    base_url = "https://api.manifold.markets/v0"

    async def fetch_markets(self) -> List[Market]:
        markets: List[Market] = []
        before: str | None = None
        pages_fetched = 0
        max_pages = 3

        while pages_fetched < max_pages:
            params: dict = {"limit": "500"}
            if before:
                params["before"] = before

            try:
                data = await self._get(f"{self.base_url}/markets", params=params)
            except Exception:
                logger.warning("Manifold: failed to fetch page %d", pages_fetched + 1)
                break

            if not isinstance(data, list) or not data:
                break

            for item in data:
                market = self._parse_market(item)
                if market:
                    markets.append(market)

            # Pagination: use last item's id as 'before' cursor
            before = data[-1].get("id")
            if len(data) < 500:
                break
            pages_fetched += 1

        logger.info("Manifold: fetched %d markets", len(markets))
        return markets

    def _parse_market(self, item: dict) -> Market | None:
        try:
            # Only binary markets
            if item.get("outcomeType") != "BINARY":
                return None

            if item.get("isResolved", False):
                return None

            question = item.get("question", "")
            if not question:
                return None

            probability = item.get("probability", 0.5)
            if probability is None:
                probability = 0.5

            yes_price = max(0.0, min(1.0, float(probability)))
            no_price = max(0.0, min(1.0, 1.0 - yes_price))

            market_id = item.get("id", "")
            slug = item.get("slug", "")

            return Market(
                platform=Platform.MANIFOLD,
                market_id=market_id,
                title=question,
                url=f"https://manifold.markets/{item.get('creatorUsername', 'market')}/{slug}" if slug else None,
                yes_price=yes_price,
                no_price=no_price,
                active=True,
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.debug("Manifold: skip market: %s", e)
            return None
