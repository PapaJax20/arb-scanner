"""Metaculus API client."""

from __future__ import annotations

import logging
from typing import List

from arb_scanner.models import Market, Platform

from .base import BaseClient

logger = logging.getLogger(__name__)


class MetaculusClient(BaseClient):
    platform = Platform.METACULUS
    base_url = "https://www.metaculus.com/api"

    async def fetch_markets(self) -> List[Market]:
        """Try the current /api/questions/ endpoint, fall back to legacy /api2/."""
        markets = await self._fetch_from("/api/questions/")
        if not markets:
            markets = await self._fetch_from("/api2/questions/")
        logger.info("Metaculus: fetched %d markets", len(markets))
        return markets

    async def _fetch_from(self, path: str) -> List[Market]:
        markets: List[Market] = []
        offset = 0
        max_pages = 3
        page_size = 100

        for _ in range(max_pages):
            params = {
                "type": "forecast",
                "forecast_type": "binary",
                "status": "open",
                "limit": str(page_size),
                "offset": str(offset),
                "order_by": "-activity",
            }

            try:
                data = await self._get(f"https://www.metaculus.com{path}", params=params)
            except Exception:
                logger.debug("Metaculus: failed to fetch %s offset %d", path, offset)
                break

            if not isinstance(data, dict):
                break

            results = data.get("results", [])
            for item in results:
                market = self._parse_market(item)
                if market:
                    markets.append(market)

            if not data.get("next"):
                break
            offset += page_size

        return markets

    def _parse_market(self, item: dict) -> Market | None:
        try:
            title = item.get("title", "")
            if not title:
                return None

            question_id = item.get("id", "")

            # Community prediction median (q2)
            community = item.get("community_prediction") or {}
            full = community.get("full") or {}
            probability = full.get("q2")

            if probability is None:
                # Try alternative path
                probability = community.get("q2")

            if probability is None:
                return None

            yes_price = max(0.0, min(1.0, float(probability)))
            no_price = max(0.0, min(1.0, 1.0 - yes_price))

            return Market(
                platform=Platform.METACULUS,
                market_id=str(question_id),
                title=title,
                url=f"https://www.metaculus.com/questions/{question_id}/",
                yes_price=yes_price,
                no_price=no_price,
                active=True,
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.debug("Metaculus: skip market: %s", e)
            return None
