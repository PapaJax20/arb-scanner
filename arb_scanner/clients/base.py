"""Abstract base client for platform APIs."""

from __future__ import annotations

import abc
import logging
from typing import List

import aiohttp

from arb_scanner import config
from arb_scanner.models import Market, Platform

logger = logging.getLogger(__name__)


class BaseClient(abc.ABC):
    """Abstract base class for all platform API clients."""

    platform: Platform
    base_url: str

    def __init__(self, session: aiohttp.ClientSession | None = None):
        self._external_session = session is not None
        self._session = session

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=config.API_TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if not self._external_session and self._session and not self._session.closed:
            await self._session.close()

    async def _get(self, url: str, params: dict | None = None) -> dict | list:
        session = await self._get_session()
        try:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                return await resp.json()
        except aiohttp.ClientError as e:
            logger.warning("%s API error on GET %s: %s", self.platform.value, url, e)
            raise
        except Exception as e:
            logger.warning(
                "%s unexpected error on GET %s: %s", self.platform.value, url, e
            )
            raise

    @abc.abstractmethod
    async def fetch_markets(self) -> List[Market]:
        """Fetch all active binary markets from this platform."""
        ...
