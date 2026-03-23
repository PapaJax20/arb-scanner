"""Platform API clients."""

from .kalshi import KalshiClient
from .manifold import ManifoldClient
from .metaculus import MetaculusClient
from .polymarket import PolymarketClient

__all__ = ["PolymarketClient", "KalshiClient", "ManifoldClient", "MetaculusClient"]
