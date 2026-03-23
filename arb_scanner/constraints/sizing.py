"""Bregman Projection / KL Divergence for optimal trade sizing.

Since prediction market prices are probabilities, we use KL divergence
(the natural Bregman divergence for the probability simplex) rather than
Euclidean distance to compute optimal trade direction and size.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Small epsilon to avoid log(0)
_EPS = 1e-10


class TradeRecommendation(BaseModel):
    """Optimal trade direction and sizing from Bregman projection."""

    market_ids: List[str]
    current_prices: List[float]
    fair_prices: List[float]
    trade_directions: List[float]  # positive = buy YES, negative = buy NO
    kl_divergence: float  # D_KL(fair || current) — magnitude of mispricing
    edge_per_market: List[float]  # Expected edge per market in probability units


class BregmanProjection:
    """Compute optimal trade sizing using KL divergence on the probability simplex.

    Given current market prices p and LP-derived fair prices q, the
    KL divergence D_KL(q || p) = sum(q_i * log(q_i / p_i)) measures the
    information-theoretic distance between the two distributions.

    The trade direction is determined by the gradient of KL divergence,
    and the magnitude indicates confidence/edge.
    """

    def __init__(self, max_edge_fraction: float = 0.5):
        """
        Args:
            max_edge_fraction: Maximum fraction of edge to capture per trade
                (Kelly-style sizing). Default 0.5 = half-Kelly.
        """
        self.max_edge_fraction = max_edge_fraction

    def compute_trades(
        self,
        market_ids: List[str],
        current_prices: List[float],
        fair_prices: List[float],
    ) -> TradeRecommendation:
        """Compute optimal trade direction and size.

        Args:
            market_ids: Identifiers for each market.
            current_prices: Current market YES prices (probabilities).
            fair_prices: LP-derived fair YES prices (probabilities).

        Returns:
            TradeRecommendation with directions and sizing.
        """
        p = np.array(current_prices, dtype=np.float64)
        q = np.array(fair_prices, dtype=np.float64)

        # Clamp to valid probability range
        p = np.clip(p, _EPS, 1.0 - _EPS)
        q = np.clip(q, _EPS, 1.0 - _EPS)

        # KL divergence: D_KL(q || p) = sum(q * log(q/p))
        kl = self._kl_divergence(q, p)

        # Trade direction: where fair > current, buy YES; where fair < current, buy NO
        edge = q - p
        directions = edge * self.max_edge_fraction

        # Edge per market
        edge_per = self._edge_per_market(p, q)

        return TradeRecommendation(
            market_ids=market_ids,
            current_prices=p.tolist(),
            fair_prices=q.tolist(),
            trade_directions=directions.tolist(),
            kl_divergence=float(kl),
            edge_per_market=edge_per.tolist(),
        )

    @staticmethod
    def _kl_divergence(q: np.ndarray, p: np.ndarray) -> float:
        """Compute D_KL(q || p) = sum(q_i * log(q_i / p_i)).

        This is the Bregman divergence associated with negative entropy,
        which is the natural divergence for probability distributions.
        """
        # For binary markets, each price is an independent probability,
        # so we compute the sum of per-market KL terms.
        # Each market contributes: q*log(q/p) + (1-q)*log((1-q)/(1-p))
        q_safe = np.clip(q, _EPS, 1.0 - _EPS)
        p_safe = np.clip(p, _EPS, 1.0 - _EPS)

        kl_yes = q_safe * np.log(q_safe / p_safe)
        kl_no = (1.0 - q_safe) * np.log((1.0 - q_safe) / (1.0 - p_safe))

        return float(np.sum(kl_yes + kl_no))

    @staticmethod
    def _edge_per_market(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Compute expected edge per market.

        Edge = |q_i - p_i| — the absolute probability difference,
        which represents the expected profit per unit bet at current prices
        if the fair price is correct.
        """
        return np.abs(q - p)

    @staticmethod
    def bregman_project_to_simplex(
        prices: np.ndarray,
        target_sum: float = 1.0,
    ) -> np.ndarray:
        """Bregman (KL) projection onto the simplex {p : sum(p) = target_sum, p > 0}.

        The KL projection of q onto the simplex is:
            p_i = q_i * (target_sum / sum(q_j))

        This is the information-theoretic projection (maximum entropy
        projection) that preserves the ratio structure of the input.
        Unlike Euclidean projection, it never produces zeros for positive inputs.
        """
        prices = np.clip(prices, _EPS, None)
        return prices * (target_sum / prices.sum())
