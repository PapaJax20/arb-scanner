"""Spread calculator with fee adjustment."""

from __future__ import annotations

import logging
from typing import List

from arb_scanner import config
from arb_scanner.models import MatchedPair, PlatformType, SpreadResult

logger = logging.getLogger(__name__)

# Fee lookup per platform (on 0-1 scale)
PLATFORM_FEES = {
    "polymarket": config.POLYMARKET_FEE,
    "kalshi": config.KALSHI_FEE,
}


def _get_fee(platform_name: str) -> float:
    return PLATFORM_FEES.get(platform_name, 0.0)


class SpreadCalculator:
    """Calculates spreads for matched market pairs."""

    def __init__(
        self,
        min_spread: float | None = None,
        calibration_threshold: float | None = None,
    ):
        self.min_spread = min_spread if min_spread is not None else config.MIN_SPREAD_THRESHOLD
        self.calibration_threshold = (
            calibration_threshold
            if calibration_threshold is not None
            else config.CALIBRATION_DIVERGENCE_THRESHOLD
        )

    def analyze(self, pairs: List[MatchedPair]) -> List[SpreadResult]:
        results: List[SpreadResult] = []
        for pair in pairs:
            result = self._analyze_pair(pair)
            if result:
                results.append(result)
        return results

    def _analyze_pair(self, pair: MatchedPair) -> SpreadResult | None:
        a = pair.market_a
        b = pair.market_b

        is_calibration = (
            a.platform_type != PlatformType.REAL_MONEY
            or b.platform_type != PlatformType.REAL_MONEY
        )

        if is_calibration:
            return self._analyze_calibration(pair)
        else:
            return self._analyze_arb(pair)

    def _analyze_arb(self, pair: MatchedPair) -> SpreadResult | None:
        """Analyze an arb opportunity between two real-money platforms."""
        a = pair.market_a
        b = pair.market_b

        fee_a = _get_fee(a.platform.value)
        fee_b = _get_fee(b.platform.value)

        # Strategy 1: Buy YES on A (cheap), sell YES on B (expensive) = buy NO on B
        # Profit = b.yes_price - a.yes_price - fees
        spread_1 = b.yes_price - a.yes_price
        direction_1 = f"Buy YES on {a.platform.value} @ {a.yes_price:.2f}, Buy NO on {b.platform.value} @ {b.no_price:.2f}"

        # Strategy 2: Buy YES on B (cheap), sell YES on A (expensive) = buy NO on A
        # Profit = a.yes_price - b.yes_price - fees
        spread_2 = a.yes_price - b.yes_price
        direction_2 = f"Buy YES on {b.platform.value} @ {b.yes_price:.2f}, Buy NO on {a.platform.value} @ {a.no_price:.2f}"

        # Pick the better direction
        if spread_1 >= spread_2:
            raw_spread = spread_1
            direction = direction_1
        else:
            raw_spread = spread_2
            direction = direction_2

        total_fees = fee_a + fee_b
        net_spread = raw_spread - total_fees

        # Convert to percentage
        spread_pct = raw_spread * 100
        fees_pct = total_fees * 100
        net_spread_pct = net_spread * 100

        is_actionable = net_spread_pct >= self.min_spread

        if net_spread_pct <= 0:
            return None

        return SpreadResult(
            pair=pair,
            spread_pct=spread_pct,
            fees_pct=fees_pct,
            net_spread_pct=net_spread_pct,
            direction=direction,
            is_actionable=is_actionable,
            is_calibration=False,
        )

    def _analyze_calibration(self, pair: MatchedPair) -> SpreadResult | None:
        """Analyze deviation between real-money and signal-only platform."""
        a = pair.market_a
        b = pair.market_b

        deviation = abs(a.yes_price - b.yes_price) * 100

        if deviation < self.calibration_threshold:
            return None

        return SpreadResult(
            pair=pair,
            spread_pct=deviation,
            fees_pct=0.0,
            net_spread_pct=deviation,
            direction=f"{a.platform.value} @ {a.yes_price:.2f} vs {b.platform.value} @ {b.yes_price:.2f}",
            is_actionable=False,
            is_calibration=True,
            deviation_pct=deviation,
        )
