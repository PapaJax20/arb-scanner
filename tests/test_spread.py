"""Unit tests for the spread calculator."""

import pytest

from arb_scanner.analysis.spread import SpreadCalculator
from arb_scanner.models import Market, MatchedPair, Platform


def _make_pair(
    platform_a: Platform,
    yes_a: float,
    platform_b: Platform,
    yes_b: float,
    score: float = 85.0,
) -> MatchedPair:
    return MatchedPair(
        market_a=Market(
            platform=platform_a,
            market_id="a",
            title="Test event",
            yes_price=yes_a,
            no_price=1.0 - yes_a,
        ),
        market_b=Market(
            platform=platform_b,
            market_id="b",
            title="Test event",
            yes_price=yes_b,
            no_price=1.0 - yes_b,
        ),
        similarity_score=score,
        normalized_title="test event",
    )


class TestSpreadCalculatorArb:
    def test_positive_spread_polymarket_kalshi(self):
        """Spread exists: poly YES=0.40, kalshi YES=0.55 → raw 15%, net 8% after 7% kalshi fee."""
        pair = _make_pair(Platform.POLYMARKET, 0.40, Platform.KALSHI, 0.55)
        calc = SpreadCalculator(min_spread=4.0)
        results = calc.analyze([pair])
        assert len(results) == 1
        r = results[0]
        assert not r.is_calibration
        assert r.spread_pct == pytest.approx(15.0, abs=0.1)
        assert r.fees_pct == pytest.approx(7.0, abs=0.1)
        assert r.net_spread_pct == pytest.approx(8.0, abs=0.1)
        assert r.is_actionable

    def test_no_spread(self):
        """Same prices → no arb after fees."""
        pair = _make_pair(Platform.POLYMARKET, 0.50, Platform.KALSHI, 0.50)
        calc = SpreadCalculator(min_spread=4.0)
        results = calc.analyze([pair])
        # Net spread is negative (0% - 7% fees), so should return None
        assert len(results) == 0

    def test_small_spread_not_actionable(self):
        """Spread below threshold."""
        pair = _make_pair(Platform.POLYMARKET, 0.50, Platform.KALSHI, 0.60)
        calc = SpreadCalculator(min_spread=4.0)
        results = calc.analyze([pair])
        assert len(results) == 1
        assert not results[0].is_actionable  # 10% - 7% = 3% < 4% threshold

    def test_direction_buy_yes_on_cheaper(self):
        """Should buy YES on the platform with lower YES price."""
        pair = _make_pair(Platform.POLYMARKET, 0.30, Platform.KALSHI, 0.55)
        calc = SpreadCalculator(min_spread=0.0)
        results = calc.analyze([pair])
        assert len(results) == 1
        assert "polymarket" in results[0].direction.lower()
        assert "Buy YES on polymarket" in results[0].direction

    def test_reverse_direction(self):
        """Kalshi cheaper → buy YES on kalshi."""
        pair = _make_pair(Platform.POLYMARKET, 0.60, Platform.KALSHI, 0.40)
        calc = SpreadCalculator(min_spread=0.0)
        results = calc.analyze([pair])
        assert len(results) == 1
        assert "Buy YES on kalshi" in results[0].direction

    def test_large_spread_actionable(self):
        """Large spread is actionable."""
        pair = _make_pair(Platform.POLYMARKET, 0.20, Platform.KALSHI, 0.50)
        calc = SpreadCalculator(min_spread=4.0)
        results = calc.analyze([pair])
        assert len(results) == 1
        r = results[0]
        assert r.is_actionable
        assert r.net_spread_pct == pytest.approx(23.0, abs=0.1)


class TestSpreadCalculatorCalibration:
    def test_calibration_divergence(self):
        """Polymarket vs Manifold with >5% divergence."""
        pair = _make_pair(Platform.POLYMARKET, 0.60, Platform.MANIFOLD, 0.50)
        calc = SpreadCalculator(calibration_threshold=5.0)
        results = calc.analyze([pair])
        assert len(results) == 1
        r = results[0]
        assert r.is_calibration
        assert r.deviation_pct == pytest.approx(10.0, abs=0.1)

    def test_calibration_no_divergence(self):
        """Close prices → no calibration alert."""
        pair = _make_pair(Platform.POLYMARKET, 0.50, Platform.MANIFOLD, 0.52)
        calc = SpreadCalculator(calibration_threshold=5.0)
        results = calc.analyze([pair])
        assert len(results) == 0

    def test_metaculus_calibration(self):
        """Metaculus as calibration signal."""
        pair = _make_pair(Platform.KALSHI, 0.70, Platform.METACULUS, 0.55)
        calc = SpreadCalculator(calibration_threshold=5.0)
        results = calc.analyze([pair])
        assert len(results) == 1
        assert results[0].is_calibration
        assert results[0].deviation_pct == pytest.approx(15.0, abs=0.1)
