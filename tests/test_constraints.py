"""Tests for the constraint arbitrage engine."""

from __future__ import annotations

import numpy as np
import pytest

from arb_scanner.constraints.engine import ConstraintSolver, ConstraintViolation
from arb_scanner.constraints.mapper import (
    MarketRelationship,
    MarketRelationshipMapper,
    RelationshipType,
)
from arb_scanner.constraints.sizing import BregmanProjection
from arb_scanner.models import Market, Platform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _market(
    title: str,
    yes_price: float,
    market_id: str = "",
    platform: Platform = Platform.POLYMARKET,
) -> Market:
    return Market(
        platform=platform,
        market_id=market_id or title[:8],
        title=title,
        yes_price=yes_price,
        no_price=round(1.0 - yes_price, 4),
    )


def _mutex_rel(markets: list[Market]) -> MarketRelationship:
    return MarketRelationship(
        type=RelationshipType.MUTEX,
        markets=markets,
        confidence=90.0,
    )


def _nested_rel(child: Market, parent: Market) -> MarketRelationship:
    return MarketRelationship(
        type=RelationshipType.NESTED,
        markets=[child, parent],
        confidence=90.0,
    )


def _complementary_rel(markets: list[Market]) -> MarketRelationship:
    return MarketRelationship(
        type=RelationshipType.COMPLEMENTARY,
        markets=markets,
        confidence=90.0,
    )


# ===========================================================================
# 1. MUTEX Detection
# ===========================================================================

class TestMutexDetection:
    """Test that the mapper detects mutually exclusive market pairs."""

    def test_mutex_detected_similar_titles(self):
        """Two markets with similar titles on the same platform → MUTEX."""
        mapper = MarketRelationshipMapper(mutex_similarity_threshold=70.0)
        markets = [
            _market("Will Trump win the 2024 presidential election?", 0.55, "m1"),
            _market("Will Biden win the 2024 presidential election?", 0.35, "m2"),
        ]
        rels = mapper.find_relationships(markets)
        mutex_rels = [r for r in rels if r.type == RelationshipType.MUTEX]
        assert len(mutex_rels) >= 1
        assert mutex_rels[0].markets[0].market_id in ("m1", "m2")

    def test_no_mutex_unrelated_markets(self):
        """Unrelated markets should not be flagged as MUTEX."""
        mapper = MarketRelationshipMapper(mutex_similarity_threshold=70.0)
        markets = [
            _market("Will it rain in NYC tomorrow?", 0.30, "rain1"),
            _market("Will Bitcoin hit 100k by 2025?", 0.20, "btc1"),
        ]
        rels = mapper.find_relationships(markets)
        mutex_rels = [r for r in rels if r.type == RelationshipType.MUTEX]
        assert len(mutex_rels) == 0


# ===========================================================================
# 2. NESTED Detection
# ===========================================================================

class TestNestedDetection:
    """Test that the mapper detects nested/conditional relationships."""

    def test_nested_detected_specific_implies_general(self):
        """A more specific market should be nested inside a more general one.

        The child title contains the parent title and is longer/more specific,
        with child price <= parent price (logically consistent).
        """
        mapper = MarketRelationshipMapper(nested_similarity_threshold=60.0)
        markets = [
            _market(
                "Will the Lakers make the NBA playoffs 2025 and win the championship?",
                0.10,
                "lakers_chip",
            ),
            _market(
                "Will the Lakers make the NBA playoffs 2025?",
                0.65,
                "lakers_playoffs",
            ),
        ]
        rels = mapper.find_relationships(markets)
        nested_rels = [r for r in rels if r.type == RelationshipType.NESTED]
        # Championship implies playoffs, so we expect at least one nested relationship
        assert len(nested_rels) >= 1

    def test_no_nested_unrelated(self):
        """Unrelated markets should not produce nested relationships."""
        mapper = MarketRelationshipMapper(nested_similarity_threshold=60.0)
        markets = [
            _market("Will it snow in July?", 0.01, "snow"),
            _market("Will gold hit 3000/oz?", 0.40, "gold"),
        ]
        rels = mapper.find_relationships(markets)
        nested_rels = [r for r in rels if r.type == RelationshipType.NESTED]
        assert len(nested_rels) == 0


# ===========================================================================
# 3. LP Solver — Violation Detection
# ===========================================================================

class TestLPSolverViolations:
    """Test the constraint solver detects known violations."""

    def test_mutex_violation_detected(self):
        """MUTEX markets summing to >1 should be flagged."""
        solver = ConstraintSolver(tolerance=0.01)
        m1 = _market("A wins", 0.60, "a")
        m2 = _market("B wins", 0.55, "b")
        rel = _mutex_rel([m1, m2])  # sum = 1.15 > 1

        violations = solver.check_violations([rel])
        assert len(violations) == 1
        v = violations[0]
        assert v.violation_magnitude == pytest.approx(0.15, abs=0.01)
        assert sum(v.fair_prices) <= 1.0 + 0.01

    def test_mutex_three_markets_violation(self):
        """Three mutually exclusive markets summing to >1."""
        solver = ConstraintSolver(tolerance=0.01)
        markets = [
            _market("X wins", 0.40, "x"),
            _market("Y wins", 0.35, "y"),
            _market("Z wins", 0.30, "z"),
        ]
        rel = _mutex_rel(markets)  # sum = 1.05 > 1

        violations = solver.check_violations([rel])
        assert len(violations) == 1
        assert sum(violations[0].fair_prices) <= 1.0 + 0.01

    def test_nested_violation_detected(self):
        """Child priced higher than parent should be flagged."""
        solver = ConstraintSolver(tolerance=0.01)
        child = _market("Win finals", 0.60, "child")
        parent = _market("Make playoffs", 0.40, "parent")
        rel = _nested_rel(child, parent)

        violations = solver.check_violations([rel])
        assert len(violations) == 1
        v = violations[0]
        assert v.violation_magnitude == pytest.approx(0.20, abs=0.01)
        # Fair prices: child <= parent
        assert v.fair_prices[0] <= v.fair_prices[1] + 0.01

    def test_complementary_violation_detected(self):
        """Complementary markets not summing to 1 should be flagged."""
        solver = ConstraintSolver(tolerance=0.01)
        m1 = _market("Rain tomorrow", 0.60, "rain")
        m2 = _market("Dry tomorrow", 0.55, "dry")
        rel = _complementary_rel([m1, m2])  # sum = 1.15 != 1.0

        violations = solver.check_violations([rel])
        assert len(violations) == 1
        assert violations[0].violation_magnitude == pytest.approx(0.15, abs=0.01)


# ===========================================================================
# 4. LP Solver — Valid Prices (No Violation)
# ===========================================================================

class TestLPSolverValid:
    """Test the solver correctly passes on valid prices."""

    def test_mutex_valid_no_violation(self):
        """MUTEX markets summing to <1 should NOT be flagged."""
        solver = ConstraintSolver(tolerance=0.01)
        m1 = _market("A wins", 0.40, "a")
        m2 = _market("B wins", 0.30, "b")
        rel = _mutex_rel([m1, m2])  # sum = 0.70 < 1

        violations = solver.check_violations([rel])
        assert len(violations) == 0

    def test_nested_valid_no_violation(self):
        """Child priced <= parent should NOT be flagged."""
        solver = ConstraintSolver(tolerance=0.01)
        child = _market("Win finals", 0.25, "child")
        parent = _market("Make playoffs", 0.60, "parent")
        rel = _nested_rel(child, parent)

        violations = solver.check_violations([rel])
        assert len(violations) == 0

    def test_complementary_valid_no_violation(self):
        """Complementary markets summing to ~1 should NOT be flagged."""
        solver = ConstraintSolver(tolerance=0.01)
        m1 = _market("Heads", 0.50, "h")
        m2 = _market("Tails", 0.50, "t")
        rel = _complementary_rel([m1, m2])

        violations = solver.check_violations([rel])
        assert len(violations) == 0


# ===========================================================================
# 5. KL Divergence / Bregman Projection
# ===========================================================================

class TestBregmanProjection:
    """Test KL divergence and Bregman projection computations."""

    def test_kl_divergence_identical(self):
        """KL divergence of identical distributions should be ~0."""
        bp = BregmanProjection()
        rec = bp.compute_trades(
            market_ids=["a", "b"],
            current_prices=[0.5, 0.5],
            fair_prices=[0.5, 0.5],
        )
        assert rec.kl_divergence == pytest.approx(0.0, abs=1e-6)
        assert all(abs(d) < 1e-6 for d in rec.trade_directions)

    def test_kl_divergence_known_value(self):
        """KL divergence with known inputs matches hand-calculated value.

        For a single binary market:
        D_KL(q||p) = q*log(q/p) + (1-q)*log((1-q)/(1-p))
        q=0.7, p=0.5:
        = 0.7*log(0.7/0.5) + 0.3*log(0.3/0.5)
        = 0.7*log(1.4) + 0.3*log(0.6)
        ≈ 0.7*0.3365 + 0.3*(-0.5108)
        ≈ 0.2356 + (-0.1532)
        ≈ 0.0823
        """
        bp = BregmanProjection()
        rec = bp.compute_trades(
            market_ids=["test"],
            current_prices=[0.5],
            fair_prices=[0.7],
        )
        expected_kl = 0.7 * np.log(0.7 / 0.5) + 0.3 * np.log(0.3 / 0.5)
        assert rec.kl_divergence == pytest.approx(expected_kl, abs=1e-4)

    def test_trade_direction_buy_yes(self):
        """When fair > current, trade direction should be positive (buy YES)."""
        bp = BregmanProjection(max_edge_fraction=1.0)
        rec = bp.compute_trades(
            market_ids=["underpriced"],
            current_prices=[0.3],
            fair_prices=[0.6],
        )
        assert rec.trade_directions[0] > 0  # buy YES

    def test_trade_direction_buy_no(self):
        """When fair < current, trade direction should be negative (buy NO)."""
        bp = BregmanProjection(max_edge_fraction=1.0)
        rec = bp.compute_trades(
            market_ids=["overpriced"],
            current_prices=[0.7],
            fair_prices=[0.4],
        )
        assert rec.trade_directions[0] < 0  # buy NO

    def test_bregman_simplex_projection(self):
        """Bregman projection onto simplex preserves ratios and sums to target."""
        prices = np.array([0.6, 0.3, 0.2])  # sum = 1.1
        projected = BregmanProjection.bregman_project_to_simplex(prices, target_sum=1.0)

        assert projected.sum() == pytest.approx(1.0, abs=1e-8)
        # Ratios preserved: p0/p1 should equal proj0/proj1
        assert (projected[0] / projected[1]) == pytest.approx(
            prices[0] / prices[1], abs=1e-6
        )
        assert all(p > 0 for p in projected)

    def test_half_kelly_sizing(self):
        """Default half-Kelly should halve the raw edge."""
        bp = BregmanProjection(max_edge_fraction=0.5)
        rec = bp.compute_trades(
            market_ids=["m1"],
            current_prices=[0.4],
            fair_prices=[0.6],
        )
        # Edge = 0.6 - 0.4 = 0.2, half-Kelly → 0.1
        assert rec.trade_directions[0] == pytest.approx(0.1, abs=1e-6)
        assert rec.edge_per_market[0] == pytest.approx(0.2, abs=1e-6)
