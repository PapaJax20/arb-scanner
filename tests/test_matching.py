"""Unit tests for the matching engine."""

import pytest

from arb_scanner.matching.engine import MatchingEngine, normalize_title
from arb_scanner.models import Market, Platform


def _make_market(platform: Platform, title: str, yes_price: float = 0.5) -> Market:
    return Market(
        platform=platform,
        market_id=f"test-{title[:10]}",
        title=title,
        yes_price=yes_price,
        no_price=1.0 - yes_price,
    )


class TestNormalizeTitle:
    def test_lowercase(self):
        assert normalize_title("Will Trump Win?") == "trump win"

    def test_strips_punctuation(self):
        assert normalize_title("Biden vs. Trump: 2024?") == "biden vs trump 2024"

    def test_strips_articles(self):
        result = normalize_title("Will the Fed raise rates?")
        assert "the" not in result.split()
        assert "will" not in result.split()

    def test_collapses_whitespace(self):
        result = normalize_title("  too   many   spaces  ")
        assert "  " not in result


class TestMatchingEngine:
    def test_exact_match(self):
        engine = MatchingEngine(threshold=70)
        poly = [_make_market(Platform.POLYMARKET, "Will Trump win the 2024 election?", 0.6)]
        kalshi = [_make_market(Platform.KALSHI, "Will Trump win the 2024 election?", 0.55)]
        matches = engine.find_matches(poly, kalshi)
        assert len(matches) == 1
        assert matches[0].similarity_score == 100.0

    def test_fuzzy_match(self):
        engine = MatchingEngine(threshold=70)
        poly = [_make_market(Platform.POLYMARKET, "Will Trump win the 2024 presidential election?")]
        kalshi = [_make_market(Platform.KALSHI, "Trump wins 2024 presidential election")]
        matches = engine.find_matches(poly, kalshi)
        assert len(matches) == 1
        assert matches[0].similarity_score >= 70

    def test_no_match_below_threshold(self):
        engine = MatchingEngine(threshold=90)
        poly = [_make_market(Platform.POLYMARKET, "Will Bitcoin reach $100k?")]
        kalshi = [_make_market(Platform.KALSHI, "Fed interest rate decision March")]
        matches = engine.find_matches(poly, kalshi)
        assert len(matches) == 0

    def test_best_match_selected(self):
        engine = MatchingEngine(threshold=70)
        poly = [_make_market(Platform.POLYMARKET, "Will Biden drop out of the race?")]
        kalshi = [
            _make_market(Platform.KALSHI, "Will Biden drop out of the 2024 race?"),
            _make_market(Platform.KALSHI, "Will Trump be convicted?"),
        ]
        matches = engine.find_matches(poly, kalshi)
        assert len(matches) == 1
        assert "biden" in matches[0].market_b.title.lower()

    def test_empty_lists(self):
        engine = MatchingEngine(threshold=70)
        assert engine.find_matches([], []) == []

    def test_same_platform_skipped(self):
        engine = MatchingEngine(threshold=70)
        poly = [
            _make_market(Platform.POLYMARKET, "Will X happen?"),
            _make_market(Platform.POLYMARKET, "Will X happen?"),
        ]
        matches = engine.find_matches(poly, poly)
        assert len(matches) == 0

    def test_cross_platform_manifold(self):
        engine = MatchingEngine(threshold=70)
        poly = [_make_market(Platform.POLYMARKET, "Will AI cause a major incident in 2025?", 0.3)]
        manifold = [_make_market(Platform.MANIFOLD, "Will AI cause a major incident in 2025?", 0.25)]
        matches = engine.find_matches(poly, manifold)
        assert len(matches) == 1

    def test_confidence_score_range(self):
        engine = MatchingEngine(threshold=50)
        poly = [_make_market(Platform.POLYMARKET, "Some question about markets")]
        kalshi = [_make_market(Platform.KALSHI, "A different question about markets")]
        matches = engine.find_matches(poly, kalshi)
        for m in matches:
            assert 0 <= m.similarity_score <= 100
