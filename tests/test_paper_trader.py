"""Tests for the paper trading engine (Phase 2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arb_scanner.constraints.engine import ConstraintViolation
from arb_scanner.constraints.mapper import MarketRelationship, RelationshipType
from arb_scanner.constraints.sizing import BregmanProjection
from arb_scanner.execution.paper_trader import PaperTrader
from arb_scanner.execution.position_tracker import PositionTracker
from arb_scanner.models import (
    Market,
    MatchedPair,
    PaperTrade,
    Platform,
    Portfolio,
    SpreadResult,
    TradeStatus,
)


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


def _arb_pair() -> tuple[MatchedPair, SpreadResult]:
    """Create a realistic arb pair with actionable spread."""
    a = _market("Will BTC hit 100k by Dec 2025?", 0.40, "btc_poly", Platform.POLYMARKET)
    b = _market("Will BTC hit 100k by Dec 2025?", 0.55, "btc_kalshi", Platform.KALSHI)
    pair = MatchedPair(market_a=a, market_b=b, similarity_score=95.0)
    spread = SpreadResult(
        pair=pair,
        spread_pct=15.0,
        fees_pct=7.0,
        net_spread_pct=8.0,
        direction="Buy YES on polymarket @ 0.40, Buy NO on kalshi @ 0.45",
        is_actionable=True,
    )
    return pair, spread


def _fresh_portfolio(balance: float = 10000.0) -> Portfolio:
    return Portfolio(
        balances={"polymarket": balance, "kalshi": balance},
    )


# ===========================================================================
# 1. Trade Execution
# ===========================================================================

class TestTradeExecution:
    """Test that paper trades execute correctly with simulated fills."""

    def test_execute_arb_creates_two_legs(self):
        """An actionable arb should create exactly 2 trade legs."""
        portfolio = _fresh_portfolio()
        trader = PaperTrader(portfolio, slippage_pct=0.005, max_position_pct=0.05)
        pair, spread = _arb_pair()

        trades = trader.execute_arb(pair, spread)
        assert len(trades) == 2
        assert all(isinstance(t, PaperTrade) for t in trades)
        assert all(t.status == TradeStatus.OPEN for t in trades)

    def test_execute_arb_deducts_balance(self):
        """Executing trades should deduct from platform balances."""
        portfolio = _fresh_portfolio(10000.0)
        trader = PaperTrader(portfolio, slippage_pct=0.005, max_position_pct=0.05)
        pair, spread = _arb_pair()

        trader.execute_arb(pair, spread)

        # At least one platform should have less than starting balance
        assert portfolio.balances["polymarket"] < 10000.0 or portfolio.balances["kalshi"] < 10000.0

    def test_non_actionable_spread_skipped(self):
        """Spreads that aren't actionable should produce no trades."""
        portfolio = _fresh_portfolio()
        trader = PaperTrader(portfolio, slippage_pct=0.005, max_position_pct=0.05)
        pair, spread = _arb_pair()
        spread.is_actionable = False

        trades = trader.execute_arb(pair, spread)
        assert len(trades) == 0


# ===========================================================================
# 2. Slippage Application
# ===========================================================================

class TestSlippage:
    """Test that slippage is correctly applied to fill prices."""

    def test_slippage_increases_fill_price(self):
        """Buying YES should fill at a worse (higher) price due to slippage."""
        portfolio = _fresh_portfolio()
        trader = PaperTrader(portfolio, slippage_pct=0.01, max_position_pct=0.05)
        pair, spread = _arb_pair()

        trades = trader.execute_arb(pair, spread)
        # At least one trade should have entry_price > raw market price
        poly_trade = [t for t in trades if t.platform == Platform.POLYMARKET]
        if poly_trade:
            # Raw YES price was 0.40, with 1% slippage → 0.404
            assert poly_trade[0].entry_price > 0.40

    def test_zero_slippage(self):
        """With zero slippage, fill price should equal market price."""
        portfolio = _fresh_portfolio()
        trader = PaperTrader(portfolio, slippage_pct=0.0, max_position_pct=0.05)
        pair, spread = _arb_pair()

        trades = trader.execute_arb(pair, spread)
        poly_trade = [t for t in trades if t.platform == Platform.POLYMARKET]
        if poly_trade:
            assert poly_trade[0].entry_price == pytest.approx(0.40, abs=0.001)


# ===========================================================================
# 3. Max Position Cap
# ===========================================================================

class TestMaxPositionCap:
    """Test that position sizing respects the max position cap."""

    def test_trade_size_capped(self):
        """Trade size should never exceed max_position_pct * balance."""
        portfolio = _fresh_portfolio(10000.0)
        trader = PaperTrader(portfolio, slippage_pct=0.0, max_position_pct=0.05)
        pair, spread = _arb_pair()

        trades = trader.execute_arb(pair, spread)
        for trade in trades:
            platform_balance = 10000.0  # starting balance
            max_size = platform_balance * 0.05
            assert trade.size <= max_size + 0.01  # small float tolerance

    def test_insufficient_balance_rejected(self):
        """Trades that exceed available balance should be skipped."""
        portfolio = _fresh_portfolio(1.0)  # Only $1 per platform
        trader = PaperTrader(portfolio, slippage_pct=0.0, max_position_pct=0.99)
        pair, spread = _arb_pair()

        # With $1 balance and high edge, the kelly size may exceed $1
        # but cost = size * price must fit within $1
        trades = trader.execute_arb(pair, spread)
        for trade in trades:
            assert trade.simulated_cost <= 1.0 + 0.01


# ===========================================================================
# 4. Kelly Sizing Integration
# ===========================================================================

class TestKellySizing:
    """Test that half-Kelly sizing produces reasonable trade sizes."""

    def test_kelly_produces_nonzero_size(self):
        """With positive edge, Kelly should produce a nonzero trade size."""
        portfolio = _fresh_portfolio(10000.0)
        trader = PaperTrader(portfolio, slippage_pct=0.0, max_position_pct=0.10)
        pair, spread = _arb_pair()

        trades = trader.execute_arb(pair, spread)
        assert len(trades) > 0
        assert all(t.size > 0 for t in trades)

    def test_higher_edge_gives_larger_size(self):
        """A larger spread should produce larger trade sizes (half-Kelly)."""
        pair_small, spread_small = _arb_pair()
        spread_small.net_spread_pct = 5.0

        pair_big, spread_big = _arb_pair()
        spread_big.net_spread_pct = 15.0

        portfolio_small = _fresh_portfolio(10000.0)
        trader_small = PaperTrader(portfolio_small, slippage_pct=0.0, max_position_pct=0.50)
        trades_small = trader_small.execute_arb(pair_small, spread_small)

        portfolio_big = _fresh_portfolio(10000.0)
        trader_big = PaperTrader(portfolio_big, slippage_pct=0.0, max_position_pct=0.50)
        trades_big = trader_big.execute_arb(pair_big, spread_big)

        if trades_small and trades_big:
            total_small = sum(t.size for t in trades_small)
            total_big = sum(t.size for t in trades_big)
            assert total_big > total_small


# ===========================================================================
# 5. Position Tracking & P&L
# ===========================================================================

class TestPositionTracking:
    """Test position tracking, mark-to-market, and settlement."""

    def test_mark_to_market_updates_unrealized_pnl(self):
        """Mark-to-market should update unrealized P&L based on current prices."""
        portfolio = _fresh_portfolio()
        trade = PaperTrade(
            trade_id="PT-TEST001",
            platform=Platform.POLYMARKET,
            market_id="btc_poly",
            market_title="BTC 100k",
            direction="YES",
            entry_price=0.40,
            size=100.0,
            simulated_cost=40.0,
        )
        portfolio.open_trades.append(trade)

        tracker = PositionTracker(portfolio)

        # Price went up → profit
        unrealized = tracker.mark_to_market({"btc_poly": 0.50})
        assert unrealized == pytest.approx(10.0, abs=0.01)  # (0.50 - 0.40) * 100
        assert portfolio.total_unrealized_pnl == pytest.approx(10.0, abs=0.01)

    def test_settlement_calculates_realized_pnl(self):
        """Settling a trade should correctly calculate realized P&L."""
        portfolio = _fresh_portfolio()
        portfolio.balances["polymarket"] = 9960.0  # after buying at cost $40
        trade = PaperTrade(
            trade_id="PT-SETTLE01",
            platform=Platform.POLYMARKET,
            market_id="btc_poly",
            market_title="BTC 100k",
            direction="YES",
            entry_price=0.40,
            size=100.0,
            simulated_cost=40.0,
        )
        portfolio.open_trades.append(trade)

        tracker = PositionTracker(portfolio)
        result = tracker.settle_trade("PT-SETTLE01", settlement_price=1.0)

        assert result is not None
        assert result.pnl == pytest.approx(60.0, abs=0.01)  # (1.0 - 0.40) * 100
        assert len(portfolio.open_trades) == 0
        assert len(portfolio.settled_trades) == 1
        assert portfolio.total_realized_pnl == pytest.approx(60.0, abs=0.01)
        # Balance should be restored: 9960 + 40 (cost) + 60 (pnl) = 10060
        assert portfolio.balances["polymarket"] == pytest.approx(10060.0, abs=0.01)

    def test_settlement_losing_trade(self):
        """Settling a losing trade should produce negative P&L."""
        portfolio = _fresh_portfolio()
        portfolio.balances["polymarket"] = 9960.0
        trade = PaperTrade(
            trade_id="PT-LOSE01",
            platform=Platform.POLYMARKET,
            market_id="btc_poly",
            market_title="BTC 100k",
            direction="YES",
            entry_price=0.40,
            size=100.0,
            simulated_cost=40.0,
        )
        portfolio.open_trades.append(trade)

        tracker = PositionTracker(portfolio)
        result = tracker.settle_trade("PT-LOSE01", settlement_price=0.0)

        assert result is not None
        assert result.pnl == pytest.approx(-40.0, abs=0.01)  # (0.0 - 0.40) * 100
        # Balance: 9960 + 40 + (-40) = 9960
        assert portfolio.balances["polymarket"] == pytest.approx(9960.0, abs=0.01)

    def test_settle_nonexistent_trade_returns_none(self):
        """Settling a trade that doesn't exist should return None."""
        portfolio = _fresh_portfolio()
        tracker = PositionTracker(portfolio)
        result = tracker.settle_trade("NONEXISTENT", 1.0)
        assert result is None


# ===========================================================================
# 6. Portfolio State Persistence
# ===========================================================================

class TestPersistence:
    """Test save/load of portfolio state to JSON."""

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        """Portfolio should survive a save/load cycle."""
        persist_file = tmp_path / "positions.json"

        portfolio = _fresh_portfolio(5000.0)
        trade = PaperTrade(
            trade_id="PT-PERSIST01",
            platform=Platform.POLYMARKET,
            market_id="btc_poly",
            market_title="BTC 100k",
            direction="YES",
            entry_price=0.40,
            size=100.0,
            simulated_cost=40.0,
        )
        portfolio.open_trades.append(trade)
        portfolio.total_realized_pnl = 25.50

        tracker = PositionTracker(portfolio, persist_path=persist_file)
        tracker.save()

        assert persist_file.exists()

        # Load back
        loaded = PositionTracker.load(persist_path=persist_file)
        assert loaded.portfolio.balances["polymarket"] == pytest.approx(5000.0)
        assert len(loaded.portfolio.open_trades) == 1
        assert loaded.portfolio.open_trades[0].trade_id == "PT-PERSIST01"
        assert loaded.portfolio.total_realized_pnl == pytest.approx(25.50)

    def test_load_creates_fresh_when_no_file(self, tmp_path: Path):
        """Loading from nonexistent path should create a fresh portfolio."""
        persist_file = tmp_path / "nonexistent.json"
        tracker = PositionTracker.load(persist_path=persist_file)
        assert tracker.portfolio.balances["polymarket"] > 0
        assert len(tracker.portfolio.open_trades) == 0

    def test_load_handles_corrupt_json(self, tmp_path: Path):
        """Loading corrupt JSON should fall back to fresh portfolio."""
        persist_file = tmp_path / "corrupt.json"
        persist_file.write_text("NOT VALID JSON {{{")
        tracker = PositionTracker.load(persist_path=persist_file)
        assert len(tracker.portfolio.open_trades) == 0


# ===========================================================================
# 7. Portfolio Summary
# ===========================================================================

class TestPortfolioSummary:
    """Test portfolio summary calculations."""

    def test_summary_win_rate(self):
        """Win rate should reflect settled trade outcomes."""
        portfolio = _fresh_portfolio()
        # 2 wins, 1 loss
        for i, pnl in enumerate([10.0, 5.0, -3.0]):
            t = PaperTrade(
                trade_id=f"PT-S{i}",
                platform=Platform.POLYMARKET,
                market_id=f"m{i}",
                market_title=f"Market {i}",
                direction="YES",
                entry_price=0.50,
                size=100.0,
                simulated_cost=50.0,
                status=TradeStatus.SETTLED,
                realized_pnl=pnl,
            )
            portfolio.settled_trades.append(t)
        portfolio.total_realized_pnl = 12.0

        tracker = PositionTracker(portfolio)
        summary = tracker.summary()

        assert summary["win_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert summary["realized_pnl"] == pytest.approx(12.0)
        assert summary["settled_positions"] == 3


# ===========================================================================
# 8. Constraint Arb Execution
# ===========================================================================

class TestConstraintArbExecution:
    """Test constraint arbitrage trade execution."""

    def test_constraint_arb_executes(self):
        """Constraint violations should produce trades when edge exists."""
        portfolio = _fresh_portfolio()
        trader = PaperTrader(portfolio, slippage_pct=0.0, max_position_pct=0.05)

        m1 = _market("A wins election", 0.60, "m1")
        m2 = _market("B wins election", 0.55, "m2")

        violation = ConstraintViolation(
            markets=[m1, m2],
            relationship=MarketRelationship(
                type=RelationshipType.MUTEX,
                markets=[m1, m2],
                confidence=90.0,
            ),
            violation_magnitude=0.15,
            current_prices=[0.60, 0.55],
            fair_prices=[0.52, 0.48],
        )

        bregman = BregmanProjection(max_edge_fraction=0.5)
        trade_rec = bregman.compute_trades(
            market_ids=["m1", "m2"],
            current_prices=[0.60, 0.55],
            fair_prices=[0.52, 0.48],
        )

        trades = trader.execute_constraint_arb(violation, trade_rec)
        # Should produce trades (direction depends on edge sign)
        assert len(trades) >= 1
        assert all(t.size > 0 for t in trades)
