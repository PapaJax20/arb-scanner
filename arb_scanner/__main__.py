"""Entry point — async main loop for arb-scanner."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

import aiohttp

from arb_scanner import config
from arb_scanner.alerts.discord import DiscordAlerter
from arb_scanner.analysis.spread import SpreadCalculator
from arb_scanner.clients.kalshi import KalshiClient
from arb_scanner.clients.manifold import ManifoldClient
from arb_scanner.clients.metaculus import MetaculusClient
from arb_scanner.clients.polymarket import PolymarketClient
from arb_scanner.constraints.engine import ConstraintSolver
from arb_scanner.constraints.mapper import MarketRelationshipMapper
from arb_scanner.constraints.sizing import BregmanProjection
from arb_scanner.execution.alerts import PaperTradeAlerter
from arb_scanner.execution.paper_trader import PaperTrader
from arb_scanner.execution.position_tracker import PositionTracker
from arb_scanner.matching.engine import MatchingEngine
from arb_scanner.models import Market, SpreadResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("arb_scanner")

_shutdown = asyncio.Event()

# Module-level references for paper trading (initialized in main_loop)
_position_tracker: PositionTracker | None = None


def _handle_signal(sig: int, frame) -> None:
    logger.info("Received signal %s — shutting down", signal.Signals(sig).name)
    _shutdown.set()


async def fetch_all_markets(session: aiohttp.ClientSession) -> dict[str, list[Market]]:
    """Fetch markets from all platforms concurrently."""
    clients = {
        "polymarket": PolymarketClient(session),
        "kalshi": KalshiClient(session),
        "manifold": ManifoldClient(session),
        "metaculus": MetaculusClient(session),
    }

    tasks = {name: asyncio.create_task(client.fetch_markets()) for name, client in clients.items()}

    results: dict[str, list[Market]] = {}
    for name, task in tasks.items():
        try:
            results[name] = await task
        except Exception as e:
            logger.error("Failed to fetch %s markets: %s", name, e)
            results[name] = []

    total = sum(len(v) for v in results.values())
    logger.info(
        "Fetched %d total markets: %s",
        total,
        ", ".join(f"{k}={len(v)}" for k, v in results.items()),
    )
    return results


async def run_scan(
    session: aiohttp.ClientSession,
    single_shot: bool = False,
    paper_trader: PaperTrader | None = None,
    position_tracker: PositionTracker | None = None,
    paper_alerter: PaperTradeAlerter | None = None,
    scan_count: int = 0,
) -> list[SpreadResult]:
    """Run one full scan cycle."""
    markets = await fetch_all_markets(session)

    engine = MatchingEngine(threshold=config.MIN_MATCH_CONFIDENCE)
    calculator = SpreadCalculator()
    alerter = DiscordAlerter()

    all_results: list[SpreadResult] = []

    # Primary arb: Polymarket ↔ Kalshi
    poly_kalshi = engine.find_matches(markets["polymarket"], markets["kalshi"])
    arb_results = calculator.analyze(poly_kalshi)
    all_results.extend(arb_results)

    # Calibration: real-money vs signal platforms
    for real_key in ("polymarket", "kalshi"):
        for signal_key in ("manifold", "metaculus"):
            if markets[real_key] and markets[signal_key]:
                pairs = engine.find_matches(markets[real_key], markets[signal_key])
                cal_results = calculator.analyze(pairs)
                all_results.extend(cal_results)

    # Intra-platform constraint arb: check Polymarket markets for constraint violations
    constraint_violations = []
    if markets["polymarket"]:
        rel_mapper = MarketRelationshipMapper()
        constraint_solver = ConstraintSolver()
        bregman = BregmanProjection()

        relationships = rel_mapper.find_relationships(markets["polymarket"])
        violations = constraint_solver.check_violations(relationships)

        for v in violations:
            trade = bregman.compute_trades(
                market_ids=[m.market_id for m in v.markets],
                current_prices=v.current_prices,
                fair_prices=v.fair_prices,
            )
            constraint_violations.append((v, trade))
            logger.info(
                "  CONSTRAINT %s | magnitude=%.4f | KL=%.6f | %s",
                v.relationship.type.value.upper(),
                v.violation_magnitude,
                trade.kl_divergence,
                v.description,
            )

    # Report
    actionable = [r for r in all_results if r.is_actionable]
    calibration = [r for r in all_results if r.is_calibration]
    logger.info(
        "Scan complete: %d actionable arbs, %d calibration divergences",
        len(actionable),
        len(calibration),
    )

    for r in actionable:
        logger.info(
            "  ARB %.1f%% net | %s ↔ %s | %s",
            r.net_spread_pct,
            r.pair.market_a.platform.value,
            r.pair.market_b.platform.value,
            r.pair.market_a.title[:80],
        )

    for r in calibration:
        logger.info(
            "  CAL %.1f%% | %s ↔ %s | %s",
            r.deviation_pct,
            r.pair.market_a.platform.value,
            r.pair.market_b.platform.value,
            r.pair.market_a.title[:80],
        )

    if not single_shot:
        await alerter.send_alerts(all_results)

    # ---------------------------------------------------------------
    # Paper Trading Execution (Phase 2)
    # ---------------------------------------------------------------
    if config.PAPER_TRADE_ENABLED and paper_trader and position_tracker and paper_alerter:
        # Execute cross-platform arbs
        for r in actionable:
            trades = paper_trader.execute_arb(r.pair, r)
            for t in trades:
                await paper_alerter.send_fill_alert(t)

        # Execute constraint arbs (magnitude > 0.02 threshold)
        for v, trade_rec in constraint_violations:
            if v.violation_magnitude > 0.02:
                trades = paper_trader.execute_constraint_arb(v, trade_rec)
                for t in trades:
                    await paper_alerter.send_fill_alert(t)

        # Mark-to-market: build price map from all fetched markets
        price_map: dict[str, float] = {}
        for platform_markets in markets.values():
            for m in platform_markets:
                price_map[m.market_id] = m.yes_price

        position_tracker.mark_to_market(price_map)

        # Periodic P&L summary
        if scan_count > 0 and scan_count % config.PNL_SUMMARY_INTERVAL == 0:
            summary = position_tracker.summary()
            logger.info(
                "[PAPER] P&L Summary — balance=$%.2f | open=%d | realized=$%.2f | unrealized=$%.2f | win_rate=%.0f%%",
                summary["total_balance"],
                summary["open_positions"],
                summary["realized_pnl"],
                summary["unrealized_pnl"],
                summary["win_rate"] * 100,
            )
            await paper_alerter.send_pnl_summary(summary)

        # Persist state after each cycle
        position_tracker.save()

    return all_results


async def main_loop() -> None:
    """Main scanning loop with graceful shutdown."""
    global _position_tracker

    logger.info("arb-scanner starting — interval=%ds, spread_threshold=%.1f%%", config.SCAN_INTERVAL, config.MIN_SPREAD_THRESHOLD)

    # Initialize paper trading
    paper_trader: PaperTrader | None = None
    position_tracker: PositionTracker | None = None
    paper_alerter: PaperTradeAlerter | None = None

    if config.PAPER_TRADE_ENABLED:
        position_tracker = PositionTracker.load()
        _position_tracker = position_tracker
        paper_trader = PaperTrader(portfolio=position_tracker.portfolio)
        paper_alerter = PaperTradeAlerter()
        logger.info("[PAPER] Paper trading enabled — loaded portfolio")

    scan_count = 0
    timeout = aiohttp.ClientTimeout(total=config.API_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while not _shutdown.is_set():
            try:
                scan_count += 1
                await run_scan(
                    session,
                    paper_trader=paper_trader,
                    position_tracker=position_tracker,
                    paper_alerter=paper_alerter,
                    scan_count=scan_count,
                )
            except Exception as e:
                logger.error("Scan cycle failed: %s", e, exc_info=True)

            try:
                await asyncio.wait_for(_shutdown.wait(), timeout=config.SCAN_INTERVAL)
                break  # shutdown signaled
            except asyncio.TimeoutError:
                pass  # timeout = time for next scan

    # Graceful shutdown: persist positions
    if position_tracker:
        logger.info("[PAPER] Persisting positions before shutdown...")
        position_tracker.save()

    logger.info("arb-scanner stopped")


async def single_scan() -> list[SpreadResult]:
    """Run a single scan iteration (for smoke testing)."""
    timeout = aiohttp.ClientTimeout(total=config.API_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return await run_scan(session, single_shot=True)


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    if "--once" in sys.argv:
        asyncio.run(single_scan())
    else:
        asyncio.run(main_loop())


if __name__ == "__main__":
    main()
