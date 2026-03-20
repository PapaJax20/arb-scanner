# Arb Scanner

Cross-platform prediction market arbitrage scanner.

## Goal
Detect real-time pricing discrepancies across prediction market platforms (Polymarket, Kalshi) with calibration signals from free platforms (Manifold, Metaculus). Alert when cross-platform spreads exceed profitable thresholds after fees.

## Owner
Jonathan Moon (@PapaJax20)

## Status
Phase 1 — Scanner MVP (in progress)

## Stack
- Python 3.11+
- asyncio + aiohttp for concurrent API polling
- NLP matching (rapidfuzz for fuzzy string matching)
- Discord webhook for alerts

## Done Criteria (Phase 1)
- [ ] Pull all active markets from Polymarket, Kalshi, Manifold, Metaculus APIs
- [ ] Cross-platform event matching engine (fuzzy NLP on titles/descriptions)
- [ ] Real-time spread calculator with fee adjustment per platform
- [ ] Alert to Discord when spread > 4% after fees
- [ ] CLI runner with configurable poll interval
- [ ] Smoke test on live market data
