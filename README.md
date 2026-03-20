# arb-scanner

Cross-platform prediction market arbitrage scanner. Detects pricing discrepancies across Polymarket, Kalshi, Manifold, and Metaculus in real-time.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your Discord webhook URL and any API keys

# Run
python -m arb_scanner
```

## What It Does

Pulls active markets from multiple prediction market platforms, matches equivalent events across platforms using NLP similarity, and alerts via Discord when cross-platform spreads exceed a configurable threshold (default: 4% after fees).

## Platforms

| Platform | Type | API | Use |
|----------|------|-----|-----|
| Polymarket | Real money (USDC) | REST + WebSocket | Primary arb leg |
| Kalshi | Real money (USD) | REST + WebSocket | Primary arb leg |
| Manifold | Play money | REST | Calibration signal |
| Metaculus | Reputation | REST | Calibration signal |

## Architecture

```
arb_scanner/
├── clients/          # Platform API clients
├── matching/         # Cross-platform event matching (NLP)
├── analysis/         # Spread calculation + fee adjustment
├── alerts/           # Discord webhook alerts
├── config.py         # Configuration
└── __main__.py       # Entry point
```

## License

Private — Trading Desk project
