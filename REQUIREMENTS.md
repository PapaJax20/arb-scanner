# Requirements

## Functional
1. **Market Ingestion:** Pull all active binary markets from Polymarket, Kalshi, Manifold, and Metaculus APIs
2. **Event Matching:** NLP-based fuzzy matching to identify equivalent events across platforms (title + description similarity)
3. **Spread Calculation:** Compute cross-platform spread adjusted for platform-specific fees:
   - Polymarket: ~0% fees (maker rebates available)
   - Kalshi: Variable (check per-market, typically 0-7¢ per contract)
   - PredictIt: 10% profit fee + 5% withdrawal fee
   - Manifold: Play money (signal only, no fee adjustment needed)
   - Metaculus: Reputation only (signal only)
4. **Alerting:** Discord webhook notification when any matched pair has spread > configurable threshold (default 4% after fees)
5. **CLI Interface:** Run as long-polling CLI process with configurable interval (default 60s)
6. **Match Confidence:** Score each cross-platform match with confidence level, filter below threshold

## Non-Functional
- Async I/O for concurrent API calls (don't block on slow platforms)
- Rate-limit aware (respect each platform's API limits)
- Structured logging
- Config via .env or YAML
- No external database — in-memory with optional JSON snapshot

## Out of Scope (Phase 1)
- Trading execution
- Position management
- UI/dashboard
- Historical data storage
- Backtesting
