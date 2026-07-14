# Legacy event-study research protocol

This file applies only to the retained event-study MVP. It does not weaken the futures promotion
gate in [strategy-governance.md](strategy-governance.md), which requires a sealed final 12 months
and at least 90 days plus 30 closed trades on both sides of a paired Demo shadow run.

## Claim boundary

The system does not claim that generic sentiment predicts price. It tests whether narrowly defined, source-controlled events have cost-adjusted out-of-sample value at specific horizons.

## Required experiment matrix

1. Event study by event type, source type, asset and 5m/15m/1h/4h/24h horizon.
2. Score deciles and forward-return monotonicity.
3. Ablations for source quality, bot score and on-chain confirmation.
4. Strategy simulation with fees, slippage and entry/exit rules.
5. Robustness across assets, windows, thresholds and three cost scenarios.
6. A separate four-to-eight-week online paper run.

## Leakage controls

- Split train, calibration and test periods by time; reserve the final 20–30% as strict out-of-sample data.
- Use the original publication timestamp and retain ingestion timestamp separately.
- Never use engagement, labels or price observations that were unavailable at decision time.
- Freeze scoring weights before evaluating the holdout.
- Report all attempted variants, including negative results.

## Minimum evidence

- Target at least 1,000 total events and 200–300 events for each prominently reported bucket.
- Report mean cost-adjusted excess return, bootstrap confidence interval and HAC/Newey-West inference when horizons overlap.
- Report out-of-sample Sharpe, profit factor, maximum drawdown, hit rate and payoff ratio.
- Treat fewer than 30 paper orders as insufficient evidence; the software reports that status explicitly.
