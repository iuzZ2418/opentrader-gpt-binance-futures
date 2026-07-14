# Data dictionary

The repository has two deliberately separate persistence domains. The retained A-share workbench
uses a local SQLite repository. The futures system uses PostgreSQL as its production fact and audit
store; SQLite implements the same audit schema only for tests. Redis carries leases, controls, and
ephemeral notifications. Only producers with an explicit PostgreSQL-backed outbox (currently the
external-evidence pipeline) can republish those notifications; Redis is never an order, raw-market,
or performance fact source.

## Futures audit ledger

| Entity | Purpose | Important fields |
|---|---|---|
| `trade_candidates` | Point-in-time bounded strategy proposal | trace, strategy version, direction, max quantity/risk, features, evidence IDs, expiry |
| `external_evidence` | Append-only source versions, exact decision inputs, and paper funding-coverage watermarks | stable evidence ID, version, prior version, observed/occurred/deleted time, content hash, payload; normalized closed OHLCV/quote/derivatives windows; per-episode funding coverage |
| `candidate_evidence_links` | Exact evidence lineage used by a candidate | trace, candidate, immutable evidence-record ID, role |
| `llm_decisions` | Strict GPT lifecycle decision | action, multiplier, confidence, evidence, thesis, invalidations, model/prompt/response/latency |
| `position_theses` | Append-only human-like position reasoning | position, version, prior thesis, evidence, add count, PnL/R, invalidations |
| `risk_decisions` | Deterministic final gate | allow/resize/reject/exit, approved quantity, reason codes, limit snapshot |
| `venue_orders` | Idempotent exchange intent/current state | client/exchange IDs, side/type, quantity/price, reduce-only, status |
| `venue_order_events` | Append-only order state machine | sequence, source event, executed quantity, average price, observed time |
| `venue_fills` | Authoritative fills | external fill ID, price, quantity, fee asset, realized PnL, fill time |
| `venue_accounting_events` | Funding and other exchange income facts | external income ID, type, asset, amount, transaction time, trade ID |
| `venue_accounting_attributions` | Point-in-time funding attribution result | accounting event, trace/order, attributed or unresolved, reason |
| `venue_fee_conversions` | Exact non-USDT fee conversion | fill, assets, rate, exact effective time, source record |
| `account_snapshots` | Reconciled account state | equity/cash, gross/net exposure, daily PnL, drawdown, positions, source/time |
| `counterfactual_outcomes` | 1h/4h/24h result for every candidate | realized return, regret, calibration error, source reliability |

## Strategy research and governance

| Entity | Purpose | Important fields |
|---|---|---|
| `strategy_research_runs` | Immutable Responses API research call | model, prompt, response, latency, evidence/sources, hypothesis, rationale, failure modes |
| `strategy_specs` | Bounded champion/challenger configuration | version, parent, status, approved parameters, prompt version |
| `backtest_runs` | Walk-forward and sealed-holdout evidence | costs, 2× stress, drawdown, DSR, PBO, concentration, trades, validation hashes |
| `shadow_results` | Demo paired-shadow evidence | elapsed days, closed trades, net return, drawdown, stress and concentration |
| `promotion_records` | Promotion decision and reasons | champion/challenger, exact backtest/shadow rows, eligibility, evaluation |
| `liquidity_observations` | Daily point-in-time universe inputs | listing date, turnover, spread, ±20 bps depth, expected order notional |
| `universe_selections` | Weekly hysteretic universe snapshot | week, as-of time, selected symbols, coverage/fallback reason |

`TradeOutcome` is derived only from authoritative fills, attributed funding, and exact fee conversion
records; it is not a permissive mutable table. For `internal-paper`, every closed episode must also
have a latest, non-deleted `paper-funding-coverage-v1` evidence version whose inclusive public-history
watermark reaches the exact audited close. The coverage evidence-record ID becomes part of the
outcome lineage. Missing, stale, or unsealed coverage prevents performance calculation and automatic
promotion, including episodes with no funding event. The current audit schema is checksum-migrated
through version 10, and production tables that represent facts are protected against update/delete.

The paper worker polls public marks for audited protective stops on an independent one-second loop.
Marking, stop enforcement, funding synchronization, maintenance, and strategy-cycle portfolio/order
mutations share one re-entrant runtime mutex. A protective close seals its episode funding coverage
before releasing that mutex. This is internal paper protection, not an exchange-hosted stop and not a
real-time latency guarantee when another serialized operation is already running.

Imported paired-shadow costs use a strict `external_evidence` subtype whose payload `schema` is
`paired-shadow-cost-v1`. The three distinct fee/slippage/funding record IDs carry `cost_type`,
`trade_id`, `episode_id`, `trace_id`, `symbol`, `strategy_version`, `closed_at`, and the corresponding
`amount`. The referenced trace must also exist in a non-external-evidence audit fact. Merely storing an
ordinary external-evidence row with the requested ID is not valid shadow accounting lineage.

For each candidate, the ledger persists the complete normalized closed-bar windows actually used,
the approval quote and derivatives snapshot, plus immutable hashes and links. It still does not
claim to be a complete point-in-time order-book, filters, listing/delisting, or research data lake.
The external simulator that submits a research manifest remains responsible for its licensed full
point-in-time dataset; the validator binds results to digests and evidence IDs but does not
reconstruct that dataset from Redis or this decision-level ledger.

The champion pointer and bounded `StrategySpec` history are also persisted atomically in
`strategy_registry.json`; PostgreSQL retains the immutable evidence used to justify each change.

## Retained A-share workbench

| Entity | Purpose |
|---|---|
| `source_entities`, `documents`, `raw_posts`, `raw_news`, `exchange_announcements` | Source identities and normalized public text |
| `canonical_assets`, `onchain_events`, `extracted_events`, `signal_scores` | Asset mapping, structured evidence, and reproducible research scores |
| `market_prices`, `paper_orders`, `paper_fills`, `positions`, `equity_curve` | Legacy local paper-research state |

These legacy rows never authorize or reconcile a Futures order.
