# Polymarekt Arbitrage Trading Strategies

This polymarket arbitrage trading bot utilizes the five arbitrage strategies implemented in the bot.

## Overview

The bot uses multiple strategies to identify arbitrage opportunities on Polymarket prediction markets. Each strategy targets different market inefficiencies and operates across multiple timeframes (5m, 15m, 1h).

## For consulting with strategies and purchase, contact me at [@roswellecho](https://t.me/roswellecho)

## Strategy 1: Intra-Market Arbitrage

**Type:** Risk-free arbitrage  
**Priority:** Highest  
**Timeframe:** All

### How It Works

In binary prediction markets, YES and NO tokens should always sum to $1.00. When the combined price of YES + NO is less than $1.00 (after fees), you can buy both outcomes and guarantee a profit.

### Example

- YES token: $0.45
- NO token: $0.50
- Total cost: $0.95
- Fees (2%): $0.019
- Gas: $0.014
- Total with fees: $0.983
- Profit: $1.00 - $0.983 = $0.017 per share (1.7% return)

### Execution

Buy both YES and NO tokens simultaneously. On market resolution, one token pays $1.00, guaranteeing profit regardless of outcome.

### Configuration

- `min_spread_pct`: Minimum profit % required (default: 1.5%)
- `max_position_usd`: Maximum position size (default: $500)
- `fee_pct`: Polymarket fee percentage (default: 2.0%)

---

## Strategy 2: Combinatorial Arbitrage

**Type:** Risk-free arbitrage  
**Priority:** High  
**Timeframe:** 1h

### How It Works

For markets with multiple outcomes (e.g., price ranges), the sum of all outcome prices should equal $1.00. When the total is less than $1.00, buy all outcomes for guaranteed profit.

### Example

Market: "BTC price range at end of month"
- $60k-$70k: $0.30
- $70k-$80k: $0.35
- $80k-$90k: $0.20
- $90k+: $0.10
- Total: $0.95
- After fees: $0.97
- Profit: $0.03 per share (3.1% return)

### Execution

Buy all outcome tokens in the market. One outcome will resolve to $1.00, ensuring profit.

### Configuration

- `min_deviation_pct`: Minimum deviation from $1.00 (default: 2.0%)
- `max_position_usd`: Maximum position size (default: $300)
- `min_outcomes`: Minimum number of outcomes required (default: 3)

---

## Strategy 3: Cross-Platform Arbitrage

**Type:** Directional trade  
**Priority:** Medium  
**Timeframe:** All

### How It Works

Compares Polymarket prediction prices with actual spot prices from exchanges (Binance/CoinGecko). When there's a significant discrepancy, trade on the assumption that Polymarket will converge to the fair price.

### Example

Market: "Will BTC be above $100k by Friday?"
- Polymarket YES price: $0.45
- Current BTC spot: $99,800 (rising)
- Fair probability estimate: ~70%
- Mispricing: $0.25 (YES is underpriced)
- Action: Buy YES, expect price to rise toward $0.70

### Fair Probability Calculation

The strategy estimates fair probability based on:
- Distance of spot price from strike price
- Price momentum (recent price changes)
- Time to market expiry

### Execution

Buy the underpriced outcome (YES or NO) and wait for market correction.

### Configuration

- `min_price_diff_pct`: Minimum price difference to trigger (default: 3.0%)
- `max_position_usd`: Maximum position size (default: $1000)
- `stale_threshold_sec`: Maximum age of price data (default: 30s)

---

## Strategy 4: Endgame Arbitrage

**Type:** High-probability directional  
**Priority:** High  
**Timeframe:** Near resolution

### How It Works

When a market is close to resolution and one outcome has very high probability (>93%), buy that outcome for a small but near-certain profit.

### Example

Market: "BTC above $50k by end of day?" (resolves in 2 hours)
- BTC currently at $67,000
- YES price: $0.96
- Buy at $0.96, collect $1.00 on resolution
- Profit: $0.04 per share (4.2% in 2 hours)
- Annualized: ~18,000% APR

### Execution

Buy the high-probability outcome and hold until resolution.

### Configuration

- `min_probability`: Minimum outcome probability (default: 0.93)
- `max_time_to_resolution_hrs`: Maximum hours until resolution (default: 48)
- `min_annualized_return_pct`: Minimum annualized return (default: 100%)
- `max_position_usd`: Maximum position size (default: $2000)

---

## Strategy 5: Momentum/Mean-Reversion

**Type:** Technical analysis  
**Priority:** Medium  
**Timeframe:** 5m, 15m, 1h

### How It Works

Tracks Polymarket YES/NO prices as a price series and applies technical indicators:
- Z-score: Measures how far current price is from mean
- RSI: Relative Strength Index (oversold/overbought)
- Rate of Change: Price momentum
- VWAP divergence: Price vs volume-weighted average

### Entry Conditions

**Mean Reversion Buy (Oversold):**
- Z-score < -threshold
- RSI < 35
- Rate of change > -0.5%
- Action: Buy YES (expect price to revert up)

**Mean Reversion Sell (Overbought):**
- Z-score > +threshold
- RSI > 65
- Rate of change < 0.5%
- Action: Buy NO (expect YES price to revert down)

### Timeframe Parameters

**5-minute (Scalping):**
- Lookback: 12 candles (1 hour)
- Entry Z-score: ±2.0σ
- Take profit: 1.5%
- Stop loss: 1.0%

**15-minute (Swing):**
- Lookback: 16 candles (4 hours)
- Entry Z-score: ±1.8σ
- Take profit: 3.0%
- Stop loss: 2.0%

**1-hour (Position):**
- Lookback: 24 candles (1 day)
- Entry Z-score: ±1.5σ
- Take profit: 5.0%
- Stop loss: 3.0%

### Configuration

Each timeframe has its own parameters in `MomentumConfig`:
- `tf_5m_lookback`, `tf_5m_entry_zscore`, `tf_5m_take_profit_pct`, `tf_5m_stop_loss_pct`
- `tf_15m_*` (same pattern)
- `tf_1h_*` (same pattern)

---

## Signal Ranking

The `StrategyAggregator` ranks signals using a composite score based on:

- **Expected profit %** (30% weight)
- **Confidence** (25% weight)
- **Strategy priority** (20% weight)
- **Urgency** (15% weight)
- **Risk/reward ratio** (10% weight)

Strategy priorities:
1. Intra-market: 1.0 (highest)
2. Combinatorial: 0.95
3. Endgame: 0.90
4. Cross-platform: 0.80
5. Momentum/Mean-reversion: 0.70

## Risk Management

All strategies are subject to risk management controls:

- Maximum position size per trade
- Maximum portfolio exposure
- Daily loss limits
- Stop-loss and take-profit levels
- Consecutive loss protection

See `risk_manager.py` for details.

## How Strategies Are Combined

The bot runs all applicable strategies on each market simultaneously, then combines and ranks the results.

### Scanning Process

1. **Market Discovery**: Bot discovers all crypto markets for BTC, ETH, XRP, SOL
2. **Parallel Scanning**: For each market, all applicable strategies run in parallel:
   - Intra-Market (if binary market)
   - Combinatorial (if 3+ outcomes)
   - Cross-Platform (if strike price can be extracted)
   - Endgame (all markets)
   - Momentum/Mean-Reversion (all binary markets, per timeframe)

3. **Signal Collection**: All signals from all strategies are collected into a single list

4. **Filtering**: Low-quality signals are filtered out:
   - Minimum confidence: 0.35 (35%)
   - Minimum profit: 0.5%

5. **Ranking**: Remaining signals are ranked by composite score

6. **Execution**: Top-ranked signals are executed (max 3 per scan cycle)

### Composite Scoring Formula

Each signal gets a composite score (0.0 to 1.0) calculated as:

```
score = (profit_score × 0.30) +
        (confidence_score × 0.25) +
        (strategy_priority × 0.20) +
        (urgency_score × 0.15) +
        (risk_reward_score × 0.10)
```

Where:
- **profit_score**: Expected profit % / 10 (capped at 1.0)
- **confidence_score**: Signal confidence (0.0 to 1.0)
- **strategy_priority**: Strategy type priority (see below)
- **urgency_score**: HIGH=1.0, MEDIUM=0.67, LOW=0.33
- **risk_reward_score**: Risk/reward ratio × 5 (capped at 1.0)

### Strategy Priority Weights

Strategies are weighted by their inherent reliability:

1. **Intra-market**: 1.0 (risk-free arbitrage)
2. **Combinatorial**: 0.95 (risk-free arbitrage)
3. **Endgame**: 0.90 (high probability)
4. **Cross-platform**: 0.80 (directional, requires convergence)
5. **Momentum/Mean-reversion**: 0.70 (technical analysis, less certain)

### Example: Multiple Signals on One Market

A single market might generate multiple signals:

**Market**: "Will BTC be above $100k by Friday?"

1. **Intra-Market Signal**: YES=$0.45, NO=$0.50 → Combined=$0.95 → Arbitrage opportunity
   - Score: 0.85 (high priority, good profit)

2. **Cross-Platform Signal**: Spot price suggests 70% probability, but YES is $0.45
   - Score: 0.72 (medium priority, good mispricing)

3. **Momentum Signal (5m)**: Oversold condition detected
   - Score: 0.58 (lower priority, technical signal)

4. **Endgame Signal**: Market resolves in 2 hours, YES at $0.96
   - Score: 0.91 (high priority, near-certain profit)

**Result**: Signals ranked: Endgame (0.91) → Intra-Market (0.85) → Cross-Platform (0.72) → Momentum (0.58)

The bot would execute the Endgame signal first, then Intra-Market if capital allows.

### Signal Deduplication

The bot can generate multiple signals for the same market from different strategies. The risk manager prevents duplicate positions in the same market, so only the highest-ranked signal will execute.

### Timeframe Handling

Some strategies (Cross-Platform, Momentum) run across multiple timeframes (5m, 15m, 1h). Each timeframe generates a separate signal, allowing the bot to:
- Capture short-term scalping opportunities (5m)
- Identify swing trades (15m)
- Take position trades (1h)

All timeframe signals are ranked together, so a high-confidence 5m signal might rank higher than a lower-confidence 1h signal.

## Configuration

Strategy parameters can be adjusted in `config.py`. Each strategy has its own configuration class:

- `IntraMarketConfig`
- `CombinatorialConfig`
- `CrossPlatformConfig`
- `EndgameConfig`
- `MomentumConfig`

Strategies can be enabled/disabled individually via the `enabled` flag in each config.

### Adjusting Strategy Weights

To change how strategies are prioritized, modify `STRATEGY_PRIORITY` in `StrategyAggregator` class:

```python
STRATEGY_PRIORITY = {
    "intra_market": 1.0,        # Increase for more risk-free arb focus
    "combinatorial": 0.95,
    "endgame": 0.90,
    "cross_platform": 0.80,     # Increase for more directional trades
    "momentum_mean_reversion": 0.70,  # Increase for more technical trades
}
```

### Adjusting Composite Score Weights

To change what factors matter most, modify the weights in `_composite_score()`:

```python
composite = (
    profit_score * 0.30 +      # Increase for profit-focused
    confidence_score * 0.25 +   # Increase for confidence-focused
    strategy_score * 0.20 +    # Increase for strategy-type preference
    urgency_score * 0.15 +     # Increase for time-sensitive trades
    rr_score * 0.10            # Increase for risk/reward focus
)
```
