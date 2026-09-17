# Crypto Micro-Cap Rotation Alpha Model (CMRAM)

## Research & Backtesting Technical Specification

**Version:** V0.1  
**Status:** Research Phase  
**Objective:** Determine whether a quantitative crypto rotation model can identify statistically significant early-stage expansion opportunities.

---

# 1. Executive Summary

## Purpose

This project investigates whether cryptocurrency micro-cap assets can be systematically identified before major expansion phases by combining:

- market structure analysis,
- liquidity assessment,
- volume acceleration,
- narrative growth,
- and saturation detection.

The objective is not to predict every explosive move.

The objective is to determine whether a measurable statistical edge exists.

The system should answer:

> "When a cryptocurrency asset shows early expansion characteristics while remaining relatively unsaturated, does it outperform comparable assets over future periods?"

---

# 2. Research Philosophy

The strategy is built around the idea that crypto micro-cap movements often follow a lifecycle:

Accumulation
↓
Early Attention
↓
Volume Expansion
↓
Narrative Growth
↓
Price Expansion
↓
Crowded Trade
↓
Distribution
↓
Reversal

The model attempts to identify the transition between:

Early Attention
↓
Volume Expansion
↓
Price Expansion

before:

Crowded Trade

---

# 3. Important Research Constraints

The AI agent must follow these rules:

## Rule 1

Do not assume the strategy works.

The purpose of research is validation.

---

## Rule 2

Do not add complexity before proving the core hypothesis.

Do not introduce:

- machine learning,
- alternative indicators,
- additional datasets,
- predictive algorithms,

until baseline testing is completed.

---

## Rule 3

Every variable must prove measurable value.

If removing a variable produces identical performance, remove it.

---

## Rule 4

Avoid overfitting.

A model that performs perfectly on historical data but fails on unseen data is considered unsuccessful.

---

# 4. Core Hypothesis

## Primary Hypothesis

Assets with:

- increasing market participation,
- accelerating attention,
- improving liquidity,
- and low narrative saturation,

have a higher probability of producing positive future returns.

---

# 5. System Overview

The system consists of:

Data Collection
↓
Feature Engineering
↓
Signal Calculation
↓
Ranking System
↓
Backtesting Engine
↓
Performance Evaluation

---

# 6. Market Universe Definition

The initial research universe should focus on micro-cap cryptocurrencies.

The universe definition must be documented.

Example:

Minimum Market Cap:
$5M

Maximum Market Cap:
$250M

Minimum Trading History:
90 days

Minimum Daily Volume:
Defined through liquidity testing

These values are research parameters, not fixed assumptions.

The AI agent should test multiple ranges.

---

# 7. Required Data Sources

## 7.1 Market Data

Required:

- Timestamp
- Token ID
- Price
- Market capitalization
- Trading volume
- High
- Low
- Open
- Close
- Trading pairs
- Exchange availability

Frequency:

Preferred:

- Daily data initially
- Hourly data later

---

# 7.2 Liquidity Data

Purpose:

Prevent false signals caused by impossible execution.

Required:

- Daily volume
- Bid/ask spread
- Order book depth
- Liquidity pool depth
- Slippage estimation

Potential liquidity metrics:

## Amihud Illiquidity Ratio

Formula:

ILLIQ =
Average(
Absolute Return / Dollar Volume
)

Purpose:

Estimate price impact caused by trading.

---

# 7.3 Social Narrative Data

Purpose:

Measure attention development.

Required:

## X/Twitter

Metrics:

- Mentions
- Unique accounts
- Engagement
- Likes
- Retweets
- Replies
- Influencer mentions

## Reddit

Metrics:

- Posts
- Comments
- Community growth
- Engagement

## Telegram/Discord

Metrics:

- Member growth
- Message frequency
- Active users

## Search Data

Metrics:

- Google Trends
- Search growth

---

# 8. Database Structure

Example:

## Asset Table

asset_id
symbol
name
market_cap
category
listing_date

---

## Market Snapshot Table

timestamp
asset_id
price
volume
market_cap
high
low
close

---

## Social Snapshot Table

timestamp
asset_id
mentions
unique_users
engagement
sentiment
community_growth

---

## Signal Table

timestamp
asset_id
MREI
NSI
Rotation_Gap
signal_status

---

# 9. Feature Engineering

Raw data must be converted into measurable factors.

The model consists of:

1. Expansion Pressure
2. Saturation Pressure

---

# 10. Micro-Cap Rotation Expansion Index (MREI)

Purpose:

Measure whether conditions indicate early-stage expansion.

Formula:

MREI =
Compression Score
+
Volume Resurrection Score
+
Narrative Acceleration Score
+
Price Confirmation Score

All components normalized:

0-100.

---

# 10.1 Compression Score

Purpose:

Identify assets that have not already completed their move.

Possible inputs:

- Distance from recent high
- Volatility contraction
- Trading range compression
- Selling pressure reduction

Desired condition:

Stable base formation
Reduced downside pressure
Potential for renewed demand

---

# 10.2 Volume Resurrection Score

Purpose:

Detect returning capital.

Metrics:

## Relative Volume

RVOL =
Current Volume /
Average Historical Volume

---

## Volume Velocity

Velocity =
(Current Volume - Previous Volume)
/ Previous Volume

---

## Volume Acceleration

Acceleration =
Current Velocity - Previous Velocity

Positive acceleration indicates increasing participation.

---

# 10.3 Narrative Acceleration Score

Purpose:

Measure attention growth.

The model should not simply measure:

"Most talked about token"

It should measure:

"Fastest increasing attention."

Metrics:

- Mention growth
- Unique participant growth
- Engagement growth
- Narrative velocity
- Narrative acceleration

---

Formula:

Narrative Acceleration =
Change in Attention Velocity

---

# 10.4 Price Confirmation Score

Purpose:

Ensure attention and volume are translating into market action.

Possible metrics:

- Higher lows
- Short-term trend recovery
- Momentum improvement
- Relative strength
- Breakout confirmation

---

# 11. Narrative Saturation Index (NSI)

Purpose:

Measure whether the opportunity is already crowded.

Formula:

NSI =
Social Saturation
+
Price Extension
+
Volume Exhaustion
+
Momentum Deceleration

---

# 11.1 Social Saturation

Measures:

- Extremely high mention volume
- Market-wide awareness
- Community crowding

---

# 11.2 Price Extension

Measures:

- Recent percentage gain
- Distance from moving averages
- Overextended conditions

---

# 11.3 Volume Exhaustion

Measures:

- Volume climax
- Declining volume acceleration

---

# 11.4 Momentum Deceleration

Measures:

Whether:

Growth rate is slowing

Example:

Day 1:
100 mentions

Day 2:
300 mentions

Day 3:
600 mentions

Day 4:
650 mentions

Attention is still high but acceleration is weakening.

---

# 12. Rotation Gap

Primary signal.

Formula:

Rotation Gap = MREI - NSI

Interpretation:

High positive value:

Expansion potential
Crowding risk

Low value:

Trade already mature

---

# 13. Signal Generation

Initial research signal:

Liquidity Requirement:
PASS

AND

MREI above tested threshold

AND

Rotation Gap above tested threshold

The AI agent must test multiple thresholds.

Example:

Gap > 20

Gap > 30

Gap > 40

Gap > 50

---

# 14. Backtesting Framework

## Objective

Determine whether historical signals produced superior future returns.

---

# 14.1 Signal Recording

Every signal must record:

Date
Asset
Entry Price
Market Cap
Volume
MREI
NSI
Rotation Gap

---

# 14.2 Forward Testing

Measure returns after:

1 Day
3 Days
7 Days
14 Days
21 Days
30 Days

---

# 15. Performance Metrics

Required:

## Return Metrics

- Average return
- Median return
- Win rate
- Best trade
- Worst trade

---

## Risk Metrics

- Maximum drawdown
- Maximum adverse excursion
- Volatility
- Loss distribution

---

## Trading Metrics

- Signal frequency
- Average holding period
- Profit factor
- Risk/reward ratio

---

# 16. Maximum Favorable Excursion

Measure:

"What was the maximum gain available after entry?"

Example:

Entry:  $1
Maximum:  $3
MFE:  200%

---

# 17. Maximum Adverse Excursion

Measure:

"How far did price move against the position?"

Example:

Entry:  $1
Lowest:  $0.75
MAE:  -25%

---

# 18. Benchmark Testing

Compare against:

## Random Selection

Random micro-cap portfolio.

---

## Simple Momentum

Assets with highest recent returns.

---

## Volume Breakout

Assets with largest volume increases.

---

## Market Benchmark

General crypto market performance.

---

# 19. Feature Contribution Testing

Test progressively.

## Model A

Price only.

---

## Model B

Price + Volume.

---

## Model C

Price + Volume + Social.

---

## Model D

Full Rotation Gap.

---

Determine whether each additional component improves:

- returns,
- consistency,
- risk-adjusted performance.

---

# 20. Future Enhancements

Only after validation:

Possible additions:

- Exchange listings
- On-chain wallet activity
- Smart money tracking
- Token unlock events
- Partnerships
- Developer activity

Each addition must be tested independently.

---

# 21. Expected AI Agent Outputs

The AI agent should produce:

## Dataset Report

Including:

- Sources
- Coverage
- Missing data
- Limitations

---

## Backtest Report

Including:

- Performance
- Statistical significance
- Weaknesses

---

## Feature Analysis

Including:

- Which variables contributed
- Which variables failed

---

## Final Recommendation

Not:

"Strategy works"

or:

"Strategy fails"

but:

A research conclusion explaining:

- evidence,
- confidence level,
- limitations,
- next steps.

---

# Final Research Objective

The project succeeds only if it can demonstrate:

High Rotation Gap
Liquidity Control
Historical Outperformance
Acceptable Risk Profile

without relying on hindsight or overfitting.
