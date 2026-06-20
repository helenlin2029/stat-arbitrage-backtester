import pandas as pd
import numpy as np
from config import (
    TRANSACTION_COST_BPS,
    CAPITAL_PER_PAIR,
    TRAIN_START, TRAIN_END,
    TEST_START, TEST_END,
)

# ── Load data ─────────────────────────────────────────────────
prices       = pd.read_csv("data/prices.csv",        index_col=0, parse_dates=True)
sized_signals= pd.read_csv("data/sized_signals.csv", index_col=0, parse_dates=True)
pairs        = pd.read_csv("data/pairs.csv")

print(f"Running backtest for {len(pairs)} pairs...\n")

# Converts basis points to decimal
COST = TRANSACTION_COST_BPS / 10_000   


# ── Backtest a single pair ────────────────────────────────────
def backtest_pair(t1, t2, hedge_ratio, signal):
    p1 = prices[t1].reindex(signal.index)
    p2 = prices[t2].reindex(signal.index)

    r1 = p1.pct_change()
    r2 = p2.pct_change()

    spread_return = r1 - hedge_ratio * r2

    # Computes the P&L in dollar amount
    pnl_raw = signal.shift(1) * spread_return * CAPITAL_PER_PAIR

    position_change = signal.diff().abs()
    costs = position_change * COST * CAPITAL_PER_PAIR

    # Computes net P&L by accounting for costs/basis points per leg
    pnl_net = pnl_raw - costs

    return pnl_net


# ── Run backtest for every pair ───────────────────────────────
all_pnl = {}

for _, row in pairs.iterrows():
    t1           = row["ticker_1"]
    t2           = row["ticker_2"]
    hedge_ratio  = row["hedge_ratio"]
    label        = f"{t1}_{t2}"

    if label not in sized_signals.columns:
        continue

    signal = sized_signals[label].dropna()
    pnl    = backtest_pair(t1, t2, hedge_ratio, signal)
    all_pnl[label] = pnl

    total    = pnl.sum()
    n_days   = pnl.notna().sum()
    print(f"  {label}: total P&L = ${total:,.0f} over {n_days} days")


# ── Aggregate into a portfolio ────────────────────────────────
pnl_df = pd.DataFrame(all_pnl).dropna(how="all")

# Finds cumulative P&L across all pairs per day
portfolio_pnl = pnl_df.sum(axis=1)

cumulative_pnl = portfolio_pnl.cumsum()

# Splits portfolio into training and testing; used to evaluate accuracy 
train_pnl = portfolio_pnl[TRAIN_START:TRAIN_END]
test_pnl  = portfolio_pnl[TEST_START:TEST_END]

# ── Save outputs ──────────────────────────────────────────────
pnl_df.to_csv("data/pnl_per_pair.csv")
portfolio_pnl.to_frame("portfolio_pnl").to_csv("data/portfolio_pnl.csv")

print(f"\nSaved P&L data to data/pnl_per_pair.csv and data/portfolio_pnl.csv")

# ── Print summary ─────────────────────────────────────────────
print("\n" + "="*50)
print("PORTFOLIO SUMMARY")
print("="*50)

for label, period_pnl in [("TRAIN (2010-2019)", train_pnl), ("TEST  (2020-2024)", test_pnl)]:
    total      = period_pnl.sum()
    daily_mean = period_pnl.mean()
    daily_std  = period_pnl.std()
    # Converts daily Sharpe to annual
    sharpe     = (daily_mean / daily_std) * np.sqrt(252) if daily_std > 0 else 0

    # Maximum drawdown
    cum       = period_pnl.cumsum()
    roll_max  = cum.cummax()
    drawdown  = cum - roll_max
    max_dd    = drawdown.min()

    print(f"\n{label}")
    print(f"  Total P&L:       ${total:,.0f}")
    print(f"  Sharpe ratio:    {sharpe:.2f}")
    print(f"  Max drawdown:    ${max_dd:,.0f}")
    print(f"  Win rate:        {(period_pnl > 0).mean():.1%}")
