import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
import warnings
warnings.filterwarnings("ignore")

from config import (
    UNIVERSE,
    TRAIN_START, TRAIN_END,
    COINT_PVALUE_THRESHOLD,
    MIN_HALF_LIFE_DAYS,
    MAX_HALF_LIFE_DAYS,
)

# ── Load price s──────────────────────
# Trains based on data generated from file 1, limited by test interval
prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
prices = prices[TRAIN_START:TRAIN_END]
print(f"Loaded {prices.shape[0]} days of training data\n")


# ── Estimate half-life of mean reversion ─────────────────────
# Performs regression on pair daily change v. previous day's spread
# If slope is positive, spread is increasing & pair is not mean-reverting --> return inf)
def half_life(spread):
    spread = pd.Series(spread)
    lag    = spread.shift(1).dropna()
    delta  = spread.diff().dropna()
    beta   = np.polyfit(lag, delta, 1)[0]
    if beta >= 0:
        return np.inf   
    return -np.log(2) / beta


# ── Compute hedge ratio (simple OLS) ─────────────────────────
# Note that the hedge ratio here is static. Dynamic is implemented via Kalman filter
def hedge_ratio(y, x):
    return np.polyfit(x, y, 1)[0]


# ── Test every within-sector pair ────────────────────────────
results = []

for sector, tickers in UNIVERSE.items():
    tickers = [t for t in tickers if t in prices.columns]
    pairs   = list(combinations(tickers, 2))
    # Change to permutation due to Engle Granger 

    print(f"Sector: {sector} — testing {len(pairs)} pairs")

    for t1, t2 in pairs:
        s1 = prices[t1]
        s2 = prices[t2]

        # ADF test
        adf1 = adfuller(s1)[1]
        adf2 = adfuller(s2)[1]
        # Finding individual tickers that are non-stationary
        if adf1 < 0.05 or adf2 < 0.05:
            continue

        # Engle-Granger cointegration test
        score, pvalue, _ = coint(s1, s2)
        if pvalue > COINT_PVALUE_THRESHOLD:
            continue    # no significant cointegration

        beta   = hedge_ratio(s1.values, s2.values)
        spread = s1 - beta * s2
        hl     = half_life(spread)

        if hl < MIN_HALF_LIFE_DAYS or hl > MAX_HALF_LIFE_DAYS:
            continue   

        results.append({
            "sector":      sector,
            "ticker_1":    t1,
            "ticker_2":    t2,
            "pvalue":      round(pvalue, 4),
            "hedge_ratio": round(beta, 4),
            "half_life":   round(hl, 1),
            "spread_mean": round(spread.mean(), 4),
            "spread_std":  round(spread.std(), 4),
        })

# ── Rank and display results ──────────────────────────────────
if not results:
    print("\nNo cointegrated pairs found. Try relaxing thresholds in config.py")
else:
    # Stores results in table, listing by descending p-value
    df = pd.DataFrame(results).sort_values("pvalue")
    print(f"\nFound {len(df)} valid pairs:\n")
    print(df.to_string(index=False))

    df.to_csv("data/pairs.csv", index=False)
    print("\nSaved to data/pairs.csv")