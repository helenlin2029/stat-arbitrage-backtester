import pandas as pd
import numpy as np
from config import KALMAN_DELTA, KALMAN_VE, TRAIN_START, TEST_END

# ── Load data ────────────────────────────────────────────────
prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
pairs  = pd.read_csv("data/pairs.csv")

print(f"Computing Kalman spreads for {len(pairs)} pairs...\n")


# ── Kalman filter implementation ─────────────────────────────
def kalman_hedge_ratio(y, x):
    """
    y: price series of stock 1 (the one we're 'buying')
    x: price series of stock 2 (the one we're hedging with)

    Returns:
        beta_series : hedge ratio at each timestep
        spread      : y - beta * x at each timestep
        std_series  : rolling std of the spread (used for z-score later)
    """
    n = len(y)

    beta     = np.zeros(n)         
    P        = np.zeros(n)      
    beta[0]  = y[0] / x[0]
    P[0]     = 1.0

    # Beta process noise
    # Beta observation noise
    Vw = KALMAN_DELTA / (1 - KALMAN_DELTA)
    Ve = KALMAN_VE

    # -- Filter loop --
    for t in range(1, n):
        beta_pred = beta[t-1]
        P_pred    = P[t-1] + Vw

        x_t  = x[t]
        y_hat = beta_pred * x_t          
        err  = y[t] - y_hat              

        K        = P_pred * x_t / (x_t**2 * P_pred + Ve)
        beta[t]  = beta_pred + K * err   
        P[t]     = (1 - K * x_t) * P_pred  

    # -- Compute spread --
    spread = y - beta * x

    return pd.Series(beta, index=y.index), pd.Series(spread, index=y.index)


# ── Process every pair and save results ──────────────────────
all_spreads    = {}
all_betas      = {}

for _, row in pairs.iterrows():
    t1     = row["ticker_1"]
    t2     = row["ticker_2"]
    label  = f"{t1}_{t2}"

    y = prices[t1]
    x = prices[t2]

    beta_series, spread = kalman_hedge_ratio(y.values, x.values)
    beta_series.index   = y.index
    spread.index        = y.index

    all_betas[label]   = beta_series
    all_spreads[label] = spread

    print(f"  {label}: hedge ratio today = {beta_series.iloc[-1]:.4f}")

# ── Save to CSV ──────────────────────────────────────────────
spreads_df = pd.DataFrame(all_spreads)
betas_df   = pd.DataFrame(all_betas)

spreads_df.to_csv("data/spreads.csv")
betas_df.to_csv("data/betas.csv")

print(f"\nSaved spreads to data/spreads.csv")
print(f"Saved hedge ratios to data/betas.csv")
print(f"\nSpread shape: {spreads_df.shape}")
print(f"\nFirst few rows of spreads:\n{spreads_df.head().round(4)}")