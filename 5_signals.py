import pandas as pd
import numpy as np
from config import (
    ROLLING_WINDOW,
    ENTRY_ZSCORE,
    EXIT_ZSCORE,
    STOP_LOSS_ZSCORE,
    TRAIN_START, TEST_END,
)

# ── Load data ─────────────────────────────────────────────────
spreads  = pd.read_csv("data/spreads.csv",  index_col=0, parse_dates=True)
p_stable = pd.read_csv("data/p_stable.csv", index_col=0, parse_dates=True)
pairs    = pd.read_csv("data/pairs.csv")

print(f"Generating signals for {len(pairs)} pairs...\n")


# ── Compute rolling z-score ───────────────────────────────────
def rolling_zscore(spread, window):
    mean = spread.rolling(window).mean()
    std  = spread.rolling(window).std()
    return (spread - mean) / (std + 1e-8)


# ── Generate raw signals from z-score ─────────────────────────
def generate_signals(z):
    signals  = pd.Series(0, index=z.index, dtype=float)
    position = 0  

    for i in range(len(z)):
        zi = z.iloc[i]

        if np.isnan(zi):
            signals.iloc[i] = 0
            continue

        if abs(zi) > STOP_LOSS_ZSCORE:
            position = 0

        elif position == 0:
            if zi > ENTRY_ZSCORE:
                position = -1   
            elif zi < -ENTRY_ZSCORE:
                position = 1    

        elif position == 1 and zi > EXIT_ZSCORE:
            position = 0
        elif position == -1 and zi < EXIT_ZSCORE:
            position = 0

        signals.iloc[i] = position

    return signals


# ── Process every pair ────────────────────────────────────────
all_zscores  = {}
all_signals  = {}
all_sized    = {}   

for _, row in pairs.iterrows():
    t1    = row["ticker_1"]
    t2    = row["ticker_2"]
    label = f"{t1}_{t2}"

    if label not in spreads.columns:
        continue
    if label not in p_stable.columns:
        continue

    spread = spreads[label].dropna()
    regime = p_stable[label].reindex(spread.index).fillna(0.5)

    # Compute z-score
    z = rolling_zscore(spread, ROLLING_WINDOW)

    # Generate raw signals 
    raw_signal = generate_signals(z)

    sized_signal = raw_signal * regime

    all_zscores[label] = z
    all_signals[label] = raw_signal
    all_sized[label]   = sized_signal

    n_trades = (raw_signal.diff().abs() > 0).sum()
    avg_size = sized_signal.abs().mean()
    print(f"  {label}: {n_trades} signal changes | avg position size = {avg_size:.2f}")


# ── Save outputs ──────────────────────────────────────────────
pd.DataFrame(all_zscores).to_csv("data/zscores.csv")
pd.DataFrame(all_signals).to_csv("data/signals.csv")
pd.DataFrame(all_sized).to_csv("data/sized_signals.csv")

print(f"\nSaved to data/zscores.csv, data/signals.csv, data/sized_signals.csv")