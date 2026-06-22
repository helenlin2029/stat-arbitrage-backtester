import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from config import TRAIN_START, TRAIN_END, TEST_START, TEST_END, ROLLING_WINDOW

# ── Load data ─────────────────────────────────────────────────
portfolio_pnl = pd.read_csv("data/portfolio_pnl.csv", index_col=0, parse_dates=True)["portfolio_pnl"]
pnl_per_pair  = pd.read_csv("data/pnl_per_pair.csv",  index_col=0, parse_dates=True)
p_stable      = pd.read_csv("data/p_stable.csv",       index_col=0, parse_dates=True)
signals       = pd.read_csv("data/signals.csv",         index_col=0, parse_dates=True)
sized_signals = pd.read_csv("data/sized_signals.csv",   index_col=0, parse_dates=True)
zscores       = pd.read_csv("data/zscores.csv",         index_col=0, parse_dates=True)
pairs         = pd.read_csv("data/pairs.csv")

# Only looks as testing period
test_pnl     = portfolio_pnl[TEST_START:TEST_END]
test_pair_pnl = pnl_per_pair[TEST_START:TEST_END]


# ── Compute performance metrics ────────────────────────────
def metrics(pnl_series):
    pnl = pnl_series.dropna()
    if len(pnl) == 0 or pnl.std() == 0:
        return {}
    cum      = pnl.cumsum()
    roll_max = cum.cummax()
    drawdown = cum - roll_max

    return {
        "total_pnl":    round(pnl.sum(), 2),
        "sharpe":       round((pnl.mean() / pnl.std()) * np.sqrt(252), 3),
        "max_drawdown": round(drawdown.min(), 2),
        "win_rate":     round((pnl > 0).mean(), 3),
        "avg_daily":    round(pnl.mean(), 2),
        "vol_daily":    round(pnl.std(), 2),
        "n_days":       len(pnl),
    }


# ── OVERALL PERFORMANCE ───────────────────────────────────
print("=" * 55)
print("SECTION 1: OVERALL PERFORMANCE (TEST PERIOD)")
print("=" * 55)

m = metrics(test_pnl)
for k, v in m.items():
    print(f"  {k:<20} {v}")


# ── REGIME-CONDITIONAL PERFORMANCE ────────────────────────
print("\n" + "=" * 55)
print("SECTION 2: PERFORMANCE BY REGIME (TEST PERIOD)")
print("=" * 55)

avg_p_stable = p_stable[TEST_START:TEST_END].mean(axis=1)
avg_p_stable = avg_p_stable.reindex(test_pnl.index)

# Allows the differentiation of P&L series
stable_days = avg_p_stable > 0.6
crisis_days = avg_p_stable <= 0.6

stable_pnl = test_pnl[stable_days]
crisis_pnl = test_pnl[crisis_days]

print(f"\n  Stable regime days : {stable_days.sum()}")
print(f"  Crisis regime days : {crisis_days.sum()}")

print("\n  --- Stable regime ---")
for k, v in metrics(stable_pnl).items():
    print(f"    {k:<20} {v}")

print("\n  --- Crisis regime ---")
for k, v in metrics(crisis_pnl).items():
    print(f"    {k:<20} {v}")


# ── REGIME FILTER V ALUE-ADD ───────────────────────────────
print("\n" + "=" * 55)
print("SECTION 3: REGIME-AWARE vs NAIVE STRATEGY")
print("=" * 55)

prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
from config import CAPITAL_PER_PAIR, TRANSACTION_COST_BPS
COST = TRANSACTION_COST_BPS / 10_000

# Simulates what ignoring HMM and trading every signal at full size would've looked like
naive_pnl_all = {}
for _, row in pairs.iterrows():
    t1, t2        = row["ticker_1"], row["ticker_2"]
    hedge_ratio   = row["hedge_ratio"]
    label         = f"{t1}_{t2}"
    if label not in signals.columns:
        continue
    sig = signals[label].dropna()
    p1  = prices[t1].reindex(sig.index)
    p2  = prices[t2].reindex(sig.index)
    r1  = p1.pct_change()
    r2  = p2.pct_change()
    spread_ret    = r1 - hedge_ratio * r2
    pnl_raw       = sig.shift(1) * spread_ret * CAPITAL_PER_PAIR
    costs         = sig.diff().abs() * COST * CAPITAL_PER_PAIR
    naive_pnl_all[label] = pnl_raw - costs

naive_portfolio = pd.DataFrame(naive_pnl_all).sum(axis=1)
naive_test      = naive_portfolio[TEST_START:TEST_END]

print("\n  --- Naive (no regime filter) ---")
for k, v in metrics(naive_test).items():
    print(f"    {k:<20} {v}")

print("\n  --- Regime-aware ---")
for k, v in metrics(test_pnl).items():
    print(f"    {k:<20} {v}")

# Positive Sharpe = regime-dependent prediction was better
sharpe_improvement = metrics(test_pnl)["sharpe"] - metrics(naive_test)["sharpe"]
print(f"\n  Sharpe improvement from regime filter: {sharpe_improvement:+.3f}")


# ── PER-PAIR CONTRIBUTION ──────────────────────────────────
print("\n" + "=" * 55)
print("SECTION 4: PER-PAIR CONTRIBUTION (TEST PERIOD)")
print("=" * 55)

pair_summary = []
for col in test_pair_pnl.columns:
    m = metrics(test_pair_pnl[col])
    if m:
        m["pair"] = col
        pair_summary.append(m)

# Identifies which pairs generated returns v. losses; can use to examine method fallabilities
pair_df = pd.DataFrame(pair_summary).set_index("pair")
pair_df = pair_df.sort_values("sharpe", ascending=False)
print(f"\n{pair_df[['total_pnl','sharpe','max_drawdown','win_rate']].to_string()}")


# ── PLOTS ──────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Statistical Arbitrage Backtester — Analysis", fontsize=14, fontweight="bold")
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.8, wspace=0.5)

# Plot 1: Cumulative P&L — regime-aware vs naive
ax1 = fig.add_subplot(gs[0, :])
test_pnl.cumsum().plot(ax=ax1, label="Regime-aware", color="#3B8BD4", linewidth=1.5)
naive_test.cumsum().plot(ax=ax1, label="Naive", color="#E24B4A", linewidth=1.5, linestyle="--")
ax1.axhline(0, color="gray", linewidth=0.5)
ax1.set_title("Cumulative P&L: regime-aware vs naive (test period)")
ax1.set_ylabel("Cumulative P&L ($)")
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Daily P&L distribution
ax2 = fig.add_subplot(gs[1, 0])
test_pnl.hist(bins=50, ax=ax2, color="#3B8BD4", alpha=0.7, edgecolor="none")
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_title("Daily P&L distribution (test)")
ax2.set_xlabel("Daily P&L ($)")
ax2.set_ylabel("Frequency")

# Plot 3: Average P(stable) over test period
ax3 = fig.add_subplot(gs[1, 1])
avg_p_stable.plot(ax=ax3, color="#639922", linewidth=1)
ax3.axhline(0.6, color="#E24B4A", linewidth=0.8, linestyle="--", label="Threshold (0.6)")
ax3.set_title("Market regime: avg P(stable)")
ax3.set_ylabel("P(stable)")
ax3.set_ylim(0, 1)
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Rolling Sharpe ratio (63-day)
ax4 = fig.add_subplot(gs[2, 0])
rolling_sharpe = (
    test_pnl.rolling(63).mean() / test_pnl.rolling(63).std()
) * np.sqrt(252)
rolling_sharpe.plot(ax=ax4, color="#7F77DD", linewidth=1.2)
ax4.axhline(0, color="gray", linewidth=0.5)
ax4.axhline(1, color="#639922", linewidth=0.8, linestyle="--", label="Sharpe = 1")
ax4.set_title("Rolling 63-day Sharpe ratio")
ax4.set_ylabel("Sharpe")
ax4.legend()
ax4.grid(alpha=0.3)

# Plot 5: Per-pair total P&L bar chart
ax5 = fig.add_subplot(gs[2, 1])
pair_totals = pair_df["total_pnl"].sort_values()
colors = ["#E24B4A" if v < 0 else "#3B8BD4" for v in pair_totals]
pair_totals.plot(kind="barh", ax=ax5, color=colors)
ax5.axvline(0, color="black", linewidth=0.8)
ax5.set_title("Total P&L by pair (test)")
ax5.set_xlabel("Total P&L ($)")
ax5.tick_params(axis="y", labelsize=8)

plt.savefig("data/analysis.png", dpi=150, bbox_inches="tight")
print("\nSaved chart to data/analysis.png")
plt.show()