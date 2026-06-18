import os
import yfinance as yf
from config import UNIVERSE, TRAIN_START, TEST_END

# ── Flatten universe dict into a single list of tickers ─────
all_tickers = [ticker for sector in UNIVERSE.values() for ticker in sector]

print(f"Downloading data for {len(all_tickers)} tickers: {all_tickers}")
print(f"Date range: {TRAIN_START} → {TEST_END}\n")

# ── Download from yfinance ───────────────────────────────────
raw = yf.download(
    tickers=all_tickers,
    start=TRAIN_START,
    end=TEST_END,
    auto_adjust=True,   # adjusts for splits and dividends automatically
    progress=True,
)

prices = raw["Close"]

# ── Clean data ───────────────────────────────────────────
# Filters out tickers that are missing more than 5% of data (eg not yet established or unlisted)
missing_pct = prices.isnull().mean()
bad_tickers = missing_pct[missing_pct > 0.05].index.tolist()
if bad_tickers:
    print(f"Dropping tickers with >5% missing data: {bad_tickers}")
    prices = prices.drop(columns=bad_tickers)

# Closes gaps in data due to international discrepancies (holidays)
prices = prices.ffill()

# Drops early rows (yeas) where there exists a ticker with an empty value 
prices = prices.dropna()

print(f"\nFinal dataset: {prices.shape[0]} trading days × {prices.shape[1]} tickers")
print(f"From {prices.index[0].date()} to {prices.index[-1].date()}")

# ── Save to CSV ──────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
prices.to_csv("data/prices.csv")
print("\nSaved to data/prices.csv")

# ── Quick check ───────────────────────────────────────
print("\nFirst 5 rows:")
print(prices.head())

print("\nBasic stats:")
print(prices.describe().round(2))