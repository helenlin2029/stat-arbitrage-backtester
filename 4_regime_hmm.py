import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from config import N_REGIMES, HMM_LOOKBACK, TRAIN_START, TRAIN_END, TEST_END


# ── Load data ─────────────────────────────────────────────────
spreads = pd.read_csv("data/spreads.csv", index_col=0, parse_dates=True)
pairs   = pd.read_csv("data/pairs.csv")

print(f"Fitting HMM for {len(pairs)} pairs...\n")


# ── Feature matrix for the HMM ────────────────────────────────
def build_features(spread, window=20):
    s = pd.Series(spread)

    roll_std  = s.rolling(window).std()
    roll_mean = s.rolling(window).mean()
    z_score   = (s - roll_mean) / (roll_std + 1e-8)
    vol_of_vol = roll_std.rolling(window).std()

    features = pd.DataFrame({
        "spread_vol":  roll_std,
        "z_score":     z_score,
        "vol_of_vol":  vol_of_vol,
    }).dropna()

    return features


# ── Identify which HMM state is "stable" ──────────────────────
def identify_stable_state(model, scaler, feature_cols):
    low_vol  = np.array([[0.5, 0.0, 0.1]])   
    high_vol = np.array([[3.0, 2.5, 1.5]])   

    low_vol_scaled  = scaler.transform(low_vol)
    high_vol_scaled = scaler.transform(high_vol)

    low_scores  = model.score(low_vol_scaled)
    high_scores = model.score(high_vol_scaled)

    state_means = model.means_[:, 0]   
    stable_state = int(np.argmin(state_means))
    return stable_state


# ── Fit HMM for each pair ─────────────────────────────────────
all_p_stable = {}

for _, row in pairs.iterrows():
    t1    = row["ticker_1"]
    t2    = row["ticker_2"]
    label = f"{t1}_{t2}"

    spread   = spreads[label].dropna()
    features = build_features(spread)

    train_features = features[TRAIN_START:TRAIN_END]
    all_features   = features[TRAIN_START:TEST_END]

    if len(train_features) < HMM_LOOKBACK:
        print(f"  {label}: not enough data to fit HMM, skipping")
        continue

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_all   = scaler.transform(all_features)

    model = GaussianHMM(
        n_components=N_REGIMES,
        covariance_type="full",
        n_iter=100,
        random_state=42,
    )
    model.fit(X_train)

    state_probs = model.predict_proba(X_all)

    stable_state = identify_stable_state(model, scaler, train_features.columns)
    p_stable     = state_probs[:, stable_state]

    all_p_stable[label] = pd.Series(p_stable, index=all_features.index)

    avg_stable = p_stable.mean()
    print(f"  {label}: avg P(stable) = {avg_stable:.2f}  |  stable state = {stable_state}")


# ── Save results ──────────────────────────────────────────────
p_stable_df = pd.DataFrame(all_p_stable)
p_stable_df.to_csv("data/p_stable.csv")

print(f"\nSaved regime probabilities to data/p_stable.csv")
print(f"Shape: {p_stable_df.shape}")
print(f"\nSample (first 5 rows):\n{p_stable_df.head().round(3)}")