"""Phase 6A: Regime-Split LightGBM Training.

Trains separate LambdaRank models for bull, bear, and sideway market regimes.
Each model learns regime-specific alpha patterns.

Key differences from Phase 4:
  - Data split by market regime (not just by date)
  - 3 separate models, one per regime
  - Continuous ensemble label (no discretization needed)
  - Single scaler fitted on all training data

Usage: python src/train_lgb_phase6.py
Output: ./model/regime/ (3 model files + scaler + feature_cols + regime_stats)
"""

import os
import random
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from config import CONFIG
from features_phase6 import engineer_phase6_features
from features_phase4 import discretize_labels_tail

warnings.filterwarnings("ignore")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_rank_ic(df: pd.DataFrame) -> float:
    """Compute daily Rank IC (Spearman correlation pred vs label)."""
    daily_ics = []
    for _, group in df.groupby("日期"):
        if len(group) < 10:
            continue
        ic, _ = spearmanr(group["pred"], group["label"])
        daily_ics.append(ic)
    return np.mean(daily_ics) if daily_ics else 0.0


# ======================================================================
# Training helper
# ======================================================================

def train_regime_model(
    df_regime: pd.DataFrame,
    feature_cols: list,
    regime_name: str,
    params: dict,
    early_stopping_rounds: int = 100,
    val_ratio: float = 0.2,
) -> tuple:
    """Train a single LightGBM Ranker on one regime's data.

    Splits regime data chronologically: first (1-val_ratio) for train,
    last val_ratio for validation.

    Args:
        df_regime: DataFrame filtered to one regime
        feature_cols: Feature column names
        regime_name: 'bull', 'bear', or 'sideway'
        params: LightGBM parameters
        early_stopping_rounds: Early stopping patience
        val_ratio: Fraction of dates to use for validation

    Returns:
        (model, val_eval_df, ndcg_score)
    """
    df_r = df_regime.sort_values(["日期", "股票代码"]).copy()

    # Time-based train/val split
    all_dates = sorted(df_r["日期"].unique())
    n_val_dates = max(1, int(len(all_dates) * val_ratio))
    train_dates = set(all_dates[:-n_val_dates])
    val_dates = set(all_dates[-n_val_dates:])

    train_df = df_r[df_r["日期"].isin(train_dates)].copy()
    val_df = df_r[df_r["日期"].isin(val_dates)].copy()

    n_train_days = len(train_dates)
    n_val_days = len(val_dates)

    print(f"\n  [{regime_name.upper()}] "
          f"Train: {len(train_df)} rows, {n_train_days} days  |  "
          f"Val: {len(val_df)} rows, {n_val_days} days")

    if n_train_days < 10 or n_val_days < 1:
        print(f"  [SKIP] Insufficient data for {regime_name} "
              f"(train_days={n_train_days}, val_days={n_val_days})")
        return None, None, None

    # Discretize labels (tail-amplified) — LambdaRank requires integer relevance
    train_df = discretize_labels_tail(train_df)
    val_df = discretize_labels_tail(val_df)

    # Prepare data
    train_groups = train_df.groupby("日期", sort=False).size().values
    val_groups = val_df.groupby("日期", sort=False).size().values

    X_train = train_df[feature_cols].fillna(0).values.astype(np.float32)
    y_train = train_df["relevance"].values.astype(np.int32)
    X_val = val_df[feature_cols].fillna(0).values.astype(np.float32)
    y_val = val_df["relevance"].values.astype(np.int32)

    train_ds = lgb.Dataset(X_train, label=y_train, group=train_groups)
    val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_ds)

    # Train
    model = lgb.train(
        params=params,
        train_set=train_ds,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(0),
        ],
    )

    # Evaluate on validation set
    val_preds = model.predict(X_val)
    val_eval = val_df.copy()
    val_eval["pred"] = val_preds
    # Restore continuous label for Top-5 scoring
    val_eval["label"] = val_df["label"].values

    ndcg = model.best_score.get("val", {}).get("ndcg@5", 0.0)
    rank_ic = compute_rank_ic(val_eval)

    # Top-5 score
    daily_scores = []
    for date, group in val_eval.groupby("日期"):
        if len(group) < 5:
            continue
        pred_top5 = group.nlargest(5, "pred")["label"].sum()
        true_top5 = group.nlargest(5, "label")["label"].sum()
        random5 = 5 * group["label"].mean()
        denom = true_top5 - random5
        score = (
            (pred_top5 - random5) / (denom + 1e-12)
            if abs(denom) > 1e-6
            else 0.0
        )
        daily_scores.append(score)

    mean_score = np.mean(daily_scores) if daily_scores else 0.0

    print(f"  [{regime_name.upper()}] "
          f"NDCG@5={ndcg:.4f}, Final Score={mean_score:.4f}, "
          f"Rank IC={rank_ic:.4f}, Best Iter={model.best_iteration}")

    return model, val_eval, ndcg


# ======================================================================
# Main
# ======================================================================

def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    # Output directory for regime models
    regime_dir = os.path.join(cfg["output_dir"], "regime")
    os.makedirs(regime_dir, exist_ok=True)

    # ---- 1. Load data ----
    print("Loading data...")
    data_path = os.path.join(cfg["data_path"], "train.csv")
    df = pd.read_csv(data_path, dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"])

    print(f"Raw data: {len(df)} rows, {df['股票代码'].nunique()} stocks")
    print(f"Date range: {df['日期'].min().date()} to {df['日期'].max().date()}")

    # Filter stocks with insufficient history
    stock_counts = df.groupby("股票代码").size()
    valid_stocks = stock_counts[stock_counts >= cfg["min_history_days"]].index
    df = df[df["股票代码"].isin(valid_stocks)]
    n_stocks = df["股票代码"].nunique()
    print(f"After min-history filter: {n_stocks} stocks")

    # ---- 2. Phase 6 feature engineering ----
    regime_cfg = cfg.get("phase6", {})
    vol_threshold = regime_cfg.get("vol_threshold", 0.02)

    df, feature_cols = engineer_phase6_features(df, vol_threshold=vol_threshold)
    df = df.dropna(subset=["label"])
    df[feature_cols] = df[feature_cols].fillna(0.0)

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples after label filter: {len(df)}")

    # ---- 3. Fit global scaler ----
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].fillna(0).values)

    # ---- 4. Train per-regime models ----
    params = cfg["lgb_params"].copy()

    # Use ALL data for training (no date-based WF split — regime IS the split)
    # Each regime model sees all dates belonging to that regime
    regimes = ["bull", "bear", "sideway"]

    print(f"\n{'=' * 50}")
    print(f"Training Regime-Split Models")
    print(f"{'=' * 50}")
    print(f"  Vol threshold: {vol_threshold}")
    print(f"  Regimes: {regimes}")

    models = {}
    regime_results = {}

    for regime in regimes:
        df_regime = df[df["regime"] == regime].copy()
        if len(df_regime) < 500:
            print(f"\n  [{regime.upper()}] Only {len(df_regime)} samples — skipping")
            continue

        # Scale features
        df_regime[feature_cols] = scaler.transform(
            df_regime[feature_cols].fillna(0).values
        )

        model, val_eval, ndcg = train_regime_model(
            df_regime,
            feature_cols,
            regime,
            params,
            early_stopping_rounds=cfg["early_stopping_rounds"],
        )

        if model is not None:
            models[regime] = model
            regime_results[regime] = {
                "ndcg": ndcg,
                "n_samples": len(df_regime),
                "n_days": df_regime["日期"].nunique(),
                "best_iter": model.best_iteration,
            }

            # Save model
            model_path = os.path.join(regime_dir, f"lgb_ranker_{regime}.txt")
            model.save_model(model_path)
            print(f"  Saved: {model_path}")

    # ---- 5. Save shared artifacts ----
    joblib.dump(scaler, os.path.join(regime_dir, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(regime_dir, "feature_cols.pkl"))
    joblib.dump(regime_results, os.path.join(regime_dir, "regime_stats.pkl"))

    # Also save vol_threshold for inference
    with open(os.path.join(regime_dir, "vol_threshold.txt"), "w") as f:
        f.write(str(vol_threshold))

    print(f"\n{'=' * 50}")
    print(f"Regime Training Summary")
    print(f"{'=' * 50}")
    for regime, info in regime_results.items():
        print(f"  {regime.upper():<8s}: NDCG@5={info['ndcg']:.4f}, "
              f"samples={info['n_samples']:,}, days={info['n_days']}, "
              f"best_iter={info['best_iter']}")

    print(f"\nAll artifacts saved to: {regime_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
