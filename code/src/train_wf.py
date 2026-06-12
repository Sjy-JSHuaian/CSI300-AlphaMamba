"""Walk-forward cross-validation — Phase 2 Temporal Shape Features.

2-Fold expanding-window validation with fixed date splits:
  Fold1: Train 2022-2024, Valid 2025H1
  Fold2: Train 2022-2025H1, Valid 2025H2

Metrics: NDCG@5, Top5 return, RankIC

Usage: python src/train_wf.py
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
from features_phase4 import discretize_labels_tail, engineer_phase4_features

warnings.filterwarnings("ignore")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_rank_ic(group_df: pd.DataFrame) -> float:
    """Compute daily Rank IC (Spearman correlation pred vs label)."""
    daily_ics = []
    for _, g in group_df.groupby("日期"):
        if len(g) < 10:
            continue
        ic, _ = spearmanr(g["pred"], g["label"])
        daily_ics.append(ic)
    return np.mean(daily_ics) if daily_ics else 0.0


def train_one_fold(train_df, val_df, feature_cols, cfg, fold_name):
    """Train LightGBM on one fold, return model, scaler, and metrics."""
    # Deduplicate: ensure no overlapping dates
    train_dates = set(train_df["日期"].unique())
    val_dates = set(val_df["日期"].unique())
    overlap = train_dates & val_dates
    if overlap:
        print(f"  WARNING: {len(overlap)} overlapping dates, removing from val")
        val_df = val_df[~val_df["日期"].isin(overlap)].copy()

    # Scale
    scaler = StandardScaler()
    train_feat = scaler.fit_transform(train_df[feature_cols].fillna(0).values)
    val_feat = scaler.transform(val_df[feature_cols].fillna(0).values)

    # Discretize labels
    train_df = discretize_labels_tail(train_df.copy())
    val_df = discretize_labels_tail(val_df.copy())

    # Sort by date for group integrity
    train_df = train_df.sort_values(["日期", "股票代码"])
    val_df = val_df.sort_values(["日期", "股票代码"])

    train_groups = train_df.groupby("日期", sort=False).size().values
    val_groups = val_df.groupby("日期", sort=False).size().values

    X_train = train_feat.astype(np.float32)
    y_train = train_df["relevance"].values.astype(np.int32)
    X_val = val_feat.astype(np.float32)
    y_val = val_df["relevance"].values.astype(np.int32)

    train_ds = lgb.Dataset(X_train, label=y_train, group=train_groups)
    val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_ds)

    params = cfg["lgb_params"].copy()

    model = lgb.train(
        params=params,
        train_set=train_ds,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(cfg["early_stopping_rounds"]),
            lgb.log_evaluation(0),
        ],
    )

    # Predict
    val_preds = model.predict(X_val)
    val_eval = val_df.copy()
    val_eval["pred"] = val_preds
    # Restore original continuous labels for scoring
    val_eval["label"] = val_df["label"].values

    # Metrics
    ndcg = model.best_score["val"]["ndcg@5"]
    rank_ic = compute_rank_ic(val_eval)

    # Top-5 evaluation
    daily_scores = []
    for date, group in val_eval.groupby("日期"):
        if len(group) < 5:
            continue
        pred_top5 = group.nlargest(5, "pred")["label"].sum()
        true_top5 = group.nlargest(5, "label")["label"].sum()
        random5 = 5 * group["label"].mean()
        denom = true_top5 - random5
        score = (pred_top5 - random5) / (denom + 1e-12) if abs(denom) > 1e-6 else 0.0
        daily_scores.append({
            "date": date,
            "fold": fold_name,
            "pred_return": pred_top5,
            "true_return": true_top5,
            "random_return": random5,
            "final_score": score,
        })

    return model, scaler, daily_scores, ndcg, rank_ic


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ---- 1. Load and preprocess ----
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

    # ---- 2. Phase 3 feature engineering ----
    df, feature_cols = engineer_phase4_features(df)
    df = df.dropna(subset=["label"])
    df[feature_cols] = df[feature_cols].fillna(0.0)

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(df)}")

    # ---- 3. Build fixed-date folds ----
    fold_splits = cfg["wf_fold_splits"]
    df = df.sort_values("日期")

    folds = []
    for i, split in enumerate(fold_splits):
        train_end = pd.Timestamp(split["train_end"])
        val_start = pd.Timestamp(split["val_start"])
        val_end = pd.Timestamp(split["val_end"])

        train_mask = df["日期"] <= train_end
        val_mask = (df["日期"] >= val_start) & (df["日期"] <= val_end)

        fold_name = (
            f"F{i+1}_train-{split['train_end'][:4]}"
            f"_val-{val_start.strftime('%Y%m')}-{val_end.strftime('%Y%m')}"
        )

        n_train = train_mask.sum()
        n_val = val_mask.sum()
        if n_train < 1000 or n_val < 100:
            print(f"  Skipping {fold_name}: insufficient data (train={n_train}, val={n_val})")
            continue

        folds.append((train_mask, val_mask, fold_name))
        print(f"  {fold_name}: train={n_train}, val={n_val}")

    print(f"\nWalk-Forward Validation: {len(folds)} folds")

    # ---- 4. Run walk-forward CV ----
    all_scores = []
    best_ndcg = -float("inf")
    best_model = None
    best_scaler = None

    for train_mask, val_mask, fold_name in folds:
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()

        print(f"\n{'='*50}")
        print(f"Training {fold_name}")
        print(f"  Train: {len(train_df)} rows, {train_df['日期'].nunique()} days "
              f"({train_df['日期'].min().date()} to {train_df['日期'].max().date()})")
        print(f"  Val:   {len(val_df)} rows, {val_df['日期'].nunique()} days "
              f"({val_df['日期'].min().date()} to {val_df['日期'].max().date()})")

        model, scaler, scores, ndcg, rank_ic = train_one_fold(
            train_df, val_df, feature_cols, cfg, fold_name
        )
        all_scores.extend(scores)

        mean_score = np.mean([s["final_score"] for s in scores]) if scores else 0.0
        mean_pred_ret = np.mean([s["pred_return"] for s in scores]) if scores else 0.0
        mean_true_ret = np.mean([s["true_return"] for s in scores]) if scores else 0.0

        print(f"  NDCG@5:    {ndcg:.4f}")
        print(f"  Final Score: {mean_score:.4f}")
        print(f"  Rank IC:   {rank_ic:.4f}")
        print(f"  Pred Top-5 Return: {mean_pred_ret:.4f}")
        print(f"  True Top-5 Return: {mean_true_ret:.4f}")

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_model = model
            best_scaler = scaler

    # ---- 5. Aggregate results ----
    scores_df = pd.DataFrame(all_scores)
    print(f"\n{'='*50}")
    print("Aggregate Walk-Forward Results")
    print(f"{'='*50}")

    if len(scores_df) > 0:
        for fold in scores_df["fold"].unique():
            fs = scores_df[scores_df["fold"] == fold]
            print(f"  {fold}:")
            print(f"    avg final_score = {fs['final_score'].mean():.4f} "
                  f"(std={fs['final_score'].std():.4f})")
            print(f"    avg pred_return = {fs['pred_return'].mean():.4f}")
            print(f"    avg true_return = {fs['true_return'].mean():.4f}")

        print(f"\n  Overall mean final_score: {scores_df['final_score'].mean():.4f}")
        print(f"  Overall std final_score:  {scores_df['final_score'].std():.4f}")

    # ---- 6. Save best model ----
    if best_model is not None:
        model_path = os.path.join(cfg["output_dir"], "lgb_ranker.txt")
        best_model.save_model(model_path)
        joblib.dump(best_scaler, os.path.join(cfg["output_dir"], "scaler.pkl"))
        joblib.dump(feature_cols, os.path.join(cfg["output_dir"], "feature_cols.pkl"))
        print(f"\nBest model saved to: {model_path}")

        # Feature importance
        importance = pd.DataFrame({
            "feature": feature_cols,
            "gain": best_model.feature_importance(importance_type="gain"),
        }).sort_values("gain", ascending=False)
        print("\n--- Top 15 Features ---")
        for _, row in importance.head(15).iterrows():
            print(f"  {row['feature']:<35s} {row['gain']:>10.0f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
