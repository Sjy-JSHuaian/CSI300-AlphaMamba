"""Train LightGBM Ranker — Phase 2 Temporal Shape Features.

Usage: python src/train_lgb.py
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


def compute_rank_ic(df: pd.DataFrame) -> float:
    """Compute daily Rank IC (Spearman correlation pred vs label)."""
    daily_ics = []
    for _, group in df.groupby("日期"):
        if len(group) < 10:
            continue
        ic, _ = spearmanr(group["pred"], group["label"])
        daily_ics.append(ic)
    return np.mean(daily_ics) if daily_ics else 0.0


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ---- 1. Load data ----
    print("Loading data...")
    data_path = os.path.join(cfg["data_path"], "train.csv")
    df = pd.read_csv(data_path, dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"])

    print(f"Raw data: {len(df)} rows, {df['股票代码'].nunique()} stocks")
    print(f"Date range: {df['日期'].min().date()} to {df['日期'].max().date()}")

    # ---- 2. Filter stocks with insufficient history ----
    stock_counts = df.groupby("股票代码").size()
    valid_stocks = stock_counts[stock_counts >= cfg["min_history_days"]].index
    df = df[df["股票代码"].isin(valid_stocks)]
    n_stocks = df["股票代码"].nunique()
    print(f"After min-history filter: {n_stocks} stocks")

    # ---- 3. Phase 4 feature engineering ----
    df, feature_cols = engineer_phase4_features(df)

    # ---- 4. Train/Val split by date ----
    df = df.sort_values(["日期", "股票代码"]).reset_index(drop=True)
    last_date = df["日期"].max()
    val_start = last_date - pd.DateOffset(months=cfg["val_months"])
    val_start = df[df["日期"] >= val_start]["日期"].min()

    train_mask = df["日期"] < val_start
    val_mask = df["日期"] >= val_start

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    n_train_days = train_df["日期"].nunique()
    n_val_days = val_df["日期"].nunique()
    print(f"\nTrain: {len(train_df)} rows, {n_train_days} days")
    print(f"Val:   {len(val_df)} rows, {n_val_days} days")
    print(f"Features: {len(feature_cols)}")
    print(f"Val starts: {val_start.date()}")

    if n_val_days < 1:
        raise ValueError("Validation set is empty — check val_months setting")

    # ---- 5. Scale features ----
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols].fillna(0))
    val_df[feature_cols] = scaler.transform(val_df[feature_cols].fillna(0))

    # Save artifacts
    joblib.dump(scaler, os.path.join(cfg["output_dir"], "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(cfg["output_dir"], "feature_cols.pkl"))
    print(f"Saved scaler and {len(feature_cols)} feature names")

    # ---- 6. Discretize labels (tail-amplified for top-5 focus) ----
    train_df = discretize_labels_tail(train_df)
    val_df = discretize_labels_tail(val_df)

    # ---- 7. Prepare LightGBM Datasets ----
    train_df = train_df.sort_values(["日期", "股票代码"])
    val_df = val_df.sort_values(["日期", "股票代码"])

    train_groups = train_df.groupby("日期", sort=False).size().values
    val_groups = val_df.groupby("日期", sort=False).size().values

    print(f"\nTrain groups: {len(train_groups)}, avg group size: {train_groups.mean():.0f}")
    print(f"Val groups:   {len(val_groups)}, avg group size: {val_groups.mean():.0f}")

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["relevance"].values.astype(np.int32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["relevance"].values.astype(np.int32)

    train_dataset = lgb.Dataset(
        X_train, label=y_train, group=train_groups,
    )
    val_dataset = lgb.Dataset(
        X_val, label=y_val, group=val_groups,
        reference=train_dataset,
    )

    # ---- 8. Train LightGBM Ranker ----
    print("\nTraining LightGBM Ranker...")

    callbacks = [
        lgb.early_stopping(cfg["early_stopping_rounds"]),
        lgb.log_evaluation(cfg["log_evaluation"]),
    ]

    model = lgb.train(
        params=cfg["lgb_params"],
        train_set=train_dataset,
        valid_sets=[train_dataset, val_dataset],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # ---- 9. Save model ----
    model_path = os.path.join(cfg["output_dir"], "lgb_ranker.txt")
    model.save_model(model_path)
    print(f"\nModel saved to: {model_path}")

    # ---- 10. Validation evaluation ----
    val_preds = model.predict(X_val)
    val_eval = val_df.copy()
    val_eval["pred"] = val_preds
    val_eval["label"] = val_df["label"].values  # original continuous label

    # NDCG
    ndcg = model.best_score["val"]["ndcg@5"]
    print(f"\n--- Validation Metrics ---")
    print(f"  Best NDCG@5: {ndcg:.4f}")

    # Top-5 score
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
            "pred_return": pred_top5,
            "true_return": true_top5,
            "random_return": random5,
            "final_score": score,
        })

    scores_df = pd.DataFrame(daily_scores)
    if len(scores_df) > 0:
        print(f"  Mean pred top-5 return:  {scores_df['pred_return'].mean():.4f}")
        print(f"  Mean true top-5 return:  {scores_df['true_return'].mean():.4f}")
        print(f"  Mean final score:        {scores_df['final_score'].mean():.4f}")

    # Rank IC
    rank_ic = compute_rank_ic(val_eval)
    print(f"  Mean Rank IC:            {rank_ic:.4f}")

    # ---- 11. Feature importance ----
    importance = pd.DataFrame({
        "feature": feature_cols,
        "gain": model.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)

    print("\n--- Top 20 Features by Gain ---")
    for _, row in importance.head(20).iterrows():
        print(f"  {row['feature']:<35s} {row['gain']:>12.0f}")

    importance_path = os.path.join(cfg["output_dir"], "feature_importance.csv")
    importance.to_csv(importance_path, index=False)
    print(f"\nFeature importance saved to: {importance_path}")

    # Save final score
    if len(scores_df) > 0:
        final_score = scores_df["final_score"].mean()
        score_path = os.path.join(cfg["output_dir"], "final_score.txt")
        with open(score_path, "w") as f:
            f.write(f"val_final_score: {final_score:.6f}\n")
            f.write(f"val_ndcg5: {ndcg:.6f}\n")
            f.write(f"val_rank_ic: {rank_ic:.6f}\n")
            f.write(f"val_days: {len(scores_df)}\n")
            f.write(f"features: {len(feature_cols)}\n")
        print(f"Final score saved to: {score_path}")

    return model


if __name__ == "__main__":
    main()
