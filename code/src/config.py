"""All configuration for Phase 2 LightGBM Ranker — Temporal Shape Features."""

CONFIG = {
    # ---- Data ----
    "data_path": "./data",
    "output_dir": "./model",
    "output_csv": "./output/result.csv",

    # ---- Feature engineering ----
    "min_history_days": 120,       # Stocks need at least this many trading days
    "return_windows": [1, 5, 10, 20, 60],
    "volatility_windows": [5, 10, 20, 60],
    "trend_windows": [5, 10, 20, 60],
    "ema_windows": [12, 26],

    # ---- Walk-Forward CV (Phase 2) ----
    "wf_fold_splits": [
        {"train_end": "2024-12-31", "val_start": "2025-01-01", "val_end": "2025-06-30"},
        {"train_end": "2025-06-30", "val_start": "2025-07-01", "val_end": "2025-12-31"},
    ],

    # ---- Train/Val split ----
    "val_months": 2,               # Last N months for validation

    # ---- LightGBM Ranker (Phase 2 tuned) ----
    "lgb_params": {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],       # Optimize for top-5 only (matches competition)
        "num_leaves": 255,
        "learning_rate": 0.03,
        "n_estimators": 3000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
        "min_child_samples": 50,
        "min_child_weight": 1e-3,
        "verbosity": -1,
        "random_state": 42,
        "force_row_wise": True,    # Faster training
    },
    "early_stopping_rounds": 100,
    "log_evaluation": 50,

    # ---- Prediction ----
    "top_k": 5,
    "equal_weight": 0.2,

    # ---- Portfolio Optimization (Phase 5) ----
    "portfolio": {
        "temperature": 0.5,          # Lower → more concentrated weights
        "max_per_industry": 2,       # Max stocks per board-level sector
        "corr_threshold": 0.8,       # Correlation above this → score penalty
        "min_score_ratio": 0.7,      # Exclude stocks with score < 70% of top-1
        "top_n": 10,                 # Candidate pool size
        "select_k": 5,               # Target portfolio size
        "corr_lookback": 60,         # Trading days for correlation estimation
    },

    # ---- Phase 6A: Regime-Split Modeling (3-class, deprecated) ----
    "phase6": {
        "vol_threshold": 0.02,       # Daily vol threshold for bull/bear distinction
        "val_ratio": 0.2,            # Fraction of dates per regime for validation
    },

    # ---- Phase 6B: Bull/Non-Bull + Soft Blend ----
    "phase6b": {
        "val_ratio": 0.2,            # Fraction of dates per regime for validation
        "blend_smoothing": 0.0,      # Extra smoothing on bull_strength (0=none)
    },

    # ---- Phase 7: Safe Mode ----
    "phase7": {
        "bull_gate": 0.6,            # bull_strength < this → 100% cash
        "turnover_penalty": 0.02,    # Score penalty for new positions
        "persistence_bonus": 0.01,   # Score bonus for existing positions
    },

    # ---- Random seed ----
    "seed": 42,
}
