"""Training entry point — Phase 6B Bull/NonBull Regime-Split Models.

Trains two LightGBM LambdaRank models (bull + nonbull) with:
  - 102-dim features (Shape + Shock + Regime + Interaction)
  - Multi-horizon ensemble label (T+3/T+5/T+10)
  - Tail-amplified discretization

Output: model/regime6b/lgb_ranker_bull.txt
        model/regime6b/lgb_ranker_nonbull.txt
        model/regime6b/scaler.pkl
        model/regime6b/feature_cols.pkl
"""

import os, sys, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
random.seed(42); np.random.seed(42)

from train_lgb_phase6b import main

if __name__ == "__main__":
    main()
