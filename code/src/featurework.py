"""Feature engineering pipeline — 102-dim features.

Imports all Phase 2-4 feature modules used by the training and prediction pipeline.

Feature categories:
  - Shape (60): 60-day window statistics (slope, autocorr, FFT, up_ratio) for 6 core factors
  - Shock/Event (10): vol_spikes, volume_shocks, gaps, abnormal returns
  - Market Regime (4): trend, vol_regime, breadth, index_slope
  - Cross-stock Interaction (5): beta, sector_rank_delta, leader_correlation
  - Cross-sectional (7): rank_alpha, rank_ret, rank_delta
  - Industry (8): industry_strength, relative_strength, sector_momentum
  - Base (8): raw values of core factors
"""

from features import (
    compute_labels, add_basic_features, add_return_features,
    add_momentum_features, add_volume_features,
    add_market_state_features, add_alpha_features,
    _get_sector, _safe_divide,
)

from features_phase2 import (
    engineer_phase2_features, _window_stats,
    _compute_shape_features_for_stock, _add_rank_delta_features,
)

from features_phase3 import (
    engineer_phase3_features, _add_shock_features,
    _add_regime_features, _add_interaction_features,
    _add_leader_correlation,
)

from features_phase4 import (
    engineer_phase4_features, compute_ensemble_label,
    discretize_labels_tail,
)

from features_phase6b import (
    engineer_phase6b_features, compute_bull_strength,
)
