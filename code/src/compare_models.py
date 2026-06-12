"""Compare Phase 7 vs submit1 on same date (2026-03-06)."""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# Actually use direct path
sys.path.insert(0, 'src')

import joblib, lightgbm as lgb, numpy as np, pandas as pd
from config import CONFIG
from features_phase6b import engineer_phase6b_features
from portfolio_optimizer import PortfolioOptimizer
from features import _get_sector

cfg = CONFIG

# Load data, truncate to submit1 range
df = pd.read_csv('data/train.csv', dtype={'股票代码': str})
df['股票代码'] = df['股票代码'].str.zfill(6)
df['日期'] = pd.to_datetime(df['日期'])
df = df[df['日期'] <= '2026-03-06']

stock_counts = df.groupby('股票代码').size()
df = df[df['股票代码'].isin(stock_counts[stock_counts >= cfg['min_history_days']].index)]
print(f'Data: {len(df)} rows, {df["股票代码"].nunique()} stocks, max={df["日期"].max().date()}')

# Features
df, feature_cols = engineer_phase6b_features(df)
df = df.dropna(subset=['label'])
print(f'After label filter: {len(df)} rows, max={df["日期"].max().date()}')

# Models
rd = 'model/regime6b'
scaler = joblib.load(f'{rd}/scaler.pkl')
bull_m = lgb.Booster(model_file=f'{rd}/lgb_ranker_bull.txt')
nonbull_m = lgb.Booster(model_file=f'{rd}/lgb_ranker_nonbull.txt')

# Predict
ld = df['日期'].max()
pred = df[df['日期'] == ld].copy()
bs = float(pred['bull_strength'].iloc[0])
print(f'Date: {ld.date()}, BS: {bs:.4f}, Regime: {pred["regime"].iloc[0]}')

mc = set(feature_cols) - set(pred.columns)
for c in mc: pred[c] = 0.0
X = scaler.transform(pred[feature_cols].fillna(0).values.astype(np.float32))

br = pd.Series(bull_m.predict(X)).rank(pct=True).values
nr = pd.Series(nonbull_m.predict(X)).rank(pct=True).values
pred['score'] = bs * br + (1-bs) * nr

if 'sector_id' not in pred.columns:
    pred['sector_id'] = pred['股票代码'].apply(_get_sector)

opt = PortfolioOptimizer(
    temperature=cfg['portfolio']['temperature'],
    max_per_industry=cfg['portfolio']['max_per_industry'],
    corr_threshold=cfg['portfolio']['corr_threshold'],
    min_score_ratio=cfg['portfolio']['min_score_ratio'],
    top_n=cfg['portfolio']['top_n'],
    select_k=cfg['portfolio']['select_k'],
    corr_lookback=cfg['portfolio']['corr_lookback'],
)
result = opt.optimize(pred, df, verbose=False)
result['weight'] = (result['weight'] * bs).round(6)

print()
print('=' * 50)
print('Phase 7 on submit1 date (2026-03-06)')
print('=' * 50)
for _, r in result.iterrows():
    print(f'  {r["stock_id"]}: {r["weight"]}')
print(f'  Sum: {result["weight"].sum():.4f}')

print()
print('=' * 50)
print('submit1 (same date)')
print('=' * 50)
for s in ['002384', '300476', '000807', '002463', '002460']:
    print(f'  {s}: 0.2')

p7s = set(result['stock_id'].values)
s1s = {'002384', '300476', '000807', '002463', '002460'}
ov = p7s & s1s
print(f'\nOverlap: {len(ov)}/5 = {ov}')
print(f'P7 only: {p7s - s1s}')
print(f'S1 only: {s1s - p7s}')
