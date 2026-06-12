"""Meta Ranker V2 Walk-Forward Backtest.

2-fold walk-forward:
  Fold N: Train Phase 6B on train data → generate P6B scores for all dates
          → Build candidate pools → Train Ridge Meta Ranker on train dates
          → Predict val weeks → Portfolio Optimizer + Phase 7 risk

Compares: Meta Ranker V2 vs Consensus Alpha vs Phase 6B vs submit1

Usage: python src/backtest_meta_ranker.py
Output: output/meta_v2_equity.csv, output/meta_v2_report.csv
"""

import os, random, warnings
from itertools import groupby
from typing import Dict, List, Optional, Set
import joblib, lightgbm as lgb, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from config import CONFIG
from features_phase6b import engineer_phase6b_features
from features_phase4 import discretize_labels_tail
from portfolio_optimizer import PortfolioOptimizer
from meta_ranker import _make_features, _feature_names, STRUCTURAL_COLS, STRUCTURAL_FALLBACK
from features import _get_sector
warnings.filterwarnings("ignore")

BULL_GATE = 0.6; TURNOVER_PENALTY = 0.02; PERSISTENCE_BONUS = 0.01; TOP_N = 40

def set_seed(s=42): random.seed(s); np.random.seed(s); os.environ["PYTHONHASHSEED"] = str(s)
def _fwd_ret(df):
    df = df.copy(); res = []
    for sid, g in df.groupby("股票代码", sort=False):
        g = g.sort_values("日期").copy()
        g["forward_ret"] = (g["开盘"].shift(-5) - g["开盘"].shift(-1)) / g["开盘"].shift(-1).replace(0, np.nan)
        res.append(g)
    return pd.concat(res, ignore_index=True)
def _weekly(dates):
    dt = pd.DatetimeIndex(dates); iso = dt.isocalendar()
    return pd.Series(list(zip(iso["year"], iso["week"])), index=dates.index)
def _train_one(train_df, fc, params, esr=100):
    df_r = train_df.sort_values(["日期", "股票代码"]).copy()
    ad = sorted(df_r["日期"].unique()); nv = max(1, int(len(ad)*0.2))
    if len(ad) < 10: return None
    vd = set(ad[-nv:]); td = set(ad[:-nv])
    st = df_r[df_r["日期"].isin(td)]; sv = df_r[df_r["日期"].isin(vd)]
    st = discretize_labels_tail(st); sv = discretize_labels_tail(sv)
    return lgb.train(params=params, train_set=lgb.Dataset(st[fc].fillna(0).values.astype(np.float32), st["relevance"].values.astype(np.int32), group=st.groupby("日期", sort=False).size().values),
                     valid_sets=[lgb.Dataset(sv[fc].fillna(0).values.astype(np.float32), sv["relevance"].values.astype(np.int32), group=sv.groupby("日期", sort=False).size().values)],
                     valid_names=["train","val"], callbacks=[lgb.early_stopping(esr), lgb.log_evaluation(0)])
def metrics(eq, rf=0.02):
    rets = eq["week_return"].values; bench = eq["benchmark_return"].values; n = len(rets)
    if n == 0: return {}
    cum = np.cumprod(1+rets)-1; total = float(cum[-1]); avg = float(np.mean(rets))
    ex = rets - rf/52; sh = float(ex.mean()/ex.std()*np.sqrt(52)) if ex.std()>1e-12 else 0
    dd_std = np.std(rets[rets<0]) if (rets<0).any() else 0
    so = float(ex.mean()/dd_std*np.sqrt(52)) if dd_std>1e-12 else 0
    eq_arr = np.concatenate([[1.0], 1+rets]); peak = np.maximum.accumulate(eq_arr)
    mdd = float(np.min((eq_arr-peak)/peak))
    signs = np.sign(rets); runs = [(k,len(list(g))) for k,g in groupby(signs)]
    return {"n_weeks":n,"total_return":round(total,4),"excess_return":round(total-float(np.cumprod(1+bench)-1)[-1],4),
            "sharpe":round(sh,4),"sortino":round(so,4),"max_dd":round(mdd,4),
            "win_rate":round(float(np.mean(rets>0)),4),"beat_rate":round(float(np.mean(rets>bench)),4)}

def main():
    cfg = CONFIG; set_seed(cfg["seed"]); os.makedirs("output", exist_ok=True)

    # Load submit1 scores
    s1p = "model/submit1_scores.csv"
    s1_df = pd.read_csv(s1p, dtype={"股票代码":str}) if os.path.exists(s1p) else None
    if s1_df is not None: s1_df["股票代码"]=s1_df["股票代码"].str.zfill(6); s1_df["日期"]=pd.to_datetime(s1_df["日期"])

    # Load data
    df = pd.read_csv("data/train.csv", dtype={"股票代码":str})
    df["股票代码"]=df["股票代码"].str.zfill(6); df["日期"]=pd.to_datetime(df["日期"])
    sc = df.groupby("股票代码").size(); df = df[df["股票代码"].isin(sc[sc>=cfg["min_history_days"]].index)]

    print("Feature engineering...")
    df, feature_cols = engineer_phase6b_features(df)
    df = df.dropna(subset=["label"]); df[feature_cols]=df[feature_cols].fillna(0.0)
    df = _fwd_ret(df)
    if ((df["forward_ret"]<-1.0)|(df["forward_ret"]>10.0)).any(): df["forward_ret"]=df["forward_ret"].clip(-1.0,10.0)

    folds = []
    for s in cfg["wf_fold_splits"]:
        tr = df["日期"]<=pd.Timestamp(s["train_end"])
        vl = (df["日期"]>=pd.Timestamp(s["val_start"]))&(df["日期"]<=pd.Timestamp(s["val_end"]))
        folds.append((tr, vl, f"F{len(folds)+1}"))

    pc = cfg["portfolio"]
    opt = PortfolioOptimizer(temperature=pc["temperature"], max_per_industry=pc["max_per_industry"],
                             corr_threshold=pc["corr_threshold"], min_score_ratio=pc["min_score_ratio"],
                             top_n=pc["top_n"], select_k=pc["select_k"], corr_lookback=pc["corr_lookback"],
                             turnover_penalty=TURNOVER_PENALTY, persistence_bonus=PERSISTENCE_BONUS)
    params = cfg["lgb_params"].copy()
    all_weeks = []

    for fi, (tr_m, vl_m, fn) in enumerate(folds):
        print(f"\n{'='*50}\nFold {fi+1}: {fn}\n{'='*50}")
        tr_df = df[tr_m].copy(); vl_df = df[vl_m].copy()
        vl_df["wl"] = _weekly(vl_df["日期"])

        # Train Phase 6B on THIS fold's training data
        scaler = StandardScaler(); scaler.fit(tr_df[feature_cols].fillna(0).values)
        bt = tr_df[tr_df["regime"]=="bull"].copy(); nt = tr_df[tr_df["regime"]=="nonbull"].copy()
        for d in [bt, nt]:
            if len(d)>0: d[feature_cols] = scaler.transform(d[feature_cols].fillna(0).values)
        bm = _train_one(bt, feature_cols, "bull", params, cfg["early_stopping_rounds"])
        nm = _train_one(nt, feature_cols, "nonbull", params, cfg["early_stopping_rounds"])
        if bm is None or nm is None: continue

        # Generate P6B scores for ALL dates in this fold
        print("  Generating P6B scores...")
        all_fold_dates = sorted(set(tr_df["日期"].unique()) | set(vl_df["日期"].unique()))
        p6b_map, s1_map = {}, {}
        for date in all_fold_dates:
            day = df[df["日期"]==date]
            if len(day) < 5: continue
            mc = set(feature_cols)-set(day.columns)
            for c in mc: day[c]=0.0
            X = scaler.transform(day[feature_cols].fillna(0).values.astype(np.float32))
            br = pd.Series(bm.predict(X)).rank(pct=True).values
            nr = pd.Series(nm.predict(X)).rank(pct=True).values
            bs = float(day["bull_strength"].iloc[0]) if "bull_strength" in day.columns else 0.5
            for sid, sc in zip(day["股票代码"], bs*br+(1-bs)*nr):
                p6b_map[(date, sid)] = float(sc)
        if s1_df is not None:
            for _, r in s1_df[s1_df["日期"].isin(all_fold_dates)].iterrows():
                s1_map[(r["日期"], r["股票代码"])] = r["submit1_score"]

        # Build candidate pools for TRAINING dates
        print(f"  Building candidate pools (top-{TOP_N})...")
        tr_dates = sorted(tr_df["日期"].unique())
        meta_rows = []
        for date in tr_dates:
            day = df[df["日期"]==date].copy()
            if len(day) < 20: continue
            day["p6b_score"] = day["股票代码"].map(lambda s: p6b_map.get((date, s), 0.0))
            day["s1_score"] = day["股票代码"].map(lambda s: s1_map.get((date, s), np.nan))
            # Build pool
            p6b_set = set(day.nlargest(TOP_N, "p6b_score")["股票代码"])
            s1_set = set()
            if day["s1_score"].notna().any():
                s1_set = set(day.nlargest(TOP_N, "s1_score")["股票代码"])
            pool_ids = p6b_set | s1_set
            if len(pool_ids) < 10: continue
            pool = day[day["股票代码"].isin(pool_ids)].copy()
            pool = _make_features(pool, df, STRUCTURAL_COLS)
            pool["label"] = day.set_index("股票代码").reindex(pool["股票代码"])["label"].values
            meta_rows.append(pool)

        if not meta_rows: continue
        meta_tr = pd.concat(meta_rows, ignore_index=True)
        feats = _feature_names(STRUCTURAL_COLS)
        feats = [c for c in feats if c in meta_tr.columns]
        for col in feats:
            if meta_tr[col].isna().any(): meta_tr[col]=meta_tr[col].fillna(STRUCTURAL_FALLBACK.get(col,0.0))

        # Split meta-train into train/val for ridge
        meta_dates = sorted(meta_tr["日期"].unique())
        n_mv = max(1, int(len(meta_dates)*0.15))
        mt_dates = set(meta_dates[:-n_mv]); mv_dates = set(meta_dates[-n_mv:])
        mtr = meta_tr[meta_tr["日期"].isin(mt_dates)]; mtv = meta_tr[meta_tr["日期"].isin(mv_dates)]

        m_scaler = StandardScaler()
        Xt = m_scaler.fit_transform(mtr[feats].fillna(0).values.astype(np.float64))
        yt = mtr["label"].values.astype(np.float64)
        Xv = m_scaler.transform(mtv[feats].fillna(0).values.astype(np.float64))
        yv = mtv["label"].values.astype(np.float64)
        ridge = Ridge(alpha=1.0); ridge.fit(Xt, yt)
        from scipy.stats import spearmanr
        ic, _ = spearmanr(ridge.predict(Xv), yv)
        print(f"  Meta Ridge Val IC: {ic:.4f}")

        # Predict each week
        prev_pf = None
        for wl, wdf in vl_df.groupby("wl", sort=True):
            pd_date = wdf["日期"].min(); pred = wdf[wdf["日期"]==pd_date].copy()
            if len(pred) < 5: continue
            bs = float(pred["bull_strength"].iloc[0])
            bench_ret = float(pred["forward_ret"].mean())
            if "sector_id" not in pred.columns: pred["sector_id"]=pred["股票代码"].apply(_get_sector)

            if bs < BULL_GATE:
                all_weeks.append({"year":int(wl[0]),"week":int(wl[1]),"date":pd_date,"fold":fn,
                                  "bs":round(bs,4),"mode":"CASH","week_return":0.0,"bench":round(bench_ret,6)})
                continue

            pred["p6b_score"] = pred["股票代码"].map(lambda s: p6b_map.get((pd_date, s), 0.0))
            pred["s1_score"] = pred["股票代码"].map(lambda s: s1_map.get((pd_date, s), np.nan))

            # Build candidate pool
            p6b_s = set(pred.nlargest(TOP_N, "p6b_score")["股票代码"])
            s1_s = set(pred.nlargest(TOP_N, "s1_score")["股票代码"]) if pred["s1_score"].notna().any() else set()
            pool = pred[pred["股票代码"].isin(p6b_s|s1_s)].copy()
            pool = _make_features(pool, df, STRUCTURAL_COLS)
            for col in feats:
                if col not in pool.columns: pool[col]=STRUCTURAL_FALLBACK.get(col,0.0)
            pool[feats]=pool[feats].fillna(0)
            Xm = m_scaler.transform(pool[feats].values.astype(np.float64))
            sm = dict(zip(pool["股票代码"], ridge.predict(Xm)))
            pred["score"] = pred["股票代码"].map(sm).fillna(-9.0)

            try:
                result = opt.optimize(pred, df, prev_portfolio=prev_pf, verbose=False)
            except Exception:
                result = pd.DataFrame(columns=["stock_id","weight"])
            if len(result)==0: continue
            result["weight"] = result["weight"] * bs
            rets = [float(pred[pred["股票代码"]==r["stock_id"]]["forward_ret"].values[0]) if len(pred[pred["股票代码"]==r["stock_id"]])>0 else 0 for _,r in result.iterrows()]
            wr = float(np.dot(result["weight"].values, rets))
            cur = set(result["stock_id"].values)
            to = 1.0-len(cur&prev_pf)/max(len(cur),len(prev_pf)) if prev_pf and len(prev_pf)>0 else 1.0
            prev_pf = cur
            all_weeks.append({"year":int(wl[0]),"week":int(wl[1]),"date":pd_date,"fold":fn,
                              "bs":round(bs,4),"mode":"INVESTED","week_return":round(wr,6),
                              "bench":round(bench_ret,6),"turnover":round(to,4)})

    if not all_weeks: print("No weeks!"); return
    eq = pd.DataFrame(all_weeks).sort_values(["year","week"]).reset_index(drop=True)
    eq["cum_return"] = (1+eq["week_return"]).cumprod()-1; eq["bench_cum"] = (1+eq["bench"]).cumprod()-1
    eq.to_csv("output/meta_v2_equity.csv", index=False)

    invested = eq[eq["mode"]=="INVESTED"]
    cash_pct = (eq["mode"]=="CASH").mean()
    m_all = metrics(pd.DataFrame({"week_return":eq["week_return"],"benchmark_return":eq["bench"]}))
    m_inv = metrics(pd.DataFrame({"week_return":invested["week_return"],"benchmark_return":invested["bench"]})) if len(invested)>0 else {}

    rows = [{"strategy":"MetaRankerV2","metric":k,"value":v} for k,v in {**m_all,"cash_ratio":round(cash_pct,4)}.items()]
    pd.DataFrame(rows).to_csv("output/meta_v2_report.csv", index=False)

    print(f"\n{'='*60}\nMeta Ranker V2 Backtest ({len(eq)} weeks)\n{'='*60}")
    print(f"  Return: {m_all.get('total_return',0):.2%}  |  Sharpe: {m_all.get('sharpe',0):.2f}  |  MaxDD: {m_all.get('max_dd',0):.2%}")
    print(f"  Win: {m_all.get('win_rate',0):.1%}  |  Cash: {cash_pct:.0%}  |  Beat: {m_all.get('beat_rate',0):.1%}")
    if m_inv: print(f"  Invested: Return={m_inv.get('total_return',0):.2%}  Sharpe={m_inv.get('sharpe',0):.2f}")
    print("Done!")

if __name__=="__main__": main()
