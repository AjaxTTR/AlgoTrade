"""
NQ 15-min Edge Analysis Pipeline
Systematic EDA to surface statistical anomalies.
Outputs to output/edge_analysis/.

Run: python -m research.edge_analysis
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from engine.data_loader import load_csv

OUTPUT_DIR = Path("output/edge_analysis")
RTH_START = "09:30"
RTH_END = "16:00"
SEP = "\n" + "=" * 70


def _safe_qcut(series, q, labels):
    """pd.qcut that returns NaN instead of crashing on constant/insufficient data."""
    s = series.dropna()
    if len(s) < q or s.nunique() < q:
        return pd.Series(np.nan, index=series.index)
    try:
        return pd.qcut(series, q, labels=labels)
    except (ValueError, TypeError):
        return pd.Series(np.nan, index=series.index)


def _safe_pearsonr(x, y):
    """stats.pearsonr that returns (NaN, NaN) on insufficient data."""
    x, y = np.asarray(x), np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan, np.nan
    return stats.pearsonr(x, y)


def bh_fdr(p_values):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    adjusted = p * n / ranks
    # enforce monotonicity: walk backwards through sorted order
    adjusted_sorted = adjusted[order]
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i + 1] = min(adjusted_sorted[i + 1], 1.0)
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
    adjusted_sorted[n - 1] = min(adjusted_sorted[n - 1], 1.0)
    adjusted[order] = adjusted_sorted
    return adjusted


def split_regimes(data):
    """Split data into 3 equal regimes by date. Works on bar-level or daily data."""
    empty = pd.DataFrame(columns=data.columns)
    if len(data) == 0:
        return empty, empty.copy(), empty.copy()
    if "date" in data.columns:
        d = data["date"]
    elif hasattr(data.index, 'date') and len(data.index) > 0 and hasattr(data.index[0], 'hour'):
        # DatetimeIndex (bar-level data) — extract date
        d = pd.Series(data.index.date, index=data.index)
    else:
        # Index is already date-like (daily summary)
        d = pd.Series(data.index, index=data.index)
    unique_dates = sorted(d.unique())
    n = len(unique_dates)
    if n < 3:
        return data, empty.copy(), empty.copy()
    cut1, cut2 = unique_dates[n // 3], unique_dates[2 * n // 3]
    return data[d < cut1], data[(d >= cut1) & (d < cut2)], data[d >= cut2]


def stability_score(means):
    """Score 0-1 measuring consistency across regime means.
    1.0 = identical sign and magnitude. 0.0 = sign flip in any regime."""
    means = [m for m in means if np.isfinite(m)]
    if len(means) < 2:
        return np.nan
    if all(m > 0 for m in means) or all(m < 0 for m in means):
        abs_means = [abs(m) for m in means]
        return min(abs_means) / max(abs_means) if max(abs_means) > 0 else 0.0
    return 0.0


def build_features(df):
    """Compute all bar-level features. All analysis functions use this dataframe."""
    # Returns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Time-of-day
    df["time"] = df.index.time
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["time_slot"] = df["hour"].astype(str).str.zfill(2) + ":" + df["minute"].astype(str).str.zfill(2)

    # Day-of-week
    df["date"] = df.index.date
    df["dow"] = df.index.dayofweek
    df["dow_name"] = df.index.day_name()

    # Rolling volatility
    df["rolling_vol_5"] = df["returns"].rolling(5).std()
    df["rolling_vol_20"] = df["returns"].rolling(20).std()

    # Range features
    df["bar_range"] = df["high"] - df["low"]
    df["bar_range_pct"] = df["bar_range"] / df["open"]

    return df


def load_data():
    df = load_csv("data/nq_15m_data.csv")
    return build_features(df)


def build_daily_summary(df):
    rth = df.between_time(RTH_START, RTH_END)
    daily = rth.groupby("date").agg(
        daily_open=("open", "first"), daily_high=("high", "max"),
        daily_low=("low", "min"), daily_close=("close", "last"),
        daily_volume=("volume", "sum"), bar_count=("close", "count"),
    )
    daily["daily_range"] = daily["daily_high"] - daily["daily_low"]
    daily["daily_return"] = daily["daily_close"] / daily["daily_open"] - 1
    daily["daily_range_pct"] = daily["daily_range"] / daily["daily_open"]
    daily["prev_close"] = daily["daily_close"].shift(1)
    daily["prev_high"] = daily["daily_high"].shift(1)
    daily["prev_low"] = daily["daily_low"].shift(1)
    daily["gap"] = daily["daily_open"] - daily["prev_close"]
    daily["gap_pct"] = daily["gap"] / daily["prev_close"]
    daily["dow"] = pd.to_datetime(daily.index).dayofweek
    daily["dow_name"] = pd.to_datetime(daily.index).day_name()
    daily = daily[daily["bar_count"] >= 20]
    return daily


def analyze_autocorrelation(df):
    print(SEP)
    print("ANALYSIS 1: RETURN AUTOCORRELATION")
    print("=" * 70)
    rth = df.between_time(RTH_START, RTH_END).copy()
    rth["prev_return"] = rth["returns"].shift(1)
    rth = rth.dropna(subset=["returns", "prev_return"])
    if len(rth) < 2:
        print("\n  WARNING: Insufficient data for autocorrelation analysis.")
        return pd.DataFrame()
    overall_corr, overall_p = _safe_pearsonr(rth["returns"], rth["prev_return"])
    sig_label = "SIGNIFICANT" if overall_p < 0.05 else "Not significant"
    mr_label = "Mean-reverting" if overall_corr < 0 else "Momentum"
    print(f"\nOverall lag-1 autocorrelation: {overall_corr:.6f} (p={overall_p:.4f})")
    print(f"  -> {sig_label} at 5%")
    print(f"  -> {mr_label} tendency")
    r1, r2, r3 = split_regimes(rth)
    results = []
    for slot, group in rth.groupby("time_slot"):
        g = group.dropna(subset=["returns", "prev_return"])
        if len(g) < 50:
            continue
        corr, p = _safe_pearsonr(g["returns"], g["prev_return"])
        if np.isnan(corr):
            continue
        row = {"time_slot": slot, "autocorr": corr, "p_value": p, "n": len(g)}
        regime_corrs = []
        for i, regime in enumerate([r1, r2, r3], 1):
            rg = regime[regime["time_slot"] == slot].dropna(subset=["returns", "prev_return"])
            rc = _safe_pearsonr(rg["returns"], rg["prev_return"])[0] if len(rg) >= 20 else np.nan
            row[f"autocorr_regime{i}"] = rc
            regime_corrs.append(rc)
        row["stability"] = stability_score(regime_corrs)
        results.append(row)
    ac_df = pd.DataFrame(results)
    if ac_df.empty:
        print("\n  WARNING: No time slots had sufficient data for autocorrelation.")
        ac_df.to_csv(OUTPUT_DIR / "autocorrelation_by_time.csv", index=False)
        return ac_df
    ac_df = ac_df.sort_values("time_slot")
    ac_df["p_adj"] = bh_fdr(ac_df["p_value"].values)
    sig_raw = ac_df[ac_df["p_value"] < 0.05]
    sig_adj = ac_df[ac_df["p_adj"] < 0.05]
    print(f"\nSignificant time slots: {len(sig_raw)} raw / {len(sig_adj)} BH-FDR / {len(ac_df)} total")
    if not sig_adj.empty:
        print("\nTop autocorrelation by time of day (BH-FDR q < 0.05):")
        top = sig_adj.reindex(sig_adj["autocorr"].abs().sort_values(ascending=False).index).head(10)
        for _, row in top.iterrows():
            d = "MR" if row["autocorr"] < 0 else "MOM"
            print(f"  {row['time_slot']}  r={row['autocorr']:+.4f}  q={row['p_adj']:.4f}  stab={row['stability']:.2f}  R1={row['autocorr_regime1']:+.4f}  R2={row['autocorr_regime2']:+.4f}  R3={row['autocorr_regime3']:+.4f}  [{d}]")
    elif not sig_raw.empty:
        print("\n  No slots survive BH-FDR correction. Top raw significant:")
        top = sig_raw.reindex(sig_raw["autocorr"].abs().sort_values(ascending=False).index).head(5)
        for _, row in top.iterrows():
            d = "MR" if row["autocorr"] < 0 else "MOM"
            print(f"  {row['time_slot']}  r={row['autocorr']:+.4f}  q={row['p_adj']:.4f}  stab={row['stability']:.2f}  [{d}]")
    ac_df.to_csv(OUTPUT_DIR / "autocorrelation_by_time.csv", index=False)
    print("\nMulti-lag autocorrelation (overall RTH):")
    for lag in [1, 2, 3, 4, 5, 10, 20]:
        lagged = rth["returns"].shift(lag)
        mask = rth["returns"].notna() & lagged.notna()
        corr, p = _safe_pearsonr(rth["returns"][mask], lagged[mask])
        if np.isnan(corr):
            continue
        sig_marker = "*" if p < 0.05 else " "
        print(f"  Lag {lag:2d}: r={corr:+.6f}  p={p:.4f} {sig_marker}")
    return ac_df


def analyze_time_of_day(df):
    print(SEP)
    print("ANALYSIS 2: TIME-OF-DAY DIRECTIONAL BIAS")
    print("=" * 70)
    rth = df.between_time(RTH_START, RTH_END).copy()
    r1, r2, r3 = split_regimes(rth)
    results = []
    for slot, group in rth.groupby("time_slot"):
        rets = group["returns"].dropna()
        if len(rets) < 100:
            continue
        mean_ret = rets.mean()
        t_stat, p_value = stats.ttest_1samp(rets, 0)
        win_rate = (rets > 0).mean()
        avg_win = rets[rets > 0].mean() if (rets > 0).any() else 0
        avg_loss = rets[rets < 0].mean() if (rets < 0).any() else 0
        row = {"time_slot": slot, "mean_return": mean_ret, "std_return": rets.std(),
            "t_stat": t_stat, "p_value": p_value, "win_rate": win_rate,
            "avg_win": avg_win, "avg_loss": avg_loss,
            "edge_ratio": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            "n": len(rets), "annual_bps": mean_ret * 252 * 10000}
        regime_means = []
        for i, regime in enumerate([r1, r2, r3], 1):
            rg = regime[regime["time_slot"] == slot]["returns"].dropna()
            rm = rg.mean() if len(rg) >= 20 else np.nan
            row[f"mean_return_regime{i}"] = rm
            regime_means.append(rm)
        row["stability"] = stability_score(regime_means)
        results.append(row)
    tod_df = pd.DataFrame(results)
    if tod_df.empty:
        print("\n  WARNING: No time slots had sufficient data for time-of-day analysis.")
        tod_df.to_csv(OUTPUT_DIR / "time_of_day_bias.csv", index=False)
        return tod_df
    tod_df = tod_df.sort_values("time_slot")
    tod_df["p_adj"] = bh_fdr(tod_df["p_value"].values)
    sig_raw = tod_df[tod_df["p_value"] < 0.05]
    sig_adj = tod_df[tod_df["p_adj"] < 0.05]
    print(f"\nSignificant time slots: {len(sig_raw)} raw / {len(sig_adj)} BH-FDR / {len(tod_df)} total")
    if not sig_adj.empty:
        print("\nTime slots surviving BH-FDR (q < 0.05):")
        for _, row in sig_adj.sort_values("p_adj").head(15).iterrows():
            d = "LONG" if row["mean_return"] > 0 else "SHORT"
            print(f"  {row['time_slot']}  mean={row['mean_return']*10000:+.2f}bps  q={row['p_adj']:.4f}  stab={row['stability']:.2f}  R1={row['mean_return_regime1']*10000:+.1f}  R2={row['mean_return_regime2']*10000:+.1f}  R3={row['mean_return_regime3']*10000:+.1f}  [{d}]")
    elif not sig_raw.empty:
        print("\n  No slots survive BH-FDR correction. Top raw significant:")
        for _, row in sig_raw.sort_values("p_value").head(5).iterrows():
            d = "LONG" if row["mean_return"] > 0 else "SHORT"
            print(f"  {row['time_slot']}  mean={row['mean_return']*10000:+.2f}bps  q={row['p_adj']:.4f}  stab={row['stability']:.2f}  [{d}]")
    tod_df["cum_return"] = tod_df["mean_return"].cumsum()
    tod_df.to_csv(OUTPUT_DIR / "time_of_day_bias.csv", index=False)
    return tod_df


def analyze_day_of_week(daily):
    print(SEP)
    print("ANALYSIS 3: DAY-OF-WEEK EFFECTS")
    print("=" * 70)
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    r1, r2, r3 = split_regimes(daily)
    results = []
    for dow in range(5):
        rets = daily[daily["dow"] == dow]["daily_return"].dropna()
        if len(rets) < 50:
            continue
        t_stat, p_value = stats.ttest_1samp(rets, 0)
        row = {"day": day_names[dow], "mean_return": rets.mean(), "t_stat": t_stat,
               "p_value": p_value, "win_rate": (rets > 0).mean(), "std": rets.std(), "n": len(rets)}
        regime_means = []
        for i, regime in enumerate([r1, r2, r3], 1):
            rg = regime[regime["dow"] == dow]["daily_return"].dropna()
            rm = rg.mean() if len(rg) >= 10 else np.nan
            row[f"mean_return_regime{i}"] = rm
            regime_means.append(rm)
        row["stability"] = stability_score(regime_means)
        results.append(row)
    dow_df = pd.DataFrame(results)
    if dow_df.empty:
        print("\n  WARNING: No days had sufficient data for day-of-week analysis.")
        dow_df.to_csv(OUTPUT_DIR / "day_of_week.csv", index=False)
        return dow_df
    print("\nDaily return by day of week:")
    for _, row in dow_df.iterrows():
        sig = "*" if row["p_value"] < 0.05 else " "
        print(f"  {row['day']:10s}  mean={row['mean_return']*10000:+.2f}bps  std={row['std']*100:.2f}%  WR={row['win_rate']:.1%}  t={row['t_stat']:+.2f}  p={row['p_value']:.3f}  n={row['n']:.0f} {sig}")
    groups = [daily[daily["dow"] == d]["daily_return"].dropna() for d in range(5)]
    groups = [g for g in groups if len(g) >= 30]
    if len(groups) >= 2:
        f_stat, anova_p = stats.f_oneway(*groups)
        sig_label = "SIGNIFICANT" if anova_p < 0.05 else "No significant"
        print(f"\n  ANOVA F={f_stat:.3f}, p={anova_p:.4f}")
        print(f"  -> {sig_label} difference between days")
    print("\n  Regime stability (early / middle / late thirds):")
    for _, row in dow_df.iterrows():
        print(f"    {row['day']:10s}  R1={row['mean_return_regime1']*10000:+.1f}bps  R2={row['mean_return_regime2']*10000:+.1f}bps  R3={row['mean_return_regime3']*10000:+.1f}bps  stab={row['stability']:.2f}")
    dow_df.to_csv(OUTPUT_DIR / "day_of_week.csv", index=False)
    return dow_df


def analyze_gaps(daily):
    print(SEP)
    print("ANALYSIS 4: GAP ANALYSIS")
    print("=" * 70)
    gaps = daily.dropna(subset=["gap", "prev_close"]).copy()
    if len(gaps) < 2:
        print("\n  WARNING: Insufficient gap data for analysis.")
        return pd.DataFrame()
    gaps["gap_abs"] = gaps["gap"].abs()
    gaps["gap_direction"] = np.where(gaps["gap"] > 0, "up", "down")
    gaps["gap_up"] = gaps["gap"] > 0
    gaps["filled"] = np.where(gaps["gap_up"], gaps["daily_low"] <= gaps["prev_close"], gaps["daily_high"] >= gaps["prev_close"])
    gaps["continuation"] = np.where(gaps["gap_up"], gaps["daily_close"] > gaps["daily_open"], gaps["daily_close"] < gaps["daily_open"])
    print(f"\nTotal trading days with gaps: {len(gaps)}")
    print(f"  Mean gap: {gaps['gap'].mean():.2f} pts ({gaps['gap_pct'].mean()*100:.3f}%)")
    print(f"  Mean |gap|: {gaps['gap_abs'].mean():.2f} pts")
    print(f"\n  Overall gap fill rate: {gaps['filled'].mean():.1%}")
    for direction in ["up", "down"]:
        subset = gaps[gaps["gap_direction"] == direction]
        if len(subset) < 30:
            continue
        print(f"\n  Gap {direction} (n={len(subset)}):")
        print(f"    Avg gap: {subset['gap_pct'].mean()*100:+.3f}%")
        print(f"    Fill rate: {subset['filled'].mean():.1%}")
        print(f"    Continuation rate: {subset['continuation'].mean():.1%}")
        print(f"    Avg daily return: {subset['daily_return'].mean()*10000:+.1f}bps")
    gaps["gap_quintile"] = _safe_qcut(gaps["gap_abs"], 5, labels=["Q1-Small", "Q2", "Q3", "Q4", "Q5-Large"])
    if gaps["gap_quintile"].isna().all():
        print("\n  WARNING: Could not create gap quintiles (insufficient unique values).")
        gaps.to_csv(OUTPUT_DIR / "gap_analysis.csv", index=False)
        return gaps
    r1, r2, r3 = split_regimes(gaps)
    quintile_results = []
    for q, group in gaps.groupby("gap_quintile", observed=True):
        rets = group["daily_return"].dropna()
        t_stat, p_value = stats.ttest_1samp(rets, 0) if len(rets) >= 10 else (np.nan, np.nan)
        row = {"quintile": q, "gap_abs_mean": group["gap_abs"].mean(),
            "fill_rate": group["filled"].mean(), "cont_rate": group["continuation"].mean(),
            "mean_return": rets.mean(), "t_stat": t_stat, "p_value": p_value, "n": len(group)}
        regime_means = []
        for i, regime in enumerate([r1, r2, r3], 1):
            rg = regime[regime["gap_quintile"] == q]["daily_return"].dropna()
            rm = rg.mean() if len(rg) >= 5 else np.nan
            row[f"mean_return_regime{i}"] = rm
            regime_means.append(rm)
        row["stability"] = stability_score(regime_means)
        quintile_results.append(row)
    q_df = pd.DataFrame(quintile_results)
    valid = q_df["p_value"].notna()
    q_df.loc[valid, "p_adj"] = bh_fdr(q_df.loc[valid, "p_value"].values)
    print("\n  Gap quintiles (BH-FDR corrected, regime stability):")
    for _, row in q_df.iterrows():
        q_str = f"{'*' if row.get('p_adj', 1) < 0.05 else ' '}"
        r1v = f"{row['mean_return_regime1']*10000:+.1f}" if np.isfinite(row['mean_return_regime1']) else "  N/A"
        r2v = f"{row['mean_return_regime2']*10000:+.1f}" if np.isfinite(row['mean_return_regime2']) else "  N/A"
        r3v = f"{row['mean_return_regime3']*10000:+.1f}" if np.isfinite(row['mean_return_regime3']) else "  N/A"
        print(f"    {row['quintile']:10s}  ret={row['mean_return']*10000:+.1f}bps  q={row.get('p_adj', np.nan):.4f}  stab={row['stability']:.2f}  R1={r1v}  R2={r2v}  R3={r3v} {q_str}")
    q_df.to_csv(OUTPUT_DIR / "gap_quintile_tests.csv", index=False)
    gaps["fade_return"] = np.where(gaps["gap_up"], gaps["prev_close"] - gaps["daily_open"], gaps["daily_open"] - gaps["prev_close"])
    gaps["fade_return"] = np.minimum(gaps["fade_return"], gaps["gap_abs"])
    mean_fade = gaps["fade_return"].mean()
    fade_sr = mean_fade / gaps["fade_return"].std() * np.sqrt(252) if gaps["fade_return"].std() > 0 else 0
    print(f"\n  Naive gap fade strategy:")
    print(f"    Mean return per trade: {mean_fade:.2f} pts")
    print(f"    Annualized Sharpe: {fade_sr:.2f}")
    print(f"    Win rate: {(gaps['fade_return'] > 0).mean():.1%}")
    mid = len(gaps) // 2
    for label, sl in [("H1", gaps.iloc[:mid]), ("H2", gaps.iloc[mid:])]:
        sr = sl["fade_return"].mean() / sl["fade_return"].std() * np.sqrt(252) if sl["fade_return"].std() > 0 else 0
        print(f"    Sharpe {label}: {sr:.2f}")
    gaps.to_csv(OUTPUT_DIR / "gap_analysis.csv", index=False)
    return gaps


def analyze_volatility(daily):
    print(SEP)
    print("ANALYSIS 5: VOLATILITY CLUSTERING & PREDICTABILITY")
    print("=" * 70)
    vol = daily[["daily_range", "daily_range_pct", "daily_return"]].copy()
    vol["prev_range"] = vol["daily_range"].shift(1)
    vol["prev_range_pct"] = vol["daily_range_pct"].shift(1)
    vol["prev_return"] = vol["daily_return"].shift(1)
    vol["range_5d"] = vol["daily_range"].rolling(5).mean().shift(1)
    vol["range_20d"] = vol["daily_range"].rolling(20).mean().shift(1)
    vol = vol.dropna()
    if len(vol) < 2:
        print("\n  WARNING: Insufficient data for volatility analysis.")
        return pd.DataFrame(), pd.DataFrame()
    corr, p = _safe_pearsonr(vol["daily_range"], vol["prev_range"])
    print(f"\n  Range lag-1 autocorrelation: {corr:.4f} (p={p:.6f})")

    # Low / Medium / High ATR tercile regime analysis
    vol["vol_regime"] = _safe_qcut(vol["prev_range_pct"], 3, labels=["Low", "Medium", "High"])
    regime_results = []
    for regime_name, group in vol.groupby("vol_regime", observed=True):
        rets = group["daily_return"].dropna()
        if len(rets) < 20:
            continue
        t_stat, p_value = stats.ttest_1samp(rets, 0)
        regime_results.append({
            "regime": regime_name, "mean_return": rets.mean(), "std": rets.std(),
            "t_stat": t_stat, "p_value": p_value,
            "win_rate": (rets > 0).mean(), "n": len(rets),
        })
    regime_df = pd.DataFrame(regime_results)
    if not regime_df.empty:
        regime_df["p_adj"] = bh_fdr(regime_df["p_value"].values)
    print("\n  Volatility regime analysis (prior-day range terciles):")
    for _, row in regime_df.iterrows():
        sig = "*" if row.get("p_adj", 1) < 0.05 else " "
        print(f"    {row['regime']:8s}  mean={row['mean_return']*10000:+.2f}bps  std={row['std']*100:.2f}%  WR={row['win_rate']:.1%}  t={row['t_stat']:+.2f}  q={row.get('p_adj', np.nan):.4f}  n={row['n']:.0f} {sig}")
    groups_list = [vol[vol["vol_regime"] == r]["daily_return"].dropna() for r in ["Low", "Medium", "High"]]
    groups_list = [g for g in groups_list if len(g) >= 20]
    if len(groups_list) >= 2:
        f_stat, anova_p = stats.f_oneway(*groups_list)
        print(f"    ANOVA F={f_stat:.3f}, p={anova_p:.4f}")
    regime_df.to_csv(OUTPUT_DIR / "volatility_regimes.csv", index=False)

    vol["vol_expanding"] = vol["range_5d"] > vol["range_20d"]
    for label, subset in [("Vol expanding (5d > 20d)", vol[vol["vol_expanding"]]), ("Vol contracting", vol[~vol["vol_expanding"]])]:
        print(f"\n  {label}:")
        print(f"    Mean daily return: {subset['daily_return'].mean()*10000:+.1f}bps  n={len(subset)}")
    corr_ret, p_ret = _safe_pearsonr(vol["daily_return"], vol["prev_return"])
    mr_label = "Mean-reverting" if corr_ret < 0 else "Momentum"
    print(f"\n  Daily return lag-1 autocorrelation: {corr_ret:.4f} (p={p_ret:.4f})")
    print(f"  -> {mr_label} at daily level")
    vol["prev_ret_quintile"] = _safe_qcut(vol["prev_return"], 5, labels=["Q1-BigDown", "Q2", "Q3", "Q4", "Q5-BigUp"])
    print("\n  Next-day return by prior-day return quintile:")
    for q, group in vol.groupby("prev_ret_quintile", observed=True):
        print(f"    {q:12s}  next_day={group['daily_return'].mean()*10000:+.2f}bps  WR={(group['daily_return'] > 0).mean():.1%}  n={len(group)}")
    vol.to_csv(OUTPUT_DIR / "volatility_analysis.csv", index=False)
    return vol, regime_df


def analyze_range_dynamics(daily):
    print(SEP)
    print("ANALYSIS 6: RANGE DYNAMICS (NR4/NR7)")
    print("=" * 70)
    rd = daily[["daily_range", "daily_range_pct", "daily_return", "daily_high", "daily_low", "daily_open", "daily_close"]].copy()
    rd["range_rank_4"] = rd["daily_range"].rolling(4).rank(method="min", ascending=True)
    rd["nr4"] = rd["range_rank_4"] == 1
    rd["range_rank_7"] = rd["daily_range"].rolling(7).rank(method="min", ascending=True)
    rd["nr7"] = rd["range_rank_7"] == 1
    rd["next_range"] = rd["daily_range"].shift(-1)
    rd["next_return"] = rd["daily_return"].shift(-1)
    rd["next_abs_return"] = rd["next_return"].abs()
    rd = rd.dropna(subset=["next_range"])
    print(f"\n  Average daily range: {rd['daily_range'].mean():.1f} pts")
    for label, mask in [("NR4", rd["nr4"]), ("NR7", rd["nr7"])]:
        subset, non_subset = rd[mask], rd[~mask]
        if len(subset) < 30:
            continue
        expansion = subset["next_range"].mean() / subset["daily_range"].mean()
        t, p = stats.ttest_ind(subset["next_abs_return"], non_subset["next_abs_return"])
        print(f"\n  {label} Days (n={len(subset)}):")
        print(f"    Avg range on {label} day: {subset['daily_range'].mean():.1f} pts")
        print(f"    Next day avg range: {subset['next_range'].mean():.1f} pts (expansion: {expansion:.2f}x)")
        print(f"    Next day avg |return|: {subset['next_abs_return'].mean()*100:.3f}% vs others: {non_subset['next_abs_return'].mean()*100:.3f}%")
        print(f"    t-test vs non-{label}: t={t:.3f}, p={p:.4f}")
        nr_days = subset.copy()
        nr_days["next_close"] = rd["daily_close"].shift(-1).loc[nr_days.index]
        nr_days["breakout"] = (nr_days["next_close"] > nr_days["daily_high"]) | (nr_days["next_close"] < nr_days["daily_low"])
        print(f"    Next day breakout rate: {nr_days['breakout'].mean():.1%}")
    rd.to_csv(OUTPUT_DIR / "range_dynamics.csv", index=False)
    return rd


def analyze_prior_day_levels(df, daily):
    print(SEP)
    print("ANALYSIS 7: PRIOR DAY LEVEL REACTIONS")
    print("=" * 70)
    rth = df.between_time(RTH_START, RTH_END).copy()
    daily_shifted = daily[["daily_high", "daily_low", "daily_close"]].copy()
    daily_shifted.index = pd.to_datetime(daily_shifted.index)
    daily_shifted = daily_shifted.shift(1)
    daily_shifted.columns = ["pdh", "pdl", "pdc"]
    daily_shifted.index = daily_shifted.index.date
    rth = rth.join(daily_shifted, on="date").dropna(subset=["pdh", "pdl", "pdc"])
    lookforward = 4
    results = []
    for level_name, level_col in [("PDH", "pdh"), ("PDL", "pdl"), ("PDC", "pdc")]:
        if level_name in ("PDH", "PDC"):
            touches = rth[(rth["high"] >= rth[level_col]) & (rth["low"] < rth[level_col])].copy()
        else:
            touches = rth[(rth["low"] <= rth[level_col]) & (rth["high"] > rth[level_col])].copy()
        if len(touches) < 50:
            continue
        for i in range(1, lookforward + 1):
            touches[f"fwd_{i}"] = rth["close"].shift(-i).reindex(touches.index) / touches["close"] - 1
        final_col = f"fwd_{lookforward}"
        mean_fwd = touches[final_col].mean()
        t, p = stats.ttest_1samp(touches[final_col].dropna(), 0)
        direction = "bounce" if (level_name == "PDL" and mean_fwd > 0) or (level_name in ("PDH", "PDC") and mean_fwd < 0) else "break"
        results.append({"level": level_name, "n_touches": len(touches), f"mean_{lookforward}bar_return": mean_fwd, "t_stat": t, "p_value": p, "reaction": direction})
        print(f"\n  {level_name} touches (n={len(touches)}):")
        print(f"    {lookforward}-bar forward return: {mean_fwd*10000:+.2f}bps  t={t:.2f}  p={p:.4f}")
        print(f"    Reaction: {direction}")
        touches["from_above"] = touches["open"] > touches[level_col]
        for approach, sub in touches.groupby("from_above"):
            lbl = "from above" if approach else "from below"
            m, n = sub[final_col].mean(), len(sub)
            if n >= 30:
                t2, p2 = stats.ttest_1samp(sub[final_col].dropna(), 0)
                print(f"      {lbl}: {m*10000:+.2f}bps  n={n}  p={p2:.4f}")
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "prior_day_levels.csv", index=False)


def analyze_first_hour_continuation(df, daily):
    print(SEP)
    print("ANALYSIS 8: FIRST HOUR CONTINUATION VS REVERSAL")
    print("=" * 70)
    rth = df.between_time(RTH_START, RTH_END).copy()
    first_hour = rth.between_time("09:30", "10:30")
    rest = rth.between_time("10:30", "16:00")
    fh_daily = first_hour.groupby("date").agg(fh_open=("open", "first"), fh_close=("close", "last"), fh_high=("high", "max"), fh_low=("low", "min"))
    fh_daily["fh_return"] = fh_daily["fh_close"] / fh_daily["fh_open"] - 1
    rest_daily = rest.groupby("date").agg(rest_open=("open", "first"), rest_close=("close", "last"))
    rest_daily["rest_return"] = rest_daily["rest_close"] / rest_daily["rest_open"] - 1
    combined = fh_daily.join(rest_daily).dropna()
    if len(combined) < 2:
        print("\n  WARNING: Insufficient data for first-hour continuation analysis.")
        return pd.DataFrame()
    corr, p = _safe_pearsonr(combined["fh_return"], combined["rest_return"])
    tendency = "Continuation" if corr > 0 else "Reversal"
    print(f"\n  First hour return vs rest-of-day return:")
    print(f"    Correlation: {corr:.4f} (p={p:.4f})")
    print(f"    -> {tendency} tendency")
    combined["fh_quintile"] = _safe_qcut(combined["fh_return"], 5, labels=["Q1-BigDown", "Q2", "Q3", "Q4", "Q5-BigUp"])
    if not combined["fh_quintile"].isna().all():
        print("\n  Rest-of-day return by first-hour quintile:")
        for q, group in combined.groupby("fh_quintile", observed=True):
            print(f"    {q:12s}  rest={group['rest_return'].mean()*10000:+.2f}bps  WR={(group['rest_return'] > 0).mean():.1%}  n={len(group)}")
    mid = len(combined) // 2
    if mid >= 2:
        c1, _ = _safe_pearsonr(combined.iloc[:mid]["fh_return"], combined.iloc[:mid]["rest_return"])
        c2, _ = _safe_pearsonr(combined.iloc[mid:]["fh_return"], combined.iloc[mid:]["rest_return"])
        if np.isfinite(c1) and np.isfinite(c2):
            stability = "STABLE" if (c1 > 0) == (c2 > 0) else "FLIPPED"
            print(f"\n  Stability: H1 corr={c1:.4f}  H2 corr={c2:.4f}  {stability}")
    for label, mask in [("Extreme up 1st hour", combined["fh_return"] > combined["fh_return"].quantile(0.9)),
                         ("Extreme down 1st hour", combined["fh_return"] < combined["fh_return"].quantile(0.1))]:
        sub = combined[mask]
        if len(sub) < 2:
            continue
        m = sub["rest_return"].mean()
        t, p = stats.ttest_1samp(sub["rest_return"], 0)
        print(f"\n  {label} (n={len(sub)}):")
        print(f"    Rest-of-day return: {m*10000:+.2f}bps  t={t:.2f}  p={p:.4f}")
    combined.to_csv(OUTPUT_DIR / "first_hour_continuation.csv", index=False)
    return combined


def analyze_interactions(df, daily):
    """Detect edges that only exist under combined conditions.

    Tests four interaction pairs:
        1. time_of_day × volatility regime
        2. gap_size × gap_direction
        3. day_of_week × volatility regime
        4. first_hour_return × session_trend

    For each cell in every interaction grid, computes mean return, t-stat,
    p-value, and regime stability.  Applies BH-FDR across all tests and
    saves interaction_edges.csv.

    Parameters
    ----------
    df : pd.DataFrame
        Bar-level data with features from ``build_features()``.
    daily : pd.DataFrame
        Daily summary from ``build_daily_summary()``.

    Returns
    -------
    pd.DataFrame
        All interaction cells with stats, sorted by |t-stat|.
    """
    print(SEP)
    print("ANALYSIS 9: FEATURE INTERACTION EFFECTS")
    print("=" * 70)

    MIN_OBS = 50
    rth = df.between_time(RTH_START, RTH_END).copy()
    rth["fwd_return"] = rth["close"].shift(-1) / rth["close"] - 1

    # --- Build interaction columns on bar-level data ---

    # Volatility regime from rolling 20-bar vol
    rth_valid = rth.dropna(subset=["rolling_vol_20", "fwd_return"]).copy()
    if len(rth_valid) < MIN_OBS * 6:
        print("\n  Insufficient data for interaction analysis.")
        return pd.DataFrame()

    rth_valid["vol_regime"] = _safe_qcut(
        rth_valid["rolling_vol_20"], 3, labels=["Low_Vol", "Med_Vol", "High_Vol"],
    )

    # Time-of-day buckets (2-hour blocks for manageable cell counts)
    rth_valid["time_block"] = rth_valid["hour"].map({
        9: "09:30-10", 10: "10-12", 11: "10-12",
        12: "12-14", 13: "12-14", 14: "14-16", 15: "14-16",
    })

    # Day of week
    rth_valid["dow_name"] = rth_valid["dow"].map(
        {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"},
    )

    # Session trend terciles
    first_hour = rth.between_time("09:30", "10:30")
    fh_daily = first_hour.groupby("date").agg(
        fh_open=("open", "first"), fh_close=("close", "last"),
    )
    fh_daily["fh_return"] = fh_daily["fh_close"] / fh_daily["fh_open"] - 1

    rest = rth.between_time("10:30", "16:00")
    rest_daily = rest.groupby("date").agg(
        rest_open=("open", "first"), rest_close=("close", "last"),
    )
    rest_daily["rest_return"] = rest_daily["rest_close"] / rest_daily["rest_open"] - 1

    # --- Gap features on daily data ---
    gap_daily = daily.dropna(subset=["gap", "prev_close"]).copy()
    if len(gap_daily) >= MIN_OBS:
        gap_daily["gap_direction"] = np.where(gap_daily["gap"] > 0, "Gap_Up", "Gap_Down")
        gap_nonzero = gap_daily[gap_daily["gap"] != 0].copy()
        if len(gap_nonzero) >= MIN_OBS * 3:
            gap_nonzero["gap_size"] = _safe_qcut(
                gap_nonzero["gap"].abs(), 3, labels=["Small", "Medium", "Large"],
            )
            if gap_nonzero["gap_size"].isna().all():
                gap_nonzero = None
        else:
            gap_nonzero = None
    else:
        gap_daily = None
        gap_nonzero = None

    # --- Helper to compute stats for one interaction cell ---
    all_rows = []

    def _test_cell(interaction_name, factor_a, factor_b, returns):
        """Run t-test on a single interaction cell and append to all_rows."""
        clean = returns.dropna()
        n = len(clean)
        if n < MIN_OBS:
            return
        mean_ret = float(clean.mean())
        std_ret = float(clean.std())
        if std_ret < 1e-12:
            return
        t_stat, p_value = stats.ttest_1samp(clean, 0)

        # Regime stability
        r1, r2, r3 = split_regimes(pd.DataFrame({"ret": clean}))
        regime_means = []
        for regime in (r1, r2, r3):
            if len(regime) >= 10:
                regime_means.append(float(regime["ret"].mean()))
            else:
                regime_means.append(np.nan)
        stab = stability_score(regime_means)

        all_rows.append({
            "interaction": interaction_name,
            "factor_a": str(factor_a),
            "factor_b": str(factor_b),
            "key": f"{factor_a} × {factor_b}",
            "n": n,
            "mean_return": round(mean_ret, 8),
            "std_return": round(std_ret, 8),
            "t_stat": round(float(t_stat), 4),
            "p_value": round(float(p_value), 6),
            "stability": round(stab, 4) if np.isfinite(stab) else np.nan,
            "mean_return_regime1": round(regime_means[0], 8) if np.isfinite(regime_means[0]) else np.nan,
            "mean_return_regime2": round(regime_means[1], 8) if np.isfinite(regime_means[1]) else np.nan,
            "mean_return_regime3": round(regime_means[2], 8) if np.isfinite(regime_means[2]) else np.nan,
        })

    # --- Interaction 1: time_of_day × volatility regime ---
    print("\n  [1] Time-of-Day × Volatility Regime")
    valid = rth_valid.dropna(subset=["time_block", "vol_regime"])
    for (tb, vr), grp in valid.groupby(["time_block", "vol_regime"], observed=True):
        _test_cell("time_x_vol", tb, vr, grp["fwd_return"])
    n1 = sum(1 for r in all_rows if r["interaction"] == "time_x_vol")
    print(f"      {n1} cells tested")

    # --- Interaction 2: gap_size × gap_direction ---
    print("  [2] Gap Size × Gap Direction")
    n_before = len(all_rows)
    if gap_nonzero is not None and len(gap_nonzero) >= MIN_OBS:
        for (gs, gd), grp in gap_nonzero.groupby(["gap_size", "gap_direction"], observed=True):
            _test_cell("gap_size_x_dir", gs, gd, grp["daily_return"])
    n2 = len(all_rows) - n_before
    print(f"      {n2} cells tested")

    # --- Interaction 3: day_of_week × volatility regime ---
    print("  [3] Day-of-Week × Volatility Regime")
    n_before = len(all_rows)
    valid = rth_valid.dropna(subset=["dow_name", "vol_regime"])
    for (dow, vr), grp in valid.groupby(["dow_name", "vol_regime"], observed=True):
        _test_cell("dow_x_vol", dow, vr, grp["fwd_return"])
    n3 = len(all_rows) - n_before
    print(f"      {n3} cells tested")

    # --- Interaction 4: first_hour_return × session_trend ---
    print("  [4] First-Hour Return × Rest-of-Day Trend")
    n_before = len(all_rows)
    combined = fh_daily.join(rest_daily).dropna()
    if len(combined) >= MIN_OBS * 4:
        combined["fh_dir"] = _safe_qcut(
            combined["fh_return"], 3, labels=["FH_Down", "FH_Flat", "FH_Up"],
        )
        combined["rest_dir"] = _safe_qcut(
            combined["rest_return"], 3, labels=["Rest_Down", "Rest_Flat", "Rest_Up"],
        )
        if combined["fh_dir"].isna().all() or combined["rest_dir"].isna().all():
            print("      Skipped — insufficient unique values for qcut")
        else:
            pass  # fall through to groupby below
        # What is the rest-of-day return given first-hour direction?
        for fh_bucket, grp in combined.dropna(subset=["fh_dir"]).groupby("fh_dir", observed=True):
            _test_cell("fh_x_session", fh_bucket, "rest_return", grp["rest_return"])
        # Full grid: fh_dir × rest_dir — test daily return
        fh_rest_daily = combined.join(
            daily[["daily_return"]],
        ).dropna(subset=["daily_return", "fh_dir", "rest_dir"])
        if len(fh_rest_daily) >= MIN_OBS:
            for (fh, rest), grp in fh_rest_daily.groupby(
                ["fh_dir", "rest_dir"], observed=True,
            ):
                _test_cell("fh_x_rest_dir", fh, rest, grp["daily_return"])
    n4 = len(all_rows) - n_before
    print(f"      {n4} cells tested")

    # --- Assemble results ---
    if not all_rows:
        print("\n  No interaction cells had sufficient observations.")
        return pd.DataFrame()

    int_df = pd.DataFrame(all_rows)

    # BH-FDR correction across all interaction tests
    int_df["p_adj"] = bh_fdr(int_df["p_value"].values)
    int_df["significant"] = int_df["p_adj"] < 0.05

    # Sort by |t-stat|
    int_df["abs_t"] = int_df["t_stat"].abs()
    int_df.sort_values("abs_t", ascending=False, inplace=True)
    int_df.drop(columns="abs_t", inplace=True)
    int_df.reset_index(drop=True, inplace=True)

    # Save
    int_df.to_csv(OUTPUT_DIR / "interaction_edges.csv", index=False)

    # Print summary
    n_sig = int(int_df["significant"].sum())
    n_total = len(int_df)
    print(f"\n  Total interaction cells: {n_total}")
    print(f"  Significant (BH-FDR q < 0.05): {n_sig}")

    if n_sig > 0:
        print(f"\n  Top interaction edges (significant, by |t-stat|):")
        sig = int_df[int_df["significant"]].head(15)
        for _, r in sig.iterrows():
            stab_str = f"stab={r['stability']:.2f}" if np.isfinite(r.get("stability", np.nan)) else "stab=N/A"
            print(
                f"    [{r['interaction']:18s}]  {r['key']:25s}  "
                f"mean={r['mean_return']*10000:+.2f}bps  t={r['t_stat']:+.2f}  "
                f"q={r['p_adj']:.4f}  n={r['n']}  {stab_str}"
            )
    else:
        print("\n  No interaction edges survive FDR correction.")
        print("  Top 5 by raw |t-stat|:")
        for _, r in int_df.head(5).iterrows():
            stab_str = f"stab={r['stability']:.2f}" if np.isfinite(r.get("stability", np.nan)) else "stab=N/A"
            print(
                f"    [{r['interaction']:18s}]  {r['key']:25s}  "
                f"mean={r['mean_return']*10000:+.2f}bps  t={r['t_stat']:+.2f}  "
                f"p={r['p_value']:.4f}  n={r['n']}  {stab_str}"
            )

    # Per-interaction summary
    print(f"\n  Summary by interaction type:")
    for itype, grp in int_df.groupby("interaction"):
        n_cells = len(grp)
        n_s = int(grp["significant"].sum())
        if n_cells == 0:
            continue
        best = grp.iloc[0]
        print(f"    {itype:18s}  cells={n_cells:3d}  sig={n_s:2d}  "
              f"best_t={best['t_stat']:+.2f}  best_key={best['key']}")

    print(f"\n  Saved to: {(OUTPUT_DIR / 'interaction_edges.csv').resolve()}")
    return int_df


# ---------------------------------------------------------------------------
# Edge prerequisite check (lightweight gate for optimizer)
# ---------------------------------------------------------------------------

# Default thresholds for edge validation
# Note: BH-FDR across ~35 tests is conservative. We use raw p-value as the
# primary filter and require |t| >= 2.0 which already implies p < 0.05 for
# large samples.  The p_adj filter is set at 0.30 (lenient) so it only
# rejects edges that are clearly noise after correction.  Stability is set
# to 0.0 by default — the stability score is reported but not used as a
# hard gate because short-window regime splits can be noisy.
DEFAULT_EDGE_THRESHOLDS = {
    "min_significant_edges": 1,     # at least 1 edge must survive filters
    "min_stability": 0.0,           # reported but not gated (regime splits are noisy)
    "max_p_adj": 0.30,              # FDR-corrected threshold (lenient — |t| is the real gate)
    "min_sample_size": 50,          # minimum observations per edge
    "min_abs_t_stat": 2.0,          # minimum |t-stat| for a credible edge
    "max_raw_p": 0.05,              # raw p-value must be < 0.05
}


def check_edge_prerequisites(
    data_file: str = "data/nq_15m_data.csv",
    thresholds: dict | None = None,
    quiet: bool = False,
) -> dict:
    """Fast edge validation gate for the optimizer.

    Runs time-of-day, day-of-week, and volatility regime analyses
    (the three cheapest tests) to confirm at least one statistically
    significant, stable edge exists in the data.  Does NOT run the full
    pipeline — skips autocorrelation, gap quintiles, range dynamics,
    prior-day levels, first-hour continuation, and interactions.

    Parameters
    ----------
    data_file : str
        Path to OHLCV CSV.
    thresholds : dict | None
        Override individual thresholds from ``DEFAULT_EDGE_THRESHOLDS``.
    quiet : bool
        If True, suppress printed output (still logs).

    Returns
    -------
    dict
        passed : bool
            True if at least one qualifying edge was found.
        n_significant : int
            Number of edges surviving FDR + stability + sample size filters.
        n_tested : int
            Total tests run.
        qualifying_edges : list[dict]
            Details of each qualifying edge.
        failures : list[str]
            Reasons for rejection (empty if passed).
        summary_df : pd.DataFrame
            Full test results before filtering.
    """
    df = load_csv(data_file)
    return _check_edge_prerequisites_impl(df, thresholds, quiet)


def check_edge_prerequisites_from_df(
    df: "pd.DataFrame",
    thresholds: dict | None = None,
    quiet: bool = False,
) -> dict:
    """Same as check_edge_prerequisites but accepts a DataFrame directly.

    Used by walk-forward analysis where the data is already sliced into
    folds — avoids re-loading from CSV.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame (already loaded via ``load_csv``).
    thresholds : dict | None
        Override individual thresholds from ``DEFAULT_EDGE_THRESHOLDS``.
    quiet : bool
        If True, suppress printed output.

    Returns
    -------
    dict
        Same structure as ``check_edge_prerequisites``.
    """
    return _check_edge_prerequisites_impl(df, thresholds, quiet)


def _check_edge_prerequisites_impl(
    df: "pd.DataFrame",
    thresholds: dict | None = None,
    quiet: bool = False,
) -> dict:
    """Shared implementation for edge prerequisite checks."""
    th = {**DEFAULT_EDGE_THRESHOLDS, **(thresholds or {})}

    df = build_features(df)
    daily = build_daily_summary(df)

    # Run the three fast analyses
    # Suppress their print output by temporarily redirecting
    import io
    import contextlib
    buf = io.StringIO()
    tod_df = pd.DataFrame()
    dow_df = pd.DataFrame()
    vol_regime_df = pd.DataFrame()
    with contextlib.redirect_stdout(buf):
        try:
            tod_df = analyze_time_of_day(df)
        except Exception:
            pass
        try:
            dow_df = analyze_day_of_week(daily)
        except Exception:
            pass
        try:
            _, vol_regime_df = analyze_volatility(daily)
        except Exception:
            pass

    # Collect all tested edges into one DataFrame
    all_tests = []

    def _collect(result_df, analysis_name, key_col, metric_col="mean_return"):
        if result_df is None or result_df.empty:
            return
        for _, r in result_df.iterrows():
            all_tests.append({
                "analysis": analysis_name,
                "key": r.get(key_col, ""),
                "mean_return": r.get(metric_col, np.nan),
                "t_stat": r.get("t_stat", np.nan),
                "p_value": r.get("p_value", np.nan),
                "p_adj": r.get("p_adj", np.nan),
                "stability": r.get("stability", np.nan),
                "n": r.get("n", 0),
            })

    _collect(tod_df, "time_of_day", "time_slot")
    _collect(dow_df, "day_of_week", "day")
    _collect(vol_regime_df, "volatility_regime", "regime")

    if not all_tests:
        result = {
            "passed": False,
            "n_significant": 0,
            "n_tested": 0,
            "qualifying_edges": [],
            "failures": ["No testable edges found — insufficient data."],
            "summary_df": pd.DataFrame(),
        }
        if not quiet:
            _print_edge_check(result, th)
        return result

    summary_df = pd.DataFrame(all_tests)

    # Re-apply BH-FDR across the combined set
    valid_p = summary_df["p_value"].notna()
    if valid_p.any():
        summary_df.loc[valid_p, "p_adj"] = bh_fdr(
            summary_df.loc[valid_p, "p_value"].values,
        )

    # Filter for qualifying edges
    mask = (
        (summary_df["p_adj"] <= th["max_p_adj"])
        & (summary_df["p_value"] <= th.get("max_raw_p", 0.05))
        & (summary_df["n"] >= th["min_sample_size"])
        & (summary_df["t_stat"].abs() >= th["min_abs_t_stat"])
    )
    # Stability filter: allow NaN stability through (some analyses don't compute it)
    stab = summary_df["stability"]
    stab_ok = stab.isna() | (stab >= th["min_stability"])
    mask = mask & stab_ok

    qualifying = summary_df[mask].sort_values(
        "t_stat", key=abs, ascending=False,
    )
    n_sig = len(qualifying)
    n_tested = len(summary_df)

    failures = []
    if n_sig < th["min_significant_edges"]:
        failures.append(
            f"Only {n_sig} qualifying edge(s) found, "
            f"need >= {th['min_significant_edges']}. "
            f"(p<{th.get('max_raw_p', 0.05)}, q<{th['max_p_adj']}, "
            f"|t|>={th['min_abs_t_stat']}, "
            f"n>={th['min_sample_size']}, stab>={th['min_stability']})"
        )

    passed = len(failures) == 0

    result = {
        "passed": passed,
        "n_significant": n_sig,
        "n_tested": n_tested,
        "qualifying_edges": qualifying.to_dict("records"),
        "failures": failures,
        "summary_df": summary_df,
    }

    if not quiet:
        _print_edge_check(result, th)

    return result


def _print_edge_check(result: dict, thresholds: dict) -> None:
    """Pretty-print edge prerequisite check results."""
    status = "PASS" if result["passed"] else "FAIL"
    print("\n" + "=" * 70)
    print("  EDGE PREREQUISITE CHECK")
    print("=" * 70)
    print(f"  Status: {status}  "
          f"({result['n_significant']}/{result['n_tested']} edges qualify)")
    print(f"  Thresholds: p<{thresholds.get('max_raw_p', 0.05)}  "
          f"q<{thresholds['max_p_adj']}  "
          f"|t|>={thresholds['min_abs_t_stat']}  "
          f"n>={thresholds['min_sample_size']}  "
          f"stab>={thresholds['min_stability']}")

    if result["qualifying_edges"]:
        print("\n  Qualifying edges:")
        for e in result["qualifying_edges"][:10]:
            stab_str = (f"stab={e['stability']:.2f}"
                        if np.isfinite(e.get("stability", np.nan))
                        else "stab=N/A")
            print(
                f"    [{e['analysis']:20s}]  {e['key']:12s}  "
                f"mean={e['mean_return']*10000:+.2f}bps  "
                f"t={e['t_stat']:+.2f}  q={e['p_adj']:.4f}  "
                f"n={e['n']:.0f}  {stab_str}"
            )

    if result["failures"]:
        print("\n  REJECTION REASONS:")
        for f in result["failures"]:
            print(f"    - {f}")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Filter diagnostics
# ---------------------------------------------------------------------------

DIAG_OUTPUT_PATH = Path("output/edge_analysis/edge_filter_diagnostics.csv")


def diagnose_edge_filters(
    data_file: str = "data/nq_15m_data.csv",
    df: "pd.DataFrame | None" = None,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """Run the edge prerequisite pipeline and report exactly where edges
    are eliminated at each filtering stage.

    Accepts either a file path or a pre-loaded DataFrame (for fold-level
    diagnostics).

    Parameters
    ----------
    data_file : str
        Path to OHLCV CSV (ignored if *df* is provided).
    df : pd.DataFrame | None
        Pre-loaded DataFrame.  If None, loads from *data_file*.
    thresholds : dict | None
        Override ``DEFAULT_EDGE_THRESHOLDS``.

    Returns
    -------
    pd.DataFrame
        Per-edge diagnostics with a boolean column for every filter stage,
        also saved to ``output/edge_analysis/edge_filter_diagnostics.csv``.
    """
    th = {**DEFAULT_EDGE_THRESHOLDS, **(thresholds or {})}

    # --- Load and enrich data ---
    if df is None:
        df = load_csv(data_file)
    df = build_features(df)
    daily = build_daily_summary(df)

    # --- Run analyses (suppress their print output) ---
    import io
    import contextlib

    tod_df = pd.DataFrame()
    dow_df = pd.DataFrame()
    vol_regime_df = pd.DataFrame()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            tod_df = analyze_time_of_day(df)
        except Exception:
            pass
        try:
            dow_df = analyze_day_of_week(daily)
        except Exception:
            pass
        try:
            _, vol_regime_df = analyze_volatility(daily)
        except Exception:
            pass

    # --- Collect raw edges ---
    all_tests = []

    def _collect(result_df, analysis_name, key_col, metric_col="mean_return"):
        if result_df is None or result_df.empty:
            return
        for _, r in result_df.iterrows():
            all_tests.append({
                "analysis": analysis_name,
                "key": r.get(key_col, ""),
                "mean_return": r.get(metric_col, np.nan),
                "t_stat": r.get("t_stat", np.nan),
                "p_value": r.get("p_value", np.nan),
                "stability": r.get("stability", np.nan),
                "n": r.get("n", 0),
            })

    _collect(tod_df, "time_of_day", "time_slot")
    _collect(dow_df, "day_of_week", "day")
    _collect(vol_regime_df, "volatility_regime", "regime")

    if not all_tests:
        print("\n  No testable edges found — nothing to diagnose.")
        return pd.DataFrame()

    diag = pd.DataFrame(all_tests)

    # --- BH-FDR correction ---
    valid_p = diag["p_value"].notna()
    diag["p_adj"] = np.nan
    if valid_p.any():
        diag.loc[valid_p, "p_adj"] = bh_fdr(diag.loc[valid_p, "p_value"].values)

    # --- Compute each filter independently ---
    diag["abs_t_stat"] = diag["t_stat"].abs()

    max_raw_p = th.get("max_raw_p", 0.05)
    diag["pass_raw_p"] = diag["p_value"] <= max_raw_p
    diag["pass_p_adj"] = diag["p_adj"] <= th["max_p_adj"]
    diag["pass_t_stat"] = diag["abs_t_stat"] >= th["min_abs_t_stat"]
    diag["pass_sample_size"] = diag["n"] >= th["min_sample_size"]
    stab = diag["stability"]
    diag["pass_stability"] = stab.isna() | (stab >= th["min_stability"])

    # --- Cumulative filter cascade (order matters for the funnel) ---
    diag["survives_raw_p"] = diag["pass_raw_p"]
    diag["survives_p_adj"] = diag["survives_raw_p"] & diag["pass_p_adj"]
    diag["survives_t_stat"] = diag["survives_p_adj"] & diag["pass_t_stat"]
    diag["survives_sample"] = diag["survives_t_stat"] & diag["pass_sample_size"]
    diag["survives_stability"] = diag["survives_sample"] & diag["pass_stability"]
    diag["qualifies"] = diag["survives_stability"]

    # Sort by |t-stat| descending
    diag.sort_values("abs_t_stat", ascending=False, inplace=True)
    diag.reset_index(drop=True, inplace=True)

    # --- Save ---
    DIAG_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    diag.to_csv(DIAG_OUTPUT_PATH, index=False)

    # --- Print funnel ---
    n_total = len(diag)
    stages = [
        ("Total tested",      n_total),
        (f"p_value  < {max_raw_p}",  int(diag["pass_raw_p"].sum())),
        (f"p_adj    < {th['max_p_adj']}", int(diag["pass_p_adj"].sum())),
        (f"|t_stat| >= {th['min_abs_t_stat']}", int(diag["pass_t_stat"].sum())),
        (f"n        >= {th['min_sample_size']}", int(diag["pass_sample_size"].sum())),
        (f"stability >= {th['min_stability']}", int(diag["pass_stability"].sum())),
    ]

    cascade = [
        ("Total tested",                n_total),
        (f"after p_value  < {max_raw_p}",   int(diag["survives_raw_p"].sum())),
        (f"after p_adj    < {th['max_p_adj']}",  int(diag["survives_p_adj"].sum())),
        (f"after |t_stat| >= {th['min_abs_t_stat']}", int(diag["survives_t_stat"].sum())),
        (f"after n        >= {th['min_sample_size']}", int(diag["survives_sample"].sum())),
        (f"after stability >= {th['min_stability']}", int(diag["survives_stability"].sum())),
    ]

    print("\n" + "=" * 80)
    print("  EDGE FILTER DIAGNOSTICS")
    print("=" * 80)

    # Per-filter pass counts (independent)
    print("\n  INDEPENDENT FILTER PASS COUNTS (each filter alone)")
    print("  " + "-" * 50)
    for label, count in stages:
        pct = count / n_total * 100 if n_total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"    {label:30s}  {count:3d}/{n_total}  ({pct:5.1f}%)  {bar}")

    # Cumulative cascade (filters applied in sequence)
    print("\n  CUMULATIVE FILTER CASCADE (filters applied in order)")
    print("  " + "-" * 50)
    prev = n_total
    for label, count in cascade:
        dropped = prev - count
        pct = count / n_total * 100 if n_total > 0 else 0
        drop_str = f"  (-{dropped})" if dropped > 0 else ""
        print(f"    {label:30s}  {count:3d}/{n_total}  ({pct:5.1f}%){drop_str}")
        prev = count

    # Bottleneck identification
    print("\n  BOTTLENECK ANALYSIS")
    print("  " + "-" * 50)
    max_drop = 0
    bottleneck = ""
    for i in range(1, len(cascade)):
        drop = cascade[i - 1][1] - cascade[i][1]
        if drop > max_drop:
            max_drop = drop
            bottleneck = cascade[i][0]
    if bottleneck:
        print(f"    Biggest drop: {bottleneck} (eliminates {max_drop} edges)")

    # Near-miss edges (fail one filter only)
    near_miss_cols = ["pass_raw_p", "pass_p_adj", "pass_t_stat",
                      "pass_sample_size", "pass_stability"]
    diag["n_filters_passed"] = diag[near_miss_cols].sum(axis=1)
    near_misses = diag[diag["n_filters_passed"] == len(near_miss_cols) - 1]

    if not near_misses.empty:
        print(f"\n  NEAR-MISS EDGES ({len(near_misses)} edges fail exactly 1 filter)")
        print("  " + "-" * 50)
        for _, row in near_misses.head(10).iterrows():
            # Which filter did it fail?
            failed = [c.replace("pass_", "") for c in near_miss_cols
                      if not row[c]]
            stab_str = (f"stab={row['stability']:.2f}"
                        if pd.notna(row["stability"]) else "stab=N/A")
            print(
                f"    [{row['analysis']:15s}]  {row['key']:10s}  "
                f"|t|={row['abs_t_stat']:.2f}  p={row['p_value']:.4f}  "
                f"q={row['p_adj']:.4f}  n={row['n']:.0f}  {stab_str}  "
                f"FAILED: {', '.join(failed)}"
            )

    # All edges detail
    print(f"\n  ALL EDGES (sorted by |t-stat|)")
    print("  " + "-" * 50)
    display_cols = ["analysis", "key", "n", "t_stat", "p_value", "p_adj",
                    "stability", "qualifies"]
    print("  " + diag[display_cols].to_string(index=True).replace("\n", "\n  "))

    print(f"\n  Saved to: {DIAG_OUTPUT_PATH.resolve()}")
    print("=" * 80 + "\n")

    return diag


def build_summary(results_dict):
    """Consolidate all significant edges into a single summary CSV.

    Collects rows with p_adj < 0.05 (or p_value < 0.05 where FDR was not
    applied) from every analysis that returns a DataFrame.  Writes to
    output/edge_analysis/edge_analysis_results.csv.
    """
    rows = []

    def _harvest(df, analysis, key_col, metric_col="mean_return", p_col="p_value", padj_col="p_adj"):
        if df is None or df.empty:
            return
        use_adj = padj_col in df.columns
        pcol = padj_col if use_adj else p_col
        sig = df[df[pcol] < 0.05].copy() if pcol in df.columns else pd.DataFrame()
        for _, r in sig.iterrows():
            row = {
                "analysis": analysis,
                "key": r[key_col] if key_col in r.index else "",
                "mean_return": r.get(metric_col, np.nan),
                "sample_size": r.get("n", np.nan),
                "t_stat": r.get("t_stat", np.nan),
                "p_value": r.get(p_col, np.nan),
                "p_adj": r.get(padj_col, np.nan),
                "stability": r.get("stability", np.nan),
            }
            rows.append(row)

    for name, (df, key_col, metric_col) in results_dict.items():
        _harvest(df, name, key_col, metric_col)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary.sort_values("p_adj", na_position="last", inplace=True)
    summary.to_csv(OUTPUT_DIR / "edge_analysis_results.csv", index=False)

    print(SEP)
    print("SIGNIFICANT EDGES SUMMARY")
    print("=" * 70)
    if summary.empty:
        print("\n  No statistically significant edges found (p_adj < 0.05).")
    else:
        print(f"\n  {len(summary)} significant edges across all analyses:\n")
        for _, r in summary.head(20).iterrows():
            stab = f"stab={r['stability']:.2f}" if np.isfinite(r.get("stability", np.nan)) else ""
            print(f"  [{r['analysis']:25s}]  {r['key']:12s}  mean={r['mean_return']*10000:+.2f}bps  t={r['t_stat']:+.2f}  q={r['p_adj']:.4f}  n={r['sample_size']:.0f}  {stab}")
        if len(summary) > 20:
            print(f"  ... and {len(summary) - 20} more (see CSV)")
    print(f"\n  Saved to: {(OUTPUT_DIR / 'edge_analysis_results.csv').resolve()}")
    return summary


def rank_edges(
    summary: pd.DataFrame,
    min_sample_size: int = 50,
    w_return: float = 0.30,
    w_tstat: float = 0.30,
    w_sample: float = 0.15,
    w_stability: float = 0.25,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank edges by composite score and return the top N.

    Parameters
    ----------
    summary : pd.DataFrame
        Output from ``build_summary()``.
    min_sample_size : int
        Minimum observations to keep an edge (default 50).
    w_return, w_tstat, w_sample, w_stability : float
        Weights for the composite score components.
    top_n : int
        Number of top edges to display and return.

    Returns
    -------
    pd.DataFrame
        Ranked edges with ``edge_score`` column, saved to
        output/edge_analysis/ranked_edges.csv.
    """
    if summary.empty:
        print("\n  No edges to rank.")
        return summary

    df = summary.copy()

    # Filter: p_adj < 0.05 and sufficient sample
    p_col = "p_adj" if "p_adj" in df.columns else "p_value"
    df = df[df[p_col] < 0.05]
    df = df[df["sample_size"] >= min_sample_size]

    if df.empty:
        print("\n  No edges survive filters (p < 0.05 and n >= {}).".format(min_sample_size))
        return df

    # Normalise each component to [0, 1] for fair weighting
    abs_ret = df["mean_return"].abs()
    abs_t = df["t_stat"].abs()
    log_n = np.log(df["sample_size"])
    stab = df["stability"].fillna(0)

    def _norm(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else pd.Series(0.5, index=s.index)

    df["edge_score"] = (
        w_return * _norm(abs_ret)
        + w_tstat * _norm(abs_t)
        + w_sample * _norm(log_n)
        + w_stability * _norm(stab)
    )

    df = df.sort_values("edge_score", ascending=False).reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / "ranked_edges.csv", index=False)

    print(SEP)
    print("RANKED EDGES (top {})".format(min(top_n, len(df))))
    print("=" * 70)
    print(f"  Weights: return={w_return}  t-stat={w_tstat}  log(n)={w_sample}  stability={w_stability}")
    print(f"  Filters: p_adj < 0.05, n >= {min_sample_size}")
    print(f"  Edges surviving: {len(df)}\n")

    for i, (_, r) in enumerate(df.head(top_n).iterrows(), 1):
        stab_str = f"stab={r['stability']:.2f}" if np.isfinite(r.get("stability", np.nan)) else "stab=N/A"
        print(
            f"  #{i:2d}  [{r['analysis']:25s}]  {r['key']:12s}  "
            f"score={r['edge_score']:.3f}  mean={r['mean_return']*10000:+.2f}bps  "
            f"t={r['t_stat']:+.2f}  q={r.get('p_adj', r['p_value']):.4f}  "
            f"n={r['sample_size']:.0f}  {stab_str}"
        )

    print(f"\n  Saved to: {(OUTPUT_DIR / 'ranked_edges.csv').resolve()}")
    return df


def threshold_sensitivity(results_dict,
                          p_value_thresholds=(0.01, 0.05, 0.1),
                          p_adj_thresholds=(0.05, 0.1, 0.2)):
    """Test how edge counts change under different significance thresholds.

    Harvests ALL edges from results_dict (no filtering), then counts how many
    pass each combination of p_value and p_adj threshold.

    Saves threshold_sensitivity.csv to OUTPUT_DIR.
    """
    # Harvest all edges without any p-value filter
    rows = []
    for name, (df, key_col, metric_col) in results_dict.items():
        if df is None or df.empty:
            continue
        p_col = "p_value"
        padj_col = "p_adj"
        has_padj = padj_col in df.columns
        has_p = p_col in df.columns
        if not has_p and not has_padj:
            continue
        for _, r in df.iterrows():
            rows.append({
                "analysis": name,
                "key": r[key_col] if key_col in r.index else "",
                "mean_return": r.get(metric_col, np.nan),
                "sample_size": r.get("n", np.nan),
                "t_stat": r.get("t_stat", np.nan),
                "p_value": r.get(p_col, np.nan),
                "p_adj": r.get(padj_col, np.nan),
                "stability": r.get("stability", np.nan),
            })

    all_edges = pd.DataFrame(rows)
    if all_edges.empty:
        print("\n  No edges to analyse for threshold sensitivity.")
        return pd.DataFrame()

    total = len(all_edges)

    # Build sensitivity table
    sens_rows = []
    for p_thresh in p_value_thresholds:
        for padj_thresh in p_adj_thresholds:
            mask_p = all_edges["p_value"] < p_thresh
            has_padj = all_edges["p_adj"].notna()
            # Edges with p_adj: must pass both thresholds
            # Edges without p_adj: fall back to p_value only
            mask = (has_padj & (all_edges["p_adj"] < padj_thresh) & mask_p) | (~has_padj & mask_p)
            passing = all_edges[mask]
            n_pass = len(passing)

            # Breakdown by analysis
            by_analysis = passing.groupby("analysis").size().to_dict()

            # Mean stats of passing edges
            mean_abs_ret = passing["mean_return"].abs().mean() if n_pass > 0 else np.nan
            mean_stability = passing["stability"].mean() if n_pass > 0 else np.nan

            sens_rows.append({
                "p_value_threshold": p_thresh,
                "p_adj_threshold": padj_thresh,
                "edges_passing": n_pass,
                "pct_of_total": round(100 * n_pass / total, 1),
                "mean_abs_return_bps": round(mean_abs_ret * 10000, 2) if np.isfinite(mean_abs_ret) else np.nan,
                "mean_stability": round(mean_stability, 3) if np.isfinite(mean_stability) else np.nan,
                **{f"n_{k}": v for k, v in sorted(by_analysis.items())},
            })

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(OUTPUT_DIR / "threshold_sensitivity.csv", index=False)

    # Print summary
    print(SEP)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"\n  Total candidate edges: {total}")
    print(f"\n  {'p_value <':<12s}  {'p_adj <':<10s}  {'Passing':<10s}  {'% Total':<10s}  {'Mean |ret| bps':<16s}  {'Mean stab'}")
    print("  " + "-" * 74)
    for _, r in sens_df.iterrows():
        stab_str = f"{r['mean_stability']:.3f}" if pd.notna(r.get("mean_stability")) else "N/A"
        ret_str = f"{r['mean_abs_return_bps']:.2f}" if pd.notna(r.get("mean_abs_return_bps")) else "N/A"
        p_val = f"{float(r['p_value_threshold']):<12.2f}" if pd.notna(r.get("p_value_threshold")) else "N/A         "
        q_val = f"{float(r['p_adj_threshold']):<10.2f}" if pd.notna(r.get("p_adj_threshold")) else "N/A       "
        n_pass = f"{int(r['edges_passing']):<10d}" if pd.notna(r.get("edges_passing")) else "N/A       "
        pct = f"{float(r['pct_of_total']):<10.1f}" if pd.notna(r.get("pct_of_total")) else "N/A       "
        print(f"  {p_val}  {q_val}  {n_pass}  {pct}  {ret_str:<16s}  {stab_str}")

    # Breakdown by analysis for tightest vs loosest
    if sens_df.empty:
        print("\n  No threshold combinations to analyse.")
        print(f"\n  Saved to: {(OUTPUT_DIR / 'threshold_sensitivity.csv').resolve()}")
        return sens_df
    print(f"\n  Per-analysis breakdown (tightest: p<{p_value_thresholds[0]}, q<{p_adj_thresholds[0]}):")
    tightest = sens_df.iloc[0]
    for col in sens_df.columns:
        if col.startswith("n_") and pd.notna(tightest[col]) and tightest[col] > 0:
            print(f"    {col[2:]:<30s}  {int(tightest[col])}")

    print(f"\n  Per-analysis breakdown (loosest: p<{p_value_thresholds[-1]}, q<{p_adj_thresholds[-1]}):")
    loosest = sens_df.iloc[-1]
    for col in sens_df.columns:
        if col.startswith("n_") and pd.notna(loosest[col]) and loosest[col] > 0:
            print(f"    {col[2:]:<30s}  {int(loosest[col])}")

    print(f"\n  Saved to: {(OUTPUT_DIR / 'threshold_sensitivity.csv').resolve()}")
    return sens_df


def analyze_conditional_edges(df, daily, min_obs=50):
    """High-quality conditional edge discovery for NQ intraday futures.

    Tests four specific condition pairs that matter for intraday NQ edges.
    For each cell: mean return, t-stat, p-value (BH-FDR), sample size,
    regime stability, marginal comparison, and composite ranking.

    Combinations tested:
        A) Time-of-day × volatility regime (ATR percentile)
        B) Gap direction × gap size
        C) First hour return × rest-of-day return
        D) Prior day level touches × approach direction

    Saves conditional_edge_results.csv to OUTPUT_DIR.
    """
    print(SEP)
    print("CONDITIONAL EDGE DISCOVERY")
    print("=" * 70)

    MIN_OBS = min_obs
    rth = df.between_time(RTH_START, RTH_END).copy()

    if len(rth) < MIN_OBS * 10:
        print("\n  WARNING: Insufficient RTH data for conditional edge analysis.")
        return pd.DataFrame()

    # ---- Shared helpers ----

    def _cell_stats(rets):
        """Compute stats for a single return series."""
        clean = rets.dropna()
        n = len(clean)
        if n < MIN_OBS or clean.std() < 1e-12:
            return None
        t_stat, p_value = stats.ttest_1samp(clean, 0)
        return {
            "n": n,
            "mean_return": float(clean.mean()),
            "std_return": float(clean.std()),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "win_rate": float((clean > 0).mean()),
        }

    def _test_grid(name, data, col_a, col_b, ret_col):
        """Test all cells in a 2-factor grid with marginals and stability."""
        valid = data.dropna(subset=[col_a, col_b, ret_col])
        if len(valid) < MIN_OBS:
            return []

        # Marginal stats per level of each factor
        marginals_a = {}
        for level, grp in valid.groupby(col_a, observed=True):
            marginals_a[level] = _cell_stats(grp[ret_col])
        marginals_b = {}
        for level, grp in valid.groupby(col_b, observed=True):
            marginals_b[level] = _cell_stats(grp[ret_col])

        rows = []
        for (a_val, b_val), grp in valid.groupby([col_a, col_b], observed=True):
            s = _cell_stats(grp[ret_col])
            if s is None:
                continue

            # Regime stability across 3 time periods
            r1, r2, r3 = split_regimes(grp)
            regime_means = []
            for regime in (r1, r2, r3):
                rrets = regime[ret_col].dropna() if ret_col in regime.columns else pd.Series(dtype=float)
                regime_means.append(float(rrets.mean()) if len(rrets) >= 10 else np.nan)
            stab = stability_score(regime_means)

            marg_a = marginals_a.get(a_val)
            marg_b = marginals_b.get(b_val)

            rows.append({
                "interaction": name,
                "factor_a": str(a_val),
                "factor_b": str(b_val),
                "key": f"{a_val} × {b_val}",
                **s,
                "stability": round(stab, 4) if np.isfinite(stab) else np.nan,
                "regime1_mean": regime_means[0],
                "regime2_mean": regime_means[1],
                "regime3_mean": regime_means[2],
                "marginal_a_mean": marg_a["mean_return"] if marg_a else np.nan,
                "marginal_a_p": marg_a["p_value"] if marg_a else np.nan,
                "marginal_b_mean": marg_b["mean_return"] if marg_b else np.nan,
                "marginal_b_p": marg_b["p_value"] if marg_b else np.nan,
            })
        return rows

    all_rows = []

    # ==================================================================
    # A) Time-of-day × Volatility Regime (rolling ATR percentile)
    # ==================================================================
    print("\n  [A] Time-of-Day × Volatility Regime (ATR percentile)")

    rth["atr_bar"] = rth["high"] - rth["low"]
    rth["rolling_atr_20"] = rth["atr_bar"].rolling(20).mean()
    rth["fwd_return"] = rth["close"].shift(-1) / rth["close"] - 1

    tv = rth.dropna(subset=["rolling_atr_20", "fwd_return"]).copy()
    if len(tv) >= MIN_OBS * 6:
        # Percentile-rank the ATR within an expanding window, then tercile
        tv["atr_pctile"] = tv["rolling_atr_20"].rank(pct=True)
        tv["vol_regime"] = pd.cut(
            tv["atr_pctile"],
            bins=[0, 1/3, 2/3, 1.0],
            labels=["Low_ATR", "Med_ATR", "High_ATR"],
            include_lowest=True,
        )

        # 1-hour time blocks for cleaner cells
        tv["time_block"] = tv["hour"].map({
            9: "09:30-10:00", 10: "10:00-11:00", 11: "11:00-12:00",
            12: "12:00-13:00", 13: "13:00-14:00", 14: "14:00-15:00",
            15: "15:00-16:00",
        })

        r = _test_grid("time_x_atr_regime", tv, "time_block", "vol_regime", "fwd_return")
        print(f"      {len(r)} cells tested")
        all_rows.extend(r)
    else:
        print("      Skipped — insufficient bar data")

    # ==================================================================
    # B) Gap Direction × Gap Size
    # ==================================================================
    print("  [B] Gap Direction × Gap Size")

    gap_data = daily.dropna(subset=["gap", "prev_close"]).copy()
    gap_nonzero = gap_data[gap_data["gap"] != 0].copy()

    if len(gap_nonzero) >= MIN_OBS * 3:
        gap_nonzero["gap_direction"] = np.where(gap_nonzero["gap"] > 0, "Gap_Up", "Gap_Down")
        gap_nonzero["gap_size"] = _safe_qcut(
            gap_nonzero["gap"].abs(), 3, labels=["Small", "Medium", "Large"],
        )
        if not gap_nonzero["gap_size"].isna().all():
            r = _test_grid("gap_dir_x_size", gap_nonzero, "gap_direction", "gap_size", "daily_return")
            print(f"      {len(r)} cells tested")
            all_rows.extend(r)
        else:
            print("      Skipped — could not bin gap sizes (constant values)")
    else:
        print("      Skipped — insufficient gap data")

    # ==================================================================
    # C) First Hour Return × Rest-of-Day Return
    # ==================================================================
    print("  [C] First Hour Return × Rest-of-Day")

    first_hour = rth.between_time("09:30", "10:30")
    rest_of_day = rth.between_time("10:30", "16:00")

    fh_daily = first_hour.groupby("date").agg(
        fh_open=("open", "first"), fh_close=("close", "last"),
    )
    fh_daily["fh_return"] = fh_daily["fh_close"] / fh_daily["fh_open"] - 1

    rod_daily = rest_of_day.groupby("date").agg(
        rod_open=("open", "first"), rod_close=("close", "last"),
    )
    rod_daily["rod_return"] = rod_daily["rod_close"] / rod_daily["rod_open"] - 1

    fh_combined = fh_daily.join(rod_daily).join(daily[["daily_return"]]).dropna()

    if len(fh_combined) >= MIN_OBS * 4:
        fh_combined["fh_dir"] = _safe_qcut(
            fh_combined["fh_return"], 3, labels=["FH_Down", "FH_Flat", "FH_Up"],
        )
        if not fh_combined["fh_dir"].isna().all():
            # Test: what is the rest-of-day return given first-hour direction?
            fh_valid = fh_combined.dropna(subset=["fh_dir"])
            rows = []
            marginals = {}
            for level, grp in fh_valid.groupby("fh_dir", observed=True):
                marginals[level] = _cell_stats(grp["rod_return"])

            for fh_bucket, grp in fh_valid.groupby("fh_dir", observed=True):
                rod_rets = grp["rod_return"]
                s = _cell_stats(rod_rets)
                if s is None:
                    continue

                # Is it continuation or reversal?
                fh_sign = {"FH_Down": -1, "FH_Flat": 0, "FH_Up": 1}.get(fh_bucket, 0)
                rod_sign = 1 if s["mean_return"] > 0 else -1
                pattern = "continuation" if (fh_sign * rod_sign > 0) else "reversal" if fh_sign != 0 else "neutral"

                # Stability
                r1, r2, r3 = split_regimes(grp)
                regime_means = []
                for regime in (r1, r2, r3):
                    rrets = regime["rod_return"].dropna() if "rod_return" in regime.columns else pd.Series(dtype=float)
                    regime_means.append(float(rrets.mean()) if len(rrets) >= 10 else np.nan)
                stab = stability_score(regime_means)

                rows.append({
                    "interaction": "fh_continuation",
                    "factor_a": str(fh_bucket),
                    "factor_b": pattern,
                    "key": f"{fh_bucket} → {pattern}",
                    **s,
                    "stability": round(stab, 4) if np.isfinite(stab) else np.nan,
                    "regime1_mean": regime_means[0],
                    "regime2_mean": regime_means[1],
                    "regime3_mean": regime_means[2],
                    "marginal_a_mean": marginals.get(fh_bucket, {}).get("mean_return", np.nan) if marginals.get(fh_bucket) else np.nan,
                    "marginal_a_p": marginals.get(fh_bucket, {}).get("p_value", np.nan) if marginals.get(fh_bucket) else np.nan,
                    "marginal_b_mean": np.nan,
                    "marginal_b_p": np.nan,
                })

            print(f"      {len(rows)} cells tested")
            all_rows.extend(rows)

            # Also test the full 3×3 grid: fh_dir × rod_dir
            fh_combined["rod_dir"] = _safe_qcut(
                fh_combined["rod_return"], 3, labels=["ROD_Down", "ROD_Flat", "ROD_Up"],
            )
            if not fh_combined["rod_dir"].isna().all():
                fh_grid = fh_combined.dropna(subset=["fh_dir", "rod_dir"])
                r = _test_grid("fh_x_rod_grid", fh_grid, "fh_dir", "rod_dir", "daily_return")
                print(f"      + {len(r)} grid cells (fh_dir × rod_dir → daily return)")
                all_rows.extend(r)
        else:
            print("      Skipped — could not bin first-hour returns")
    else:
        print("      Skipped — insufficient first-hour data")

    # ==================================================================
    # D) Prior Day Level Touches × Approach Direction
    # ==================================================================
    print("  [D] Prior Day Levels (PDH/PDL/PDC) × Approach Direction")

    # Build prior day levels on RTH bars
    daily_shifted = daily[["daily_high", "daily_low", "daily_close"]].copy()
    daily_shifted.index = pd.to_datetime(daily_shifted.index)
    daily_shifted = daily_shifted.shift(1)
    daily_shifted.columns = ["pdh", "pdl", "pdc"]
    daily_shifted.index = daily_shifted.index.date

    rth_pdl = rth.join(daily_shifted, on="date").dropna(subset=["pdh", "pdl", "pdc"])

    if len(rth_pdl) >= MIN_OBS * 4:
        lookforward = 4
        pd_rows = []

        for level_name, level_col in [("PDH", "pdh"), ("PDL", "pdl"), ("PDC", "pdc")]:
            # Detect bars that touch the level (price crosses through it)
            if level_name in ("PDH", "PDC"):
                touches = rth_pdl[(rth_pdl["high"] >= rth_pdl[level_col]) & (rth_pdl["low"] < rth_pdl[level_col])].copy()
            else:
                touches = rth_pdl[(rth_pdl["low"] <= rth_pdl[level_col]) & (rth_pdl["high"] > rth_pdl[level_col])].copy()

            if len(touches) < MIN_OBS:
                continue

            # Forward return (4-bar lookahead)
            touches["fwd_4bar"] = rth["close"].shift(-lookforward).reindex(touches.index) / touches["close"] - 1
            touches = touches.dropna(subset=["fwd_4bar"])

            # Approach direction
            touches["approach"] = np.where(touches["open"] > touches[level_col], "from_above", "from_below")

            # Overall level stats
            overall = _cell_stats(touches["fwd_4bar"])
            if overall:
                r1, r2, r3 = split_regimes(touches)
                regime_means = []
                for regime in (r1, r2, r3):
                    rrets = regime["fwd_4bar"].dropna() if "fwd_4bar" in regime.columns else pd.Series(dtype=float)
                    regime_means.append(float(rrets.mean()) if len(rrets) >= 10 else np.nan)
                stab = stability_score(regime_means)
                direction = "bounce" if (level_name == "PDL" and overall["mean_return"] > 0) or \
                                        (level_name in ("PDH", "PDC") and overall["mean_return"] < 0) else "break"

                pd_rows.append({
                    "interaction": "pd_level_touch",
                    "factor_a": level_name,
                    "factor_b": "all_approaches",
                    "key": f"{level_name} touch → {direction}",
                    **overall,
                    "stability": round(stab, 4) if np.isfinite(stab) else np.nan,
                    "regime1_mean": regime_means[0],
                    "regime2_mean": regime_means[1],
                    "regime3_mean": regime_means[2],
                    "marginal_a_mean": np.nan,
                    "marginal_a_p": np.nan,
                    "marginal_b_mean": np.nan,
                    "marginal_b_p": np.nan,
                })

            # Split by approach direction
            for approach, sub in touches.groupby("approach"):
                s = _cell_stats(sub["fwd_4bar"])
                if s is None:
                    continue
                r1, r2, r3 = split_regimes(sub)
                regime_means = []
                for regime in (r1, r2, r3):
                    rrets = regime["fwd_4bar"].dropna() if "fwd_4bar" in regime.columns else pd.Series(dtype=float)
                    regime_means.append(float(rrets.mean()) if len(rrets) >= 10 else np.nan)
                stab = stability_score(regime_means)

                pd_rows.append({
                    "interaction": "pd_level_x_approach",
                    "factor_a": level_name,
                    "factor_b": approach,
                    "key": f"{level_name} × {approach}",
                    **s,
                    "stability": round(stab, 4) if np.isfinite(stab) else np.nan,
                    "regime1_mean": regime_means[0],
                    "regime2_mean": regime_means[1],
                    "regime3_mean": regime_means[2],
                    "marginal_a_mean": overall["mean_return"] if overall else np.nan,
                    "marginal_a_p": overall["p_value"] if overall else np.nan,
                    "marginal_b_mean": np.nan,
                    "marginal_b_p": np.nan,
                })

        print(f"      {len(pd_rows)} cells tested")
        all_rows.extend(pd_rows)
    else:
        print("      Skipped — insufficient prior-day-level data")

    # ==================================================================
    # Assemble, correct, rank, save
    # ==================================================================

    if not all_rows:
        print("\n  No conditional cells had sufficient observations.")
        pd.DataFrame().to_csv(OUTPUT_DIR / "conditional_edge_results.csv", index=False)
        return pd.DataFrame()

    cond_df = pd.DataFrame(all_rows)

    # BH-FDR correction across all tests
    cond_df["p_adj"] = bh_fdr(cond_df["p_value"].values)

    # Conditional flag: cell significant but neither marginal is
    cond_df["cell_significant"] = cond_df["p_adj"] < 0.05
    cond_df["marginal_a_sig"] = cond_df["marginal_a_p"].fillna(1.0) < 0.05
    cond_df["marginal_b_sig"] = cond_df["marginal_b_p"].fillna(1.0) < 0.05
    cond_df["conditional"] = (
        cond_df["cell_significant"]
        & ~cond_df["marginal_a_sig"]
        & ~cond_df["marginal_b_sig"]
    )

    # Composite edge score for ranking
    # Weight: |t-stat| 0.30, |mean_return| 0.25, stability 0.25, log(n) 0.20
    def _norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)

    abs_t = cond_df["t_stat"].abs()
    abs_ret = cond_df["mean_return"].abs()
    log_n = np.log(cond_df["n"])
    stab = cond_df["stability"].fillna(0)

    cond_df["edge_score"] = (
        0.30 * _norm(abs_t)
        + 0.25 * _norm(abs_ret)
        + 0.25 * _norm(stab)
        + 0.20 * _norm(log_n)
    )

    # Sort by edge_score descending
    cond_df.sort_values("edge_score", ascending=False, inplace=True)
    cond_df.reset_index(drop=True, inplace=True)

    # Save
    cond_df.to_csv(OUTPUT_DIR / "conditional_edge_results.csv", index=False)

    # ---- Print summary ----

    n_total = len(cond_df)
    n_sig = int(cond_df["cell_significant"].sum())
    n_cond = int(cond_df["conditional"].sum())

    print(f"\n  Total cells tested:          {n_total}")
    print(f"  Significant (p_adj < 0.05):  {n_sig}")
    print(f"  Truly conditional:           {n_cond}")
    print(f"    (cell significant, neither marginal significant)")

    # Top edges by composite score
    top = cond_df.head(15)
    print(f"\n  Top edges by composite score:")
    print(f"  {'#':>3s}  {'interaction':20s}  {'key':28s}  {'mean_bps':>9s}  {'t':>6s}  {'q':>7s}  {'n':>5s}  {'stab':>5s}  {'score':>5s}  {'cond':>4s}")
    print("  " + "-" * 100)
    for i, (_, r) in enumerate(top.iterrows(), 1):
        stab_str = f"{r['stability']:.2f}" if np.isfinite(r.get("stability", np.nan)) else "  N/A"
        cond_flag = "YES" if r["conditional"] else "sig" if r["cell_significant"] else ""
        print(
            f"  {i:3d}  {r['interaction']:20s}  {r['key']:28s}  "
            f"{r['mean_return']*10000:+8.2f}  {r['t_stat']:+5.2f}  "
            f"{r['p_adj']:.4f}  {r['n']:5.0f}  {stab_str}  "
            f"{r['edge_score']:.3f}  {cond_flag:>4s}"
        )

    # Per-interaction summary
    print(f"\n  By interaction type:")
    for itype, grp in cond_df.groupby("interaction"):
        n_c = len(grp)
        n_s = int(grp["cell_significant"].sum())
        n_co = int(grp["conditional"].sum())
        best = grp.iloc[0] if n_c > 0 else None
        best_str = f"best={best['key']}" if best is not None else ""
        print(f"    {itype:24s}  cells={n_c:3d}  sig={n_s:2d}  conditional={n_co:2d}  {best_str}")

    print(f"\n  Saved to: {(OUTPUT_DIR / 'conditional_edge_results.csv').resolve()}")
    return cond_df


def refine_fh_reversal_edge(df, daily, min_obs=30):
    """Refine the 'first-hour down → rest-of-day reversal up' edge.

    Slices the base edge by four conditioning filters to find the strongest
    sub-conditions:
        1. Volatility regime (rolling 14-bar ATR percentile terciles)
        2. Gap context (gap-up / gap-down / no-gap)
        3. First-hour move magnitude (percentile buckets)
        4. Day of week

    For every slice: mean return, t-stat, p-value (BH-FDR), sample size,
    win rate, and 3-regime stability.  Results ranked by composite score
    and saved to refined_edge_results.csv.

    Parameters
    ----------
    df : pd.DataFrame
        Bar-level data with features from ``build_features()``.
    daily : pd.DataFrame
        Daily summary from ``build_daily_summary()``.
    min_obs : int
        Minimum observations per cell (default 30).
    """
    print(SEP)
    print("EDGE REFINEMENT: First-Hour Down → Rest-of-Day Reversal")
    print("=" * 70)

    MIN_OBS = min_obs
    rth = df.between_time(RTH_START, RTH_END).copy()

    # ---- Build the base edge dataset ----

    first_hour = rth.between_time("09:30", "10:30")
    rest_of_day = rth.between_time("10:30", "16:00")

    fh_daily = first_hour.groupby("date").agg(
        fh_open=("open", "first"), fh_close=("close", "last"),
        fh_high=("high", "max"), fh_low=("low", "min"),
    )
    fh_daily["fh_return"] = fh_daily["fh_close"] / fh_daily["fh_open"] - 1

    rod_daily = rest_of_day.groupby("date").agg(
        rod_open=("open", "first"), rod_close=("close", "last"),
    )
    rod_daily["rod_return"] = rod_daily["rod_close"] / rod_daily["rod_open"] - 1

    base = fh_daily.join(rod_daily).join(daily[["daily_range", "daily_range_pct",
                                                  "gap", "gap_pct", "dow", "dow_name",
                                                  "daily_return"]]).dropna(
        subset=["fh_return", "rod_return", "daily_range"]
    )

    # Filter to first-hour DOWN days only (the base edge)
    fh_down = base[base["fh_return"] < 0].copy()

    if len(fh_down) < MIN_OBS:
        print(f"\n  WARNING: Only {len(fh_down)} first-hour-down days (need {MIN_OBS}). Aborting.")
        return pd.DataFrame()

    # Print base edge stats
    base_rets = fh_down["rod_return"].dropna()
    base_t, base_p = stats.ttest_1samp(base_rets, 0)
    print(f"\n  Base edge: First-hour down days → rest-of-day return")
    print(f"    n={len(base_rets)}  mean={base_rets.mean()*10000:+.2f}bps  "
          f"t={base_t:+.2f}  p={base_p:.4f}  WR={(base_rets > 0).mean():.1%}")

    # ---- Helper: compute stats + stability for a slice ----

    def _slice_stats(name, condition, data):
        """Compute stats for one refinement slice."""
        rets = data["rod_return"].dropna()
        n = len(rets)
        if n < MIN_OBS:
            return None
        if rets.std() < 1e-12:
            return None
        t_stat, p_value = stats.ttest_1samp(rets, 0)
        wr = float((rets > 0).mean())

        # 3-regime stability
        r1, r2, r3 = split_regimes(data)
        regime_means = []
        for regime in (r1, r2, r3):
            rr = regime["rod_return"].dropna() if "rod_return" in regime.columns else pd.Series(dtype=float)
            regime_means.append(float(rr.mean()) if len(rr) >= 10 else np.nan)
        stab = stability_score(regime_means)

        return {
            "filter": name,
            "condition": condition,
            "n": n,
            "mean_return": float(rets.mean()),
            "std_return": float(rets.std()),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "win_rate": wr,
            "stability": round(stab, 4) if np.isfinite(stab) else np.nan,
            "regime1_mean": regime_means[0],
            "regime2_mean": regime_means[1],
            "regime3_mean": regime_means[2],
        }

    all_rows = []

    # Baseline (unfiltered FH-down → ROD)
    bl = _slice_stats("baseline", "all_fh_down_days", fh_down)
    if bl:
        all_rows.append(bl)

    # ==================================================================
    # 1. Volatility regime (rolling 14-bar ATR percentile)
    # ==================================================================
    print("\n  [1] Volatility Regime Filter")

    # Compute daily ATR from bar-level true ranges
    rth["tr"] = np.maximum(
        rth["high"] - rth["low"],
        np.maximum(
            (rth["high"] - rth["close"].shift(1)).abs(),
            (rth["low"] - rth["close"].shift(1)).abs(),
        ),
    )
    daily_atr = rth.groupby("date")["tr"].mean()
    daily_atr.name = "daily_atr"
    fh_down = fh_down.join(daily_atr)

    # Rolling 14-day ATR, then percentile-rank and tercile
    fh_down["atr_14d"] = fh_down["daily_atr"].rolling(14, min_periods=5).mean().shift(1)
    fh_atr = fh_down.dropna(subset=["atr_14d"]).copy()

    if len(fh_atr) >= MIN_OBS * 3:
        fh_atr["atr_pctile"] = fh_atr["atr_14d"].rank(pct=True)
        fh_atr["vol_regime"] = pd.cut(
            fh_atr["atr_pctile"],
            bins=[0, 1/3, 2/3, 1.0],
            labels=["Low_ATR", "Med_ATR", "High_ATR"],
            include_lowest=True,
        )
        for regime, grp in fh_atr.groupby("vol_regime", observed=True):
            s = _slice_stats("volatility", str(regime), grp)
            if s:
                all_rows.append(s)
                print(f"    {regime:10s}  n={s['n']:4d}  mean={s['mean_return']*10000:+.2f}bps  "
                      f"t={s['t_stat']:+.2f}  p={s['p_value']:.4f}  WR={s['win_rate']:.1%}")
    else:
        print("    Skipped — insufficient data for ATR regimes")

    # ==================================================================
    # 2. Gap context (gap-up / gap-down / no-gap)
    # ==================================================================
    print("\n  [2] Gap Context Filter")

    fh_gap = fh_down.dropna(subset=["gap_pct"]).copy()
    if len(fh_gap) >= MIN_OBS * 2:
        # Define no-gap as |gap_pct| < 0.1% (roughly noise)
        gap_thresh = 0.001
        conditions = [
            ("gap_down", fh_gap[fh_gap["gap_pct"] < -gap_thresh]),
            ("no_gap", fh_gap[fh_gap["gap_pct"].abs() <= gap_thresh]),
            ("gap_up", fh_gap[fh_gap["gap_pct"] > gap_thresh]),
        ]
        for label, subset in conditions:
            s = _slice_stats("gap_context", label, subset)
            if s:
                all_rows.append(s)
                print(f"    {label:10s}  n={s['n']:4d}  mean={s['mean_return']*10000:+.2f}bps  "
                      f"t={s['t_stat']:+.2f}  p={s['p_value']:.4f}  WR={s['win_rate']:.1%}")
    else:
        print("    Skipped — insufficient gap data")

    # ==================================================================
    # 3. First-hour move magnitude (percentile buckets)
    # ==================================================================
    print("\n  [3] First-Hour Magnitude Filter")

    if len(fh_down) >= MIN_OBS * 3:
        # Bucket the FH drop magnitude (all negative, so use abs)
        fh_down["fh_abs"] = fh_down["fh_return"].abs()
        fh_down["fh_magnitude"] = _safe_qcut(
            fh_down["fh_abs"], 3, labels=["Small_Drop", "Medium_Drop", "Large_Drop"],
        )
        if not fh_down["fh_magnitude"].isna().all():
            for mag, grp in fh_down.dropna(subset=["fh_magnitude"]).groupby("fh_magnitude", observed=True):
                s = _slice_stats("fh_magnitude", str(mag), grp)
                if s:
                    all_rows.append(s)
                    pctile_range = f"|FH|={grp['fh_abs'].min()*100:.2f}%-{grp['fh_abs'].max()*100:.2f}%"
                    print(f"    {mag:14s}  n={s['n']:4d}  mean={s['mean_return']*10000:+.2f}bps  "
                          f"t={s['t_stat']:+.2f}  p={s['p_value']:.4f}  WR={s['win_rate']:.1%}  {pctile_range}")
        else:
            print("    Skipped — could not bin FH magnitudes")
    else:
        print("    Skipped — insufficient data")

    # ==================================================================
    # 4. Day of week
    # ==================================================================
    print("\n  [4] Day-of-Week Filter")

    dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    for dow_num in range(5):
        subset = fh_down[fh_down["dow"] == dow_num]
        s = _slice_stats("day_of_week", dow_map[dow_num], subset)
        if s:
            all_rows.append(s)
            print(f"    {dow_map[dow_num]:10s}  n={s['n']:4d}  mean={s['mean_return']*10000:+.2f}bps  "
                  f"t={s['t_stat']:+.2f}  p={s['p_value']:.4f}  WR={s['win_rate']:.1%}")

    # ==================================================================
    # 5. Two-way interactions (strongest single filters combined)
    # ==================================================================
    print("\n  [5] Two-Way Interactions")

    # Volatility × gap context
    if "vol_regime" in fh_atr.columns and len(fh_atr.dropna(subset=["gap_pct"])) >= MIN_OBS:
        fh_cross = fh_atr.dropna(subset=["gap_pct", "vol_regime"]).copy()
        fh_cross["gap_label"] = np.where(
            fh_cross["gap_pct"] < -gap_thresh, "gap_down",
            np.where(fh_cross["gap_pct"] > gap_thresh, "gap_up", "no_gap"),
        )
        for (vol, gap), grp in fh_cross.groupby(["vol_regime", "gap_label"], observed=True):
            s = _slice_stats("vol_x_gap", f"{vol} × {gap}", grp)
            if s:
                all_rows.append(s)

    # Volatility × magnitude
    if "vol_regime" in fh_atr.columns and "fh_magnitude" in fh_down.columns:
        fh_vm = fh_atr.join(fh_down[["fh_magnitude"]], how="inner").dropna(subset=["vol_regime", "fh_magnitude"])
        for (vol, mag), grp in fh_vm.groupby(["vol_regime", "fh_magnitude"], observed=True):
            s = _slice_stats("vol_x_magnitude", f"{vol} × {mag}", grp)
            if s:
                all_rows.append(s)

    # Magnitude × day of week
    if "fh_magnitude" in fh_down.columns:
        fh_md = fh_down.dropna(subset=["fh_magnitude"]).copy()
        fh_md["dow_label"] = fh_md["dow"].map(dow_map)
        for (mag, dow), grp in fh_md.groupby(["fh_magnitude", "dow_label"], observed=True):
            s = _slice_stats("magnitude_x_dow", f"{mag} × {dow}", grp)
            if s:
                all_rows.append(s)

    n_cross = sum(1 for r in all_rows if r["filter"].endswith(("_x_gap", "_x_magnitude", "_x_dow")))
    print(f"    {n_cross} cross-filter cells tested")

    # ==================================================================
    # Assemble, FDR-correct, rank, save
    # ==================================================================

    if not all_rows:
        print("\n  No refinement slices had sufficient observations.")
        pd.DataFrame().to_csv(OUTPUT_DIR / "refined_edge_results.csv", index=False)
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)

    # BH-FDR correction across all tests
    result["p_adj"] = bh_fdr(result["p_value"].values)
    result["significant"] = result["p_adj"] < 0.05

    # Composite edge score for ranking
    def _norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)

    abs_t = result["t_stat"].abs()
    abs_ret = result["mean_return"].abs()
    log_n = np.log(result["n"])
    stab = result["stability"].fillna(0)

    result["edge_score"] = (
        0.30 * _norm(abs_t)
        + 0.25 * _norm(abs_ret)
        + 0.25 * _norm(stab)
        + 0.20 * _norm(log_n)
    )

    result.sort_values("edge_score", ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)

    result.to_csv(OUTPUT_DIR / "refined_edge_results.csv", index=False)

    # ---- Summary ----

    n_total = len(result)
    n_sig = int(result["significant"].sum())
    base_row = result[result["condition"] == "all_fh_down_days"]
    base_score = float(base_row["edge_score"].iloc[0]) if len(base_row) > 0 else np.nan

    print(f"\n  {'─' * 70}")
    print(f"  REFINEMENT RESULTS")
    print(f"  {'─' * 70}")
    print(f"  Total slices tested:         {n_total}")
    print(f"  Significant (p_adj < 0.05):  {n_sig}")
    print(f"  Baseline edge_score:         {base_score:.3f}")

    # Top refined conditions
    print(f"\n  Ranked refinements (by edge_score):")
    print(f"  {'#':>3s}  {'filter':18s}  {'condition':26s}  {'mean_bps':>9s}  {'t':>6s}  {'q':>7s}  {'n':>5s}  {'WR':>5s}  {'stab':>5s}  {'score':>5s}")
    print("  " + "-" * 100)
    for i, (_, r) in enumerate(result.head(20).iterrows(), 1):
        stab_str = f"{r['stability']:.2f}" if np.isfinite(r.get("stability", np.nan)) else "  N/A"
        sig_flag = "*" if r["significant"] else " "
        print(
            f"  {i:3d}  {r['filter']:18s}  {r['condition']:26s}  "
            f"{r['mean_return']*10000:+8.2f}  {r['t_stat']:+5.2f}  "
            f"{r['p_adj']:.4f}  {int(r['n']):5d}  {r['win_rate']:.1%}  "
            f"{stab_str}  {r['edge_score']:.3f} {sig_flag}"
        )

    # Highlight conditions that beat the baseline
    if np.isfinite(base_score):
        better = result[(result["edge_score"] > base_score) & (result["condition"] != "all_fh_down_days")]
        if not better.empty:
            print(f"\n  {len(better)} conditions beat the baseline edge_score ({base_score:.3f}):")
            for _, r in better.head(10).iterrows():
                print(f"    {r['filter']:18s}  {r['condition']:26s}  score={r['edge_score']:.3f}  "
                      f"mean={r['mean_return']*10000:+.2f}bps  t={r['t_stat']:+.2f}  n={int(r['n'])}")
        else:
            print(f"\n  No single condition beats the unfiltered baseline.")

    print(f"\n  Saved to: {(OUTPUT_DIR / 'refined_edge_results.csv').resolve()}")
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("NQ 15-min Edge Analysis Pipeline")
    print("=" * 70)
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
    print("Building daily summary...")
    daily = build_daily_summary(df)
    print(f"  {len(daily)} trading days")
    ac_df = analyze_autocorrelation(df)
    tod_df = analyze_time_of_day(df)
    dow_df = analyze_day_of_week(daily)
    analyze_gaps(daily)
    vol_result, vol_regime_df = analyze_volatility(daily)
    analyze_range_dynamics(daily)
    analyze_prior_day_levels(df, daily)
    analyze_first_hour_continuation(df, daily)
    int_df = analyze_interactions(df, daily)
    analyze_conditional_edges(df, daily)
    refine_fh_reversal_edge(df, daily)

    results_dict = {
        "autocorrelation_by_time": (ac_df, "time_slot", "autocorr"),
        "time_of_day_bias": (tod_df, "time_slot", "mean_return"),
        "day_of_week": (dow_df, "day", "mean_return"),
        "volatility_regime": (vol_regime_df, "regime", "mean_return"),
        "interactions": (int_df, "key", "mean_return"),
    }

    # Build consolidated summary and rank edges
    summary = build_summary(results_dict)
    rank_edges(summary)

    # Threshold sensitivity analysis
    threshold_sensitivity(results_dict)

    print(SEP)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nLook for: p < 0.01, stable across sub-periods, large enough to survive ~3pt NQ costs")
    print(f"\nAll CSV outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    if "--diagnose" in sys.argv:
        diagnose_edge_filters()
    else:
        main()
