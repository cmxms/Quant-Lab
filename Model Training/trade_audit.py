"""
trade_audit.py
==============
Winning Trade DNA Audit + Simplified Re-Test

Pipeline Stage : Post-Backtest Analysis (Phase 5b)

Steps
-----
1. Rerun the 0.45-threshold simulation, recording all 14 feature values
   at each signal bar for both winning (TP) and losing (SL/EOD) trades.
2. Compute mean feature values for wins vs. losses.
3. Rank by Cohen's d to find the "Power 3" most separating indicators.
4. Run a simplified re-test using ONLY those 3 features as an additional
   filter gate, with symmetric 2.0/2.0 ATR barriers (1:1 R:R).

Usage
-----
    python trade_audit.py
    python trade_audit.py --threshold 0.45
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# ── Configuration ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_HERE               = Path(__file__).parent.resolve()
_FEATURE_OUTPUT_DIR = _HERE.parent / "Feature Engineering" / "Outputs"
_MODEL_OUTPUT_DIR   = _HERE / "Outputs"

TARGET_COL   = "Target"
DROP_COLS    = ["Open","High","Low","Close","Volume","Minutes_From_Open","Day_Of_Week"]
CANONICAL    = np.array([-1, 0, 1])

NQ_MULT      = 20.0
SLIPPAGE_PTS = 0.25
COMMISSION   = 4.00
EOD_HOUR     = 15
EOD_MINUTE   = 55

# Primary backtest params
PRIMARY_THRESHOLD = 0.45
PRIMARY_TP_MULT   = 6.0
PRIMARY_SL_MULT   = 4.0

# Simplified re-test params
RETEST_TP_MULT = 2.0
RETEST_SL_MULT = 2.0

W_LGBM   = 0.50
W_RF     = 0.35
W_LOGREG = 0.15

FEATURE_COLS = [
    "9EMA_Dist","200EMA_Dist","EMA_Spread","VWAP_Dist",
    "RSI_14","WillR_14","MACD_Hist_Norm","Rolling_15m_Return",
    "BB_PctB","ATR_Norm","RVOL","ADX_14","RVOL_60m","Body_Wick_Ratio_10",
]

# ---------------------------------------------------------------------------
# ── Logging ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
DIVIDER = "-" * 68

def _section(title):
    log.info(DIVIDER)
    log.info(f"  {title}")
    log.info(DIVIDER)

# ---------------------------------------------------------------------------
# ── Helpers ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _latest(directory, pattern, label):
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        log.error(f"[MISSING] No '{pattern}' in {directory}")
        sys.exit(1)
    log.info(f"  {label:<12}: {matches[-1].name}")
    return matches[-1]

def _align(proba, model_classes):
    out = np.zeros((proba.shape[0], 3), dtype=np.float64)
    for i, cls in enumerate(CANONICAL):
        col = np.where(model_classes == cls)[0]
        if col.size:
            out[:, i] = proba[:, col[0]]
    return out

def _cohens_d(a, b):
    """Cohen's d effect size between two arrays."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

# ---------------------------------------------------------------------------
# ── Build Signal DataFrame ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def build_df(holdout_path: Path) -> pd.DataFrame:
    _section("Step 1 / 4  -  Computing Weighted Vote")

    log.info("  Loading models ...")
    lgbm  = joblib.load(_latest(_MODEL_OUTPUT_DIR, "lgbm_model_*.joblib",     "LGBM"))
    rf    = joblib.load(_latest(_MODEL_OUTPUT_DIR, "rf_model_*.joblib",       "RF"))
    lrpipe = joblib.load(_latest(_MODEL_OUTPUT_DIR, "logreg_pipeline_*.joblib","LogReg"))
    lr_inner = lrpipe.named_steps["model"]

    log.info(f"  Loading: {holdout_path.name}")
    df = pd.read_parquet(holdout_path)

    drop = [c for c in [TARGET_COL] + DROP_COLS if c in df.columns]
    X    = df.drop(columns=drop)

    log.info("  Running predict_proba() ...")
    lgbm_p1 = _align(lgbm.predict_proba(X),   lgbm.classes_)[:, 2]
    rf_p1   = _align(rf.predict_proba(X),     rf.classes_)[:, 2]
    lr_p1   = _align(lrpipe.predict_proba(X), lr_inner.classes_)[:, 2]

    df["weighted_vote"] = lgbm_p1 * W_LGBM + rf_p1 * W_RF + lr_p1 * W_LOGREG
    df["atr_pts"]       = df["ATR_Norm"] * df["Close"]

    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_convert("US/Eastern").tz_localize(None)

    log.info(f"  Done. Vote range: [{df['weighted_vote'].min():.4f}, {df['weighted_vote'].max():.4f}]")
    return df

# ---------------------------------------------------------------------------
# ── Simulation with Feature Snapshot Recording ───────────────────────────────
# ---------------------------------------------------------------------------

def simulate_with_audit(df: pd.DataFrame, threshold: float,
                        tp_mult: float, sl_mult: float,
                        extra_filters: dict = None) -> pd.DataFrame:
    """
    Run bar-by-bar simulation and record feature values at each signal bar.

    extra_filters: dict of {feature_name: (direction, cutoff)}
        direction = 'gt' (feature > cutoff) or 'lt' (feature < cutoff)
    Returns DataFrame of trade records.
    """
    opens   = df["Open"].to_numpy(np.float64)
    highs   = df["High"].to_numpy(np.float64)
    lows    = df["Low"].to_numpy(np.float64)
    closes  = df["Close"].to_numpy(np.float64)
    votes   = df["weighted_vote"].to_numpy(np.float64)
    atrs    = df["atr_pts"].to_numpy(np.float64)
    hours   = df.index.hour
    minutes = df.index.minute
    n       = len(df)

    # Pre-extract feature arrays for speed
    feat_arrays = {f: df[f].to_numpy(np.float64) for f in FEATURE_COLS if f in df.columns}

    records = []
    i = 0
    while i < n - 1:
        eod = hours[i] > EOD_HOUR or (hours[i] == EOD_HOUR and minutes[i] >= EOD_MINUTE)
        if eod or votes[i] <= threshold:
            i += 1
            continue

        # Apply Power 3 filters if provided
        if extra_filters:
            passed = True
            for feat, (direction, cutoff) in extra_filters.items():
                val = feat_arrays.get(feat, np.zeros(n))[i]
                if direction == "gt" and val <= cutoff:
                    passed = False; break
                if direction == "lt" and val >= cutoff:
                    passed = False; break
            if not passed:
                i += 1
                continue

        # Snapshot feature values at signal bar
        snap = {f: feat_arrays[f][i] for f in feat_arrays}
        snap["vote"] = votes[i]

        entry_bar = i + 1
        entry_px  = opens[entry_bar] + SLIPPAGE_PTS
        atr       = atrs[i]
        tp        = entry_px + tp_mult * atr
        sl        = entry_px - sl_mult * atr

        exit_px, exit_type = None, None
        for j in range(entry_bar, n):
            bar_eod = hours[j] > EOD_HOUR or (hours[j] == EOD_HOUR and minutes[j] >= EOD_MINUTE)
            if bar_eod:
                exit_px, exit_type = closes[j] - SLIPPAGE_PTS, "EOD"
                i = j + 1; break
            hit_tp = highs[j] >= tp
            hit_sl = lows[j]  <= sl
            if hit_tp and hit_sl:
                exit_px, exit_type = sl, "SL"; i = j + 1; break
            elif hit_sl:
                exit_px, exit_type = sl, "SL"; i = j + 1; break
            elif hit_tp:
                exit_px, exit_type = tp, "TP"; i = j + 1; break
        else:
            exit_px, exit_type = closes[-1], "EOD"; i = n

        if exit_px is None:
            i += 1; continue

        net_pnl = (exit_px - entry_px) * NQ_MULT - COMMISSION
        snap["exit_type"] = exit_type
        snap["net_pnl"]   = net_pnl
        snap["won"]       = exit_type == "TP"
        records.append(snap)

    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# ── Main ────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Winning Trade DNA Audit")
    parser.add_argument("--threshold",    type=float, default=PRIMARY_THRESHOLD)
    parser.add_argument("--holdout_path", type=Path,  default=None)
    args = parser.parse_args()

    holdout_path = args.holdout_path or (_FEATURE_OUTPUT_DIR / "long_clean_holdout.parquet")
    if not holdout_path.exists():
        log.error(f"[MISSING] {holdout_path}")
        sys.exit(1)

    log.info("")
    _section("Trade DNA Audit  -  Quantitative Trading Pipeline  |  Phase 5b")
    log.info(f"  Primary threshold : {args.threshold}")
    log.info(f"  Primary barriers  : TP={PRIMARY_TP_MULT}x ATR  /  SL={PRIMARY_SL_MULT}x ATR")
    log.info(f"  Re-test barriers  : TP={RETEST_TP_MULT}x ATR  /  SL={RETEST_SL_MULT}x ATR (1:1)")
    log.info("")

    df = build_df(holdout_path)

    # ── Step 2: Simulate & Record ────────────────────────────────────────────
    _section("Step 2 / 4  -  Simulating 0.45 Threshold with Feature Snapshots")
    trades_df = simulate_with_audit(df, args.threshold, PRIMARY_TP_MULT, PRIMARY_SL_MULT)

    total  = len(trades_df)
    wins   = trades_df[trades_df["won"] == True]
    losses = trades_df[trades_df["won"] == False]

    log.info(f"  Total trades  : {total}")
    log.info(f"  TP exits (W)  : {len(wins)}  ({len(wins)/total*100:.1f}%)")
    log.info(f"  SL/EOD  (L)   : {len(losses)}  ({len(losses)/total*100:.1f}%)")
    log.info(f"  Total Net P&L : ${trades_df['net_pnl'].sum():+,.0f}")

    # ── Step 3: Power 3 Analysis ─────────────────────────────────────────────
    _section("Step 3 / 4  -  Win vs. Loss Feature Analysis  (Cohen's d Ranking)")

    feat_cols = [f for f in FEATURE_COLS if f in trades_df.columns]
    analysis  = []

    for feat in feat_cols:
        w_vals = wins[feat].dropna().to_numpy()
        l_vals = losses[feat].dropna().to_numpy()
        d      = _cohens_d(w_vals, l_vals)
        t_stat, p_val = stats.ttest_ind(w_vals, l_vals, equal_var=False)
        analysis.append({
            "feature":    feat,
            "win_mean":   w_vals.mean() if len(w_vals) else np.nan,
            "loss_mean":  l_vals.mean() if len(l_vals) else np.nan,
            "delta":      w_vals.mean() - l_vals.mean() if len(w_vals) and len(l_vals) else np.nan,
            "cohens_d":   d,
            "abs_d":      abs(d),
            "p_value":    p_val,
            "direction":  "gt" if d > 0 else "lt",   # win_mean > loss_mean -> filter gt
        })

    audit = pd.DataFrame(analysis).sort_values("abs_d", ascending=False)

    log.info(f"  {'Feature':<22} {'Win Mean':>10} {'Loss Mean':>11} {'Delta':>10} {'Cohen d':>9} {'p-value':>9}")
    log.info(f"  {'-'*22} {'-'*10} {'-'*11} {'-'*10} {'-'*9} {'-'*9}")
    for _, row in audit.iterrows():
        sig = "***" if row["p_value"] < 0.001 else ("** " if row["p_value"] < 0.01 else "*  " if row["p_value"] < 0.05 else "   ")
        log.info(
            f"  {row['feature']:<22} {row['win_mean']:>10.4f} {row['loss_mean']:>11.4f}"
            f" {row['delta']:>+10.4f} {row['cohens_d']:>+9.4f} {row['p_value']:>9.4f} {sig}"
        )

    top3 = audit.head(3)
    power3 = {}

    log.info("")
    log.info("  *** THE POWER 3 (Largest Win/Loss Separation) ***")
    log.info("")
    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        direction = row["direction"]
        midpoint  = (row["win_mean"] + row["loss_mean"]) / 2
        power3[row["feature"]] = (direction, midpoint)
        arrow = ">" if direction == "gt" else "<"
        log.info(f"  #{rank}  {row['feature']:<22}")
        log.info(f"       Win avg  : {row['win_mean']:>10.4f}")
        log.info(f"       Loss avg : {row['loss_mean']:>10.4f}")
        log.info(f"       Delta    : {row['delta']:>+10.4f}  (Cohen d = {row['cohens_d']:+.3f})")
        log.info(f"       Filter   : {row['feature']} {arrow} {midpoint:.4f}  (win/loss midpoint)")
        log.info("")

    # ── Step 4: Simplified Re-Test ───────────────────────────────────────────
    _section("Step 4 / 4  -  Simplified Re-Test: Power 3 Filter + 2.0/2.0 ATR")

    filter_desc = "  |  ".join(
        f"{f} {'>' if d=='gt' else '<'} {c:.4f}"
        for f, (d, c) in power3.items()
    )
    log.info(f"  Active filters : vote > {args.threshold}  AND  {filter_desc}")
    log.info(f"  Barriers       : TP = +{RETEST_TP_MULT}x ATR  |  SL = -{RETEST_SL_MULT}x ATR (1:1 R:R)")
    log.info(f"  Break-even WR  : ~52% (after $4 commission on ~1-2 ATR trades)")
    log.info("")

    retest_df = simulate_with_audit(
        df, args.threshold, RETEST_TP_MULT, RETEST_SL_MULT,
        extra_filters=power3,
    )

    if len(retest_df) == 0:
        log.warning("  No trades generated with Power 3 filters. Filters may be too restrictive.")
    else:
        rt_total  = len(retest_df)
        rt_wins   = retest_df["won"].sum()
        rt_wr     = rt_wins / rt_total * 100
        rt_pnl    = retest_df["net_pnl"].sum()
        rt_avg    = rt_pnl / rt_total
        rt_tp     = (retest_df["exit_type"] == "TP").sum()
        rt_sl     = (retest_df["exit_type"] == "SL").sum()
        rt_eod    = (retest_df["exit_type"] == "EOD").sum()

        # Max drawdown
        cum = retest_df["net_pnl"].cumsum().to_numpy()
        running_max = np.maximum.accumulate(cum)
        max_dd = float((running_max - cum).max()) if len(cum) else 0

        log.info(f"  Total Trades   : {rt_total}")
        log.info(f"  TP / SL / EOD  : {rt_tp} / {rt_sl} / {rt_eod}")
        log.info(f"  Win Rate       : {rt_wr:.1f}%  (break-even ~52%)")
        log.info(f"  Total Net P&L  : ${rt_pnl:+,.0f}")
        log.info(f"  Avg per Trade  : ${rt_avg:+,.0f}")
        log.info(f"  Max Drawdown   : ${max_dd:,.0f}")
        log.info("")

        verdict_pnl   = "PROFITABLE" if rt_pnl   > 0  else "UNPROFITABLE"
        verdict_wr    = "ABOVE"      if rt_wr    > 52 else "BELOW"
        verdict_freq  = "SUFFICIENT" if rt_total > 20 else "LOW  (< 20 trades -- statistically thin)"
        log.info(f"  Verdict (P&L)  : {verdict_pnl}")
        log.info(f"  Verdict (WR)   : {verdict_wr} break-even  ({rt_wr:.1f}% vs 52% target)")
        log.info(f"  Verdict (Freq) : {verdict_freq}")
        log.info("")

        # ── Comparison Table ─────────────────────────────────────────────────
        primary_pnl = trades_df["net_pnl"].sum()
        primary_wr  = trades_df["won"].mean() * 100
        primary_n   = len(trades_df)

        log.info("  Comparison: Original vs. Re-Test")
        log.info(f"  {'':30} {'Original 6/4 ATR':>18} {'Retest 2/2 ATR':>16}")
        log.info(f"  {'-'*30} {'-'*18} {'-'*16}")
        log.info(f"  {'Threshold':<30} {args.threshold:>18.2f} {args.threshold:>16.2f}")
        log.info(f"  {'Active Filters':<30} {'Vote only':>18} {'Vote + Power 3':>16}")
        log.info(f"  {'Total Trades':<30} {primary_n:>18} {rt_total:>16}")
        log.info(f"  {'Win Rate':<30} {primary_wr:>17.1f}% {rt_wr:>15.1f}%")
        log.info(f"  {'Net P&L':<30} ${primary_pnl:>+17,.0f} ${rt_pnl:>+15,.0f}")
        log.info(f"  {'Max Drawdown':<30} ${max_dd:>17,.0f}    (retest)")
        log.info("")

        # ── Equity Curve ─────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#1a1f2e")

        for ax, tdf, title, color in [
            (axes[0], trades_df,  f"Original  (vote > {args.threshold}, 6/4 ATR)", "#2196F3"),
            (axes[1], retest_df, f"Re-Test  (Power 3 filter, 2/2 ATR)",             "#4CAF50"),
        ]:
            ax.set_facecolor("#1a1f2e")
            cum_pnl = tdf["net_pnl"].cumsum().to_numpy()
            ax.plot(cum_pnl, color=color, linewidth=1.4)
            ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
            ax.fill_between(range(len(cum_pnl)), 0, cum_pnl,
                            where=cum_pnl >= 0, alpha=0.15, color="#4CAF50")
            ax.fill_between(range(len(cum_pnl)), 0, cum_pnl,
                            where=cum_pnl < 0,  alpha=0.15, color="#F44336")
            n_t = len(tdf)
            wr  = tdf["won"].mean() * 100
            pnl = tdf["net_pnl"].sum()
            ax.set_title(f"{title}\n"
                         f"Trades: {n_t}  |  Win: {wr:.1f}%  |  P&L: ${pnl:+,.0f}",
                         color="white", fontsize=10)
            ax.set_xlabel("Trade #", color="#aaa")
            ax.set_ylabel("Cumulative P&L ($)", color="#aaa")
            ax.tick_params(colors="#aaa")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        plt.suptitle(
            f"Trade DNA Audit  |  Power 3: {', '.join(power3.keys())}",
            color="white", fontsize=12, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        out_path = _MODEL_OUTPUT_DIR / "trade_audit_curves.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1f2e")
        plt.close(fig)
        log.info(f"  Equity curves saved -> {out_path}")

    log.info("")
    _section("Done  [v]  trade_audit.py completed successfully")
    log.info("")


if __name__ == "__main__":
    main()
