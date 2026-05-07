"""
threshold_analysis.py
=====================
Threshold Sensitivity Analysis for the Weighted Ensemble Vote Signal.

Runs a fast vectorized backtest (no Backtrader overhead) across four
thresholds to identify the optimal trade frequency / expectancy balance.

Simulation Rules (identical to holdout_backtest.py)
----------------------------------------------------
  Entry  : Next bar's Open + 0.25 pt slippage (1 tick)
  Target : Entry + (6.0 x ATR)    [ATR = ATR_Norm x Close]
  Stop   : Entry - (4.0 x ATR)
  EOD    : Flatten at bar's Close if time >= 15:55 EST (no new entries)
  Costs  : $4.00 round-trip commission | NQ multiplier = $20/point
  Logic  : If TP and SL both breached in same bar -> pessimistic (SL wins)

Usage
-----
    python threshold_analysis.py
    python threshold_analysis.py --thresholds 0.40 0.45 0.50 0.55 0.60
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ── Configuration ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_HERE               = Path(__file__).parent.resolve()
_FEATURE_OUTPUT_DIR = _HERE.parent / "Feature Engineering" / "Outputs"
_MODEL_OUTPUT_DIR   = _HERE / "Outputs"

TARGET_COL   = "Target"
DROP_COLS    = ["Open", "High", "Low", "Close", "Volume", "Minutes_From_Open", "Day_Of_Week"]
CANONICAL    = np.array([-1, 0, 1])

NQ_MULT      = 20.0     # $/point
SLIPPAGE_PTS = 0.25     # 1 tick
COMMISSION   = 4.00     # $ round trip
TP_MULT      = 6.0
SL_MULT      = 4.0
EOD_HOUR     = 15
EOD_MINUTE   = 55

W_LGBM   = 0.50
W_RF     = 0.35
W_LOGREG = 0.15

DEFAULT_THRESHOLDS = [0.45, 0.50, 0.55, 0.60]

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
DIVIDER = "-" * 70

def _section(title):
    log.info(DIVIDER)
    log.info(f"  {title}")
    log.info(DIVIDER)

# ---------------------------------------------------------------------------
# ── Helpers ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _latest(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        log.error(f"[MISSING] No '{pattern}' in {directory}")
        sys.exit(1)
    log.info(f"  {label:<12}: {matches[-1].name}")
    return matches[-1]


def _align(proba: np.ndarray, model_classes: np.ndarray) -> np.ndarray:
    out = np.zeros((proba.shape[0], 3), dtype=np.float64)
    for i, cls in enumerate(CANONICAL):
        col = np.where(model_classes == cls)[0]
        if col.size:
            out[:, i] = proba[:, col[0]]
    return out


def _max_drawdown(pnl_series: list) -> float:
    """Compute maximum drawdown from a list of cumulative P&L values."""
    if not pnl_series:
        return 0.0
    cum = np.array(pnl_series)
    running_max = np.maximum.accumulate(cum)
    drawdowns   = running_max - cum
    return float(drawdowns.max())

# ---------------------------------------------------------------------------
# ── Build Signal DataFrame ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def build_signal_df(holdout_path: Path) -> pd.DataFrame:
    _section("Step 1 / 2  -  Computing Weighted Vote on Holdout")

    log.info("  Loading models ...")
    lgbm_model  = joblib.load(_latest(_MODEL_OUTPUT_DIR, "lgbm_model_*.joblib",     "LGBM"))
    rf_model    = joblib.load(_latest(_MODEL_OUTPUT_DIR, "rf_model_*.joblib",       "RF"))
    logreg_pipe = joblib.load(_latest(_MODEL_OUTPUT_DIR, "logreg_pipeline_*.joblib","LogReg"))
    logreg_inner = logreg_pipe.named_steps["model"]

    log.info(f"  Loading holdout: {holdout_path.name}")
    df = pd.read_parquet(holdout_path)
    log.info(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    drop = [c for c in [TARGET_COL] + DROP_COLS if c in df.columns]
    X    = df.drop(columns=drop)

    log.info("  Running predict_proba() ...")
    lgbm_p1   = _align(lgbm_model.predict_proba(X),   lgbm_model.classes_)[:, 2]
    rf_p1     = _align(rf_model.predict_proba(X),     rf_model.classes_)[:, 2]
    logreg_p1 = _align(logreg_pipe.predict_proba(X),  logreg_inner.classes_)[:, 2]

    df["weighted_vote"] = lgbm_p1 * W_LGBM + rf_p1 * W_RF + logreg_p1 * W_LOGREG
    df["atr_pts"]       = df["ATR_Norm"] * df["Close"]   # ATR in price points

    # Normalize index to naive Eastern time for hour/minute checks
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_convert("US/Eastern").tz_localize(None)

    log.info(f"  Weighted Vote range: [{df['weighted_vote'].min():.4f}, {df['weighted_vote'].max():.4f}]")
    log.info(f"  Holdout spans: {df.index[0].date()}  to  {df.index[-1].date()}")
    return df

# ---------------------------------------------------------------------------
# ── Vectorized Simulation for One Threshold ──────────────────────────────────
# ---------------------------------------------------------------------------

def simulate(df: pd.DataFrame, threshold: float) -> dict:
    """
    Fast bar-by-bar simulation matching holdout_backtest.py rules exactly.

    Returns a dict of performance metrics.
    """
    opens   = df["Open"].to_numpy(dtype=np.float64)
    highs   = df["High"].to_numpy(dtype=np.float64)
    lows    = df["Low"].to_numpy(dtype=np.float64)
    closes  = df["Close"].to_numpy(dtype=np.float64)
    votes   = df["weighted_vote"].to_numpy(dtype=np.float64)
    atrs    = df["atr_pts"].to_numpy(dtype=np.float64)
    hours   = df.index.hour
    minutes = df.index.minute
    n       = len(df)

    trades      = []
    cum_pnl     = []
    running_pnl = 0.0

    i = 0
    while i < n - 1:
        # Skip if EOD
        is_eod = (hours[i] > EOD_HOUR) or (hours[i] == EOD_HOUR and minutes[i] >= EOD_MINUTE)
        if is_eod:
            i += 1
            continue

        # Entry signal check
        if votes[i] <= threshold:
            i += 1
            continue

        # Entry at next bar open + slippage
        entry_bar = i + 1
        if entry_bar >= n:
            break

        entry_px = opens[entry_bar] + SLIPPAGE_PTS
        atr_at_signal = atrs[i]
        tp = entry_px + TP_MULT * atr_at_signal
        sl = entry_px - SL_MULT * atr_at_signal

        # Scan forward for exit
        exit_px   = None
        exit_type = None

        for j in range(entry_bar, n):
            bar_eod = (hours[j] > EOD_HOUR) or (
                hours[j] == EOD_HOUR and minutes[j] >= EOD_MINUTE
            )

            if bar_eod:
                # EOD flatten at close
                exit_px   = closes[j] - SLIPPAGE_PTS   # 1-tick slippage on exit too
                exit_type = "EOD"
                i = j + 1
                break

            hit_tp = highs[j] >= tp
            hit_sl = lows[j]  <= sl

            if hit_tp and hit_sl:
                # Pessimistic: assume SL hit first
                exit_px   = sl
                exit_type = "SL"
                i = j + 1
                break
            elif hit_sl:
                exit_px   = sl
                exit_type = "SL"
                i = j + 1
                break
            elif hit_tp:
                exit_px   = tp
                exit_type = "TP"
                i = j + 1
                break
        else:
            # Reached end of data with open position
            exit_px   = closes[-1]
            exit_type = "EOD"
            i = n

        if exit_px is None:
            i += 1
            continue

        gross_pnl = (exit_px - entry_px) * NQ_MULT
        net_pnl   = gross_pnl - COMMISSION
        running_pnl += net_pnl
        cum_pnl.append(running_pnl)
        trades.append({
            "net_pnl":   net_pnl,
            "exit_type": exit_type,
            "won":       net_pnl > 0,
        })

    total_trades = len(trades)
    won_trades   = sum(1 for t in trades if t["won"])
    win_rate     = won_trades / total_trades * 100 if total_trades else 0.0
    total_profit = sum(t["net_pnl"] for t in trades)
    max_dd       = _max_drawdown(cum_pnl)

    # Trade frequency per year (holdout is ~1 year)
    holdout_days = (df.index[-1] - df.index[0]).days
    trades_per_year = total_trades / (holdout_days / 365.25) if holdout_days else 0

    return {
        "threshold":       threshold,
        "total_trades":    total_trades,
        "trades_per_yr":   trades_per_year,
        "win_rate":        win_rate,
        "total_profit":    total_profit,
        "max_drawdown":    max_dd,
        "tp_exits":        sum(1 for t in trades if t["exit_type"] == "TP"),
        "sl_exits":        sum(1 for t in trades if t["exit_type"] == "SL"),
        "eod_exits":       sum(1 for t in trades if t["exit_type"] == "EOD"),
        "avg_pnl":         total_profit / total_trades if total_trades else 0.0,
    }

# ---------------------------------------------------------------------------
# ── Main ────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Threshold sensitivity analysis for the weighted ensemble backtest."
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="List of vote thresholds to test (default: 0.45 0.50 0.55 0.60)",
    )
    parser.add_argument(
        "--holdout_path", type=Path, default=None,
    )
    args = parser.parse_args()

    holdout_path = args.holdout_path or (_FEATURE_OUTPUT_DIR / "long_clean_holdout.parquet")
    if not holdout_path.exists():
        log.error(f"[MISSING] Holdout file not found: {holdout_path}")
        sys.exit(1)

    log.info("")
    _section("Threshold Sensitivity Analysis  -  Quantitative Trading Pipeline")
    log.info(f"  Holdout file : {holdout_path.name}")
    log.info(f"  Thresholds   : {args.thresholds}")
    log.info(f"  Rules        : Entry @ next-bar Open + {SLIPPAGE_PTS}pt slippage")
    log.info(f"               : TP = +{TP_MULT}x ATR  |  SL = -{SL_MULT}x ATR")
    log.info(f"               : EOD flatten @ {EOD_HOUR}:{EOD_MINUTE:02d}  |  Comm = ${COMMISSION:.2f} RT")
    log.info("")

    df = build_signal_df(holdout_path)

    _section("Step 2 / 2  -  Running Simulations")
    results = []
    for thresh in sorted(args.thresholds):
        log.info(f"  Simulating threshold = {thresh} ...")
        r = simulate(df, thresh)
        results.append(r)
        log.info(
            f"    Trades: {r['total_trades']:>4}  |  "
            f"Win%: {r['win_rate']:>5.1f}  |  "
            f"P&L: ${r['total_profit']:>+10,.0f}  |  "
            f"MaxDD: ${r['max_drawdown']:>8,.0f}"
        )

    # ── Summary Table ─────────────────────────────────────────────────────────
    _section("Sensitivity Analysis Results")

    H = f"{'Threshold':>10} {'Trades':>7} {'Trades/Yr':>10} {'Win%':>7} {'TP':>5} {'SL':>5} {'EOD':>5} {'Net P&L':>12} {'Avg/Trade':>11} {'Max DD':>11}"
    log.info("  " + H)
    log.info("  " + "-" * len(H))

    for r in results:
        flag = ""
        if 50 <= r["trades_per_yr"] <= 200:
            flag = "  <-- TARGET RANGE"
        log.info(
            f"  {r['threshold']:>10.2f}"
            f" {r['total_trades']:>7}"
            f" {r['trades_per_yr']:>10.1f}"
            f" {r['win_rate']:>7.1f}%"
            f" {r['tp_exits']:>5}"
            f" {r['sl_exits']:>5}"
            f" {r['eod_exits']:>5}"
            f" ${r['total_profit']:>+11,.0f}"
            f" ${r['avg_pnl']:>+10,.0f}"
            f" ${r['max_drawdown']:>10,.0f}"
            f"{flag}"
        )

    # ── Recommendation ────────────────────────────────────────────────────────
    _section("Recommendation")

    # Find thresholds in 50-200 trades/yr band with positive expectancy
    candidates = [r for r in results if 50 <= r["trades_per_yr"] <= 200 and r["avg_pnl"] > 0]
    positive   = [r for r in results if r["avg_pnl"] > 0]

    if candidates:
        # Best = highest avg P&L per trade within target range
        best = max(candidates, key=lambda r: r["avg_pnl"])
        log.info(
            f"  OPTIMAL THRESHOLD: {best['threshold']}\n"
            f"\n"
            f"    Trades/year  : {best['trades_per_yr']:.0f}  (within 50-200 target band)\n"
            f"    Win Rate     : {best['win_rate']:.1f}%\n"
            f"    Net Profit   : ${best['total_profit']:+,.0f}\n"
            f"    Avg per Trade: ${best['avg_pnl']:+,.0f}\n"
            f"    Max Drawdown : ${best['max_drawdown']:,.0f}\n"
        )
    elif positive:
        best = max(positive, key=lambda r: r["total_profit"])
        log.info(
            f"  No threshold lands in the 50-200 trades/yr band with positive expectancy.\n"
            f"  Best positive-expectancy threshold: {best['threshold']}\n"
            f"    Trades/year  : {best['trades_per_yr']:.0f}\n"
            f"    Net Profit   : ${best['total_profit']:+,.0f}\n"
            f"    Avg per Trade: ${best['avg_pnl']:+,.0f}\n"
        )
    else:
        log.info(
            "  No threshold produced positive average P&L per trade.\n"
            "  Recommendation: revisit label quality (lookahead window, ATR multipliers)\n"
            "  or re-examine the feature set before live deployment.\n"
        )

    log.info(DIVIDER)
    log.info("")
    _section("Done  [v]  threshold_analysis.py completed successfully")
    log.info("")


if __name__ == "__main__":
    main()
