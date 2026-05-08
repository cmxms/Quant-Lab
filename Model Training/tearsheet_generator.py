"""
tearsheet_generator.py
======================
Phase 4: Professional Financial Tearsheet

Reconstructs the Pre-Veto Council trades from the holdout backtest,
translates R-multiples into dollar P&L, and generates a full equity
curve with drawdown visualization.

Pipeline Stage : Walk-Forward Testing (Phase 4)
Inputs         : nq_final_clean_holdout.parquet + all Phase 3 artifacts
Outputs        : Outputs/holdout_equity_curve.png
                 Console tearsheet

Usage
-----
    python tearsheet_generator.py
"""

import os, sys, logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

os.system('')

# ── ANSI ────────────────────────────────────────────────────────────────────
class C:
    CYAN='\033[96m'; GREEN='\033[92m'; YELLOW='\033[93m'; RED='\033[91m'
    MAGENTA='\033[95m'; WHITE='\033[97m'; BOLD='\033[1m'; DIM='\033[2m'
    RESET='\033[0m'

# ── Config ──────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).parent.resolve()
_FEATURE_DIR = _HERE.parent / "Feature Engineering" / "Outputs"
_MODEL_DIR   = _HERE / "Outputs"

TARGET_COL = "Target"
DROP_COLS  = ["Open","High","Low","Close","Volume",
              "Minutes_From_Open","Day_Of_Week","datetime","ts_event"]

STARTING_BALANCE = 50_000.00
RISK_PER_TRADE   = 50.00

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)
DIV  = "=" * 62
THIN = "-" * 58

def _latest(directory, pattern):
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        log.error(f"{C.RED}No file matching '{pattern}' in {directory}{C.RESET}")
        sys.exit(1)
    return matches[-1]


def main():
    log.info("")
    log.info(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}   TEARSHEET GENERATOR  (Holdout Backtest){C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")

    # ── Load Artifacts ───────────────────────────────────────────────────────
    log.info(f"\n  {C.CYAN}Loading Phase 3 artifacts...{C.RESET}")
    lgbm_model = joblib.load(_latest(_MODEL_DIR, "lgbm_model_*.joblib"))
    rf_model   = joblib.load(_latest(_MODEL_DIR, "rf_model_*.joblib"))
    mlp_model  = joblib.load(_latest(_MODEL_DIR, "mlp_model_*.joblib"))
    mlp_scaler = joblib.load(_latest(_MODEL_DIR, "mlp_scaler_*.joblib"))
    thresholds = joblib.load(_latest(_MODEL_DIR, "optimal_thresholds_*.joblib"))
    log.info(f"    All models & thresholds loaded.")

    # ── Load Holdout ─────────────────────────────────────────────────────────
    log.info(f"\n  {C.CYAN}Loading holdout data...{C.RESET}")
    holdout_path = _latest(_FEATURE_DIR, "*holdout*")
    df = pd.read_parquet(holdout_path)

    target = TARGET_COL if TARGET_COL in df.columns else TARGET_COL.lower()
    y_holdout = df[target].to_numpy()

    # Extract timestamps BEFORE dropping columns
    dt_col = None
    for c in ["ts_event", "datetime"]:
        if c in df.columns:
            dt_col = c
            break
    if dt_col:
        timestamps = pd.to_datetime(df[dt_col], utc=True).dt.tz_convert("US/Eastern")
    else:
        timestamps = None
        log.info(f"    {C.YELLOW}Warning: No datetime column found. RTH filter disabled.{C.RESET}")

    X_holdout = df.drop(columns=[target])
    to_drop = [c for c in DROP_COLS if c in X_holdout.columns]
    if to_drop:
        X_holdout = X_holdout.drop(columns=to_drop)
    non_num = X_holdout.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        X_holdout = X_holdout.drop(columns=non_num)

    log.info(f"    Holdout: {len(X_holdout):,} bars  |  Features: {X_holdout.shape[1]}")

    # R-Multiple parameters
    LONG_WIN_R   = thresholds["long_win_r"]
    LONG_LOSS_R  = thresholds["long_loss_r"]
    SHORT_WIN_R  = thresholds["short_win_r"]
    SHORT_LOSS_R = thresholds["short_loss_r"]
    lt = thresholds["long_threshold"]
    st = thresholds["short_threshold"]

    # ── Reconstruct Council Signals ──────────────────────────────────────────
    log.info(f"\n  {C.CYAN}Reconstructing Pre-Veto Council trades...{C.RESET}")
    X_scaled = mlp_scaler.transform(X_holdout)
    proba_lgbm = lgbm_model.predict_proba(X_holdout)
    proba_rf   = rf_model.predict_proba(X_holdout)
    proba_mlp  = mlp_model.predict_proba(X_scaled)
    ensemble_proba = (proba_lgbm + proba_rf + proba_mlp) / 3.0

    ref_classes = lgbm_model.classes_
    long_idx  = int(np.where(ref_classes == 1)[0][0])
    short_idx = int(np.where(ref_classes == -1)[0][0])

    long_probs  = ensemble_proba[:, long_idx]
    short_probs = ensemble_proba[:, short_idx]

    long_mask  = long_probs >= lt
    short_mask = short_probs >= st
    both = long_mask & short_mask
    long_signal  = long_mask.copy()
    short_signal = short_mask.copy()
    long_signal[both]  = long_probs[both] >= short_probs[both]
    short_signal[both] = short_probs[both] > long_probs[both]
    triggered = long_signal | short_signal

    direction = np.zeros(len(y_holdout), dtype=int)
    direction[long_signal]  = 1
    direction[short_signal] = -1

    t_idx = np.where(triggered)[0]
    raw_signals = len(t_idx)
    log.info(f"    Raw signals before filters: {C.YELLOW}{raw_signals:,}{C.RESET}")

    # ── Market Microstructure Filters ────────────────────────────────────────
    log.info(f"\n  {C.CYAN}Applying Cooldown Filter (Night Shift Mode — No RTH Gate)...{C.RESET}")

    COOLDOWN_MINUTES = 15

    cooldown_filtered = 0
    last_trade_time = None

    trades = []
    for i in t_idx:
        d = direction[i]
        y = y_holdout[i]

        # ── Cooldown Timer (15 min between trades) ───────────────────────────
        if timestamps is not None:
            ts = timestamps.iloc[i]
            if last_trade_time is not None:
                delta = (ts - last_trade_time).total_seconds() / 60.0
                if delta < COOLDOWN_MINUTES:
                    cooldown_filtered += 1
                    continue
            last_trade_time = ts

        # ── Build trade entry ────────────────────────────────────────────────
        if d == 1 and y == 1:
            r_mult = LONG_WIN_R
            outcome = "Long Win"
        elif d == 1:
            r_mult = LONG_LOSS_R
            outcome = "Long Loss"
        elif d == -1 and y == -1:
            r_mult = SHORT_WIN_R
            outcome = "Short Win"
        else:
            r_mult = SHORT_LOSS_R
            outcome = "Short Loss"

        dollar_pnl = r_mult * RISK_PER_TRADE
        trades.append({
            "bar_index": i,
            "timestamp": timestamps.iloc[i] if timestamps is not None else None,
            "direction": "LONG" if d == 1 else "SHORT",
            "outcome": outcome,
            "r_multiple": r_mult,
            "dollar_pnl": dollar_pnl,
        })

    log.info(f"    {C.RED}Cooldown Filter removed:  {cooldown_filtered:,} signals{C.RESET}")
    log.info(f"    {C.GREEN}Trades surviving:         {len(trades):,}{C.RESET}")

    trade_df = pd.DataFrame(trades)
    trade_df["cumulative_equity"] = STARTING_BALANCE + trade_df["dollar_pnl"].cumsum()

    # ── Metric Calculation ───────────────────────────────────────────────────
    log.info(f"\n  {C.CYAN}Calculating performance metrics...{C.RESET}")

    final_balance = trade_df["cumulative_equity"].iloc[-1]
    net_profit    = final_balance - STARTING_BALANCE
    total_trades  = len(trade_df)

    gross_profit = trade_df.loc[trade_df["dollar_pnl"] > 0, "dollar_pnl"].sum()
    gross_loss   = abs(trade_df.loc[trade_df["dollar_pnl"] < 0, "dollar_pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    wins   = (trade_df["dollar_pnl"] > 0).sum()
    losses = (trade_df["dollar_pnl"] < 0).sum()
    win_rate = wins / total_trades * 100

    avg_win  = trade_df.loc[trade_df["dollar_pnl"] > 0, "dollar_pnl"].mean()
    avg_loss = trade_df.loc[trade_df["dollar_pnl"] < 0, "dollar_pnl"].mean()

    # Max Drawdown
    equity = trade_df["cumulative_equity"].values
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_dd = drawdown.min()
    max_dd_end = np.argmin(drawdown)
    max_dd_start = np.argmax(equity[:max_dd_end + 1]) if max_dd_end > 0 else 0

    # Max Consecutive Losses
    pnl_signs = (trade_df["dollar_pnl"] < 0).astype(int).values
    max_consec_loss = 0
    current_streak = 0
    for s in pnl_signs:
        if s == 1:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    # Max Consecutive Wins
    win_signs = (trade_df["dollar_pnl"] > 0).astype(int).values
    max_consec_win = 0
    current_streak = 0
    for s in win_signs:
        if s == 1:
            current_streak += 1
            max_consec_win = max(max_consec_win, current_streak)
        else:
            current_streak = 0

    total_r = trade_df["r_multiple"].sum()
    avg_r   = total_r / total_trades

    # ── Console Tearsheet ────────────────────────────────────────────────────
    log.info("")
    log.info(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}   HOLDOUT TEARSHEET  (Night Shift + 15m CD){C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")
    log.info(f"  {C.DIM}  Risk: ${RISK_PER_TRADE:.2f}/trade  |  Start: ${STARTING_BALANCE:,.2f}  |  24/7 Trading  |  CD: {COOLDOWN_MINUTES}min{C.RESET}")
    log.info(f"  {THIN}")

    log.info(f"  {C.WHITE}{'Total Trades':<30}{C.YELLOW}{total_trades:>16,}{C.RESET}")
    log.info(f"  {C.WHITE}{'Wins / Losses':<30}{C.GREEN}{wins}{C.RESET} / {C.RED}{losses}{C.RESET}")
    log.info(f"  {C.WHITE}{'Win Rate':<30}{win_rate:>15.2f}%{C.RESET}")
    log.info(f"  {THIN}")

    net_color = C.GREEN if net_profit >= 0 else C.RED
    log.info(f"  {C.WHITE}{'Starting Balance':<30}${STARTING_BALANCE:>14,.2f}{C.RESET}")
    log.info(f"  {C.WHITE}{'Final Balance':<30}{C.BOLD}{net_color}${final_balance:>14,.2f}{C.RESET}")
    log.info(f"  {C.WHITE}{'Net Profit':<30}{net_color}${net_profit:>14,.2f}{C.RESET}")
    log.info(f"  {C.WHITE}{'Return on Account':<30}{net_color}{net_profit/STARTING_BALANCE*100:>14.2f}%{C.RESET}")
    log.info(f"  {THIN}")

    log.info(f"  {C.WHITE}{'Gross Profit':<30}{C.GREEN}${gross_profit:>14,.2f}{C.RESET}")
    log.info(f"  {C.WHITE}{'Gross Loss':<30}{C.RED}${gross_loss:>14,.2f}{C.RESET}")
    pf_color = C.GREEN if profit_factor > 1 else C.RED
    log.info(f"  {C.WHITE}{'Profit Factor':<30}{pf_color}{profit_factor:>16.2f}{C.RESET}")
    log.info(f"  {THIN}")

    log.info(f"  {C.WHITE}{'Avg Winning Trade':<30}{C.GREEN}${avg_win:>14,.2f}{C.RESET}")
    log.info(f"  {C.WHITE}{'Avg Losing Trade':<30}{C.RED}${avg_loss:>14,.2f}{C.RESET}")
    log.info(f"  {C.WHITE}{'Avg R per Trade':<30}{avg_r:>15.2f}R{C.RESET}")
    log.info(f"  {C.WHITE}{'Total R Gained':<30}{C.BOLD}{C.GREEN}{total_r:>15.2f}R{C.RESET}")
    log.info(f"  {THIN}")

    log.info(f"  {C.WHITE}{'Max Drawdown':<30}{C.RED}${max_dd:>14,.2f}{C.RESET}")
    log.info(f"  {C.WHITE}{'Max Consecutive Losses':<30}{C.RED}{max_consec_loss:>16}{C.RESET}")
    log.info(f"  {C.WHITE}{'Max Consecutive Wins':<30}{C.GREEN}{max_consec_win:>16}{C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")

    # ── Equity Curve Chart ───────────────────────────────────────────────────
    log.info(f"\n  {C.CYAN}Generating equity curve...{C.RESET}")

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    trade_nums = np.arange(1, total_trades + 1)

    # Main equity line
    ax.plot(trade_nums, equity, color='#58a6ff', linewidth=1.8, label='Equity', zorder=3)

    # Peak line
    ax.plot(trade_nums, peak, color='#8b949e', linewidth=0.8, linestyle='--',
            alpha=0.6, label='Peak Equity', zorder=2)

    # Drawdown shading (entire drawdown region)
    ax.fill_between(trade_nums, equity, peak, where=(drawdown < 0),
                    color='#f85149', alpha=0.15, zorder=1)

    # Max drawdown region highlight
    if max_dd_end > max_dd_start:
        dd_range = range(max_dd_start, max_dd_end + 1)
        ax.fill_between(
            trade_nums[max_dd_start:max_dd_end+1],
            equity[max_dd_start:max_dd_end+1],
            peak[max_dd_start:max_dd_end+1],
            color='#f85149', alpha=0.40, label=f'Max Drawdown (${max_dd:,.2f})', zorder=2
        )

    # Starting balance line
    ax.axhline(y=STARTING_BALANCE, color='#8b949e', linewidth=0.6, linestyle=':', alpha=0.5)

    # Styling
    ax.set_title('Out-of-Sample Equity Curve  (Night Shift + 15min Cooldown)',
                 fontsize=16, fontweight='bold', color='#e6edf3', pad=15)
    ax.set_xlabel('Trade #', fontsize=12, color='#8b949e')
    ax.set_ylabel('Account Equity ($)', fontsize=12, color='#8b949e')

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.tick_params(colors='#8b949e')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')
    ax.grid(True, alpha=0.1, color='#8b949e')

    legend = ax.legend(loc='upper left', fontsize=10, facecolor='#161b22',
                       edgecolor='#30363d', labelcolor='#e6edf3')

    # Annotation box
    stats_text = (
        f"Net Profit: ${net_profit:,.2f}\n"
        f"Win Rate: {win_rate:.1f}%\n"
        f"Profit Factor: {profit_factor:.2f}\n"
        f"Max DD: ${max_dd:,.2f}\n"
        f"Total R: {total_r:.2f}R"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='#161b22',
                 edgecolor='#30363d', alpha=0.9)
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            color='#e6edf3', bbox=props, family='monospace')

    plt.tight_layout()
    out_path = _MODEL_DIR / "holdout_equity_curve.png"
    plt.savefig(out_path, dpi=200, facecolor='#0d1117', edgecolor='none')
    plt.close()

    log.info(f"    Saved to: {out_path}")
    log.info("")


if __name__ == "__main__":
    main()
