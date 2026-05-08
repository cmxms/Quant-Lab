"""
master_tearsheet.py
===================
Combined 24/7 Strategy: Day Shift + Night Shift merged chronologically.

Bot A (Day):   08:00-16:45 ET, 10-min Cooldown, Meta-Veto ON
Bot B (Night): 18:00-08:00 ET, 10-min Cooldown, Meta-Veto ON
Dead Zone:     16:45-18:00 ET (no trading)
"""
import os, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

os.system('')
C_C='\033[96m'; C_G='\033[92m'; C_Y='\033[93m'; C_R='\033[91m'
C_M='\033[95m'; C_W='\033[97m'; C_B='\033[1m'; C_D='\033[2m'; RST='\033[0m'
DIV = "=" * 72; THIN = "-" * 72

_MODEL = Path(r"c:\Users\chadm\OneDrive\Desktop\Quant Lab\Model Training\Outputs")
_FEAT  = Path(r"c:\Users\chadm\OneDrive\Desktop\Quant Lab\Feature Engineering\Outputs")
_OUT   = Path(r"c:\Users\chadm\OneDrive\Desktop\Quant Lab\Model Training")

STARTING_BALANCE = 50_000.0
RISK_PER_TRADE   = 50.0
COOLDOWN_MIN     = 10

DROP_COLS = ['Open','High','Low','Close','Volume','Minutes_From_Open',
             'Day_Of_Week','datetime','ts_event']

def _latest(d, pat):
    m = sorted(d.glob(pat), key=lambda p: p.stat().st_mtime)
    if not m: print(f"  ERROR: no {pat}"); sys.exit(1)
    return m[-1]

def _run_regime(prefix, holdout_pattern, time_filter_fn, label):
    """Run ensemble + veto + cooldown for one regime, return trade DataFrame."""
    mp = f"{prefix}_"
    print(f"\n  {C_C}--- Loading {label} artifacts ---{RST}")

    lgbm   = joblib.load(_latest(_MODEL, f"{mp}lgbm_model_*.joblib"))
    rf     = joblib.load(_latest(_MODEL, f"{mp}rf_model_*.joblib"))
    mlp    = joblib.load(_latest(_MODEL, f"{mp}mlp_model_*.joblib"))
    scaler = joblib.load(_latest(_MODEL, f"{mp}mlp_scaler_*.joblib"))
    meta   = joblib.load(_latest(_MODEL, f"{mp}meta_model_*.joblib"))
    thresh = joblib.load(_latest(_MODEL, f"{mp}optimal_thresholds_*.joblib"))

    lt = thresh["long_threshold"]; st = thresh["short_threshold"]
    LWR = thresh["long_win_r"]; LLR = thresh["long_loss_r"]
    SWR = thresh["short_win_r"]; SLR = thresh["short_loss_r"]

    hp = _latest(_FEAT, holdout_pattern)
    df = pd.read_parquet(hp)
    print(f"    Holdout: {hp.name} ({len(df):,} rows)")

    dt_col = 'ts_event' if 'ts_event' in df.columns else 'datetime'
    timestamps = pd.to_datetime(df[dt_col])

    target = "Target" if "Target" in df.columns else "target"
    y = df[target].to_numpy()
    X = df.drop(columns=[target])
    X = X.drop(columns=[c for c in DROP_COLS if c in X.columns])
    X = X.select_dtypes(include=[np.number])

    # Ensemble
    Xs = scaler.transform(X)
    ep = (lgbm.predict_proba(X) + rf.predict_proba(X) + mlp.predict_proba(Xs)) / 3.0
    ref = lgbm.classes_
    li = int(np.where(ref == 1)[0][0])
    si = int(np.where(ref == -1)[0][0])
    ci = int(np.where(ref == 0)[0][0])

    lp = ep[:, li]; sp = ep[:, si]
    lm = lp >= lt; sm = sp >= st
    both = lm & sm
    ls = lm.copy(); ss = sm.copy()
    ls[both] = lp[both] >= sp[both]
    ss[both] = sp[both] > lp[both]
    triggered = ls | ss

    direction = np.zeros(len(y), dtype=int)
    direction[ls] = 1; direction[ss] = -1
    t_idx = np.where(triggered)[0]

    # Meta veto
    Xt = X.iloc[t_idx].copy()
    Xt['ens_long_proba']  = ep[t_idx, li]
    Xt['ens_short_proba'] = ep[t_idx, si]
    Xt['ens_chop_proba']  = ep[t_idx, ci]
    Xt['ens_direction']   = direction[t_idx]
    approved = meta.predict(Xt) == 1
    approved_idx = t_idx[approved]

    # Build trade frame
    trade_times = timestamps.iloc[approved_idx].values
    trade_dirs  = direction[approved_idx]
    trade_y     = y[approved_idx]

    # Time filter (e.g. Night Shift needs 18:00-08:00 only)
    if time_filter_fn is not None:
        ts_series = pd.Series(trade_times)
        hours = pd.to_datetime(ts_series).dt.hour
        minutes = pd.to_datetime(ts_series).dt.minute
        time_min = hours * 60 + minutes
        keep_mask = time_filter_fn(time_min).values
        trade_times = trade_times[keep_mask]
        trade_dirs = trade_dirs[keep_mask]
        trade_y = trade_y[keep_mask]

    # 10-min cooldown
    cd_ns = np.timedelta64(COOLDOWN_MIN, 'm')
    keep = []
    last_t = None
    for i in range(len(trade_times)):
        if last_t is not None and (trade_times[i] - last_t) < cd_ns:
            continue
        keep.append(i)
        last_t = trade_times[i]

    trade_times = trade_times[keep]
    trade_dirs = trade_dirs[keep]
    trade_y = trade_y[keep]

    # PnL
    pnl = []
    for d, yv in zip(trade_dirs, trade_y):
        if d == 1 and yv == 1:     pnl.append(LWR * RISK_PER_TRADE)
        elif d == 1:               pnl.append(LLR * RISK_PER_TRADE)
        elif d == -1 and yv == -1: pnl.append(SWR * RISK_PER_TRADE)
        else:                      pnl.append(SLR * RISK_PER_TRADE)

    trades_df = pd.DataFrame({
        'timestamp': trade_times,
        'direction': trade_dirs,
        'actual': trade_y,
        'pnl': pnl,
        'regime': label
    })

    wins = (trades_df['pnl'] > 0).sum()
    print(f"    Post-Veto+Cooldown: {len(trades_df):,} trades  |  WR: {wins/len(trades_df)*100:.1f}%  |  Net: ${trades_df['pnl'].sum():,.2f}")
    return trades_df


def main():
    print(f"\n  {C_B}{C_M}{DIV}{RST}")
    print(f"  {C_B}{C_M}   QUANT LAB 24/7 MASTER TEARSHEET{RST}")
    print(f"  {C_B}{C_M}{DIV}{RST}")
    print(f"  {C_D}  Bot A: Day  08:00-16:45 ET | Bot B: Night 18:00-08:00 ET{RST}")
    print(f"  {C_D}  Cooldown: {COOLDOWN_MIN}min | Meta-Veto: ON | Risk: ${RISK_PER_TRADE}/trade{RST}")

    # Day Shift: data already filtered to 08:00-16:45, no extra time filter needed
    day_trades = _run_regime("day_shift", "*day_shift_clean_holdout*", None, "Day")

    # Night Shift: data is 16:45-08:00, but we only want 18:00-08:00
    # time_min < 480 (before 08:00) OR time_min >= 1080 (after 18:00)
    night_filter = lambda tm: (tm < 480) | (tm >= 1080)
    night_trades = _run_regime("night_shift", "*night_shift_clean_holdout*", night_filter, "Night")

    # Merge chronologically
    print(f"\n  {C_C}--- Merging into unified timeline ---{RST}")
    all_trades = pd.concat([day_trades, night_trades], ignore_index=True)
    all_trades.sort_values('timestamp', inplace=True)
    all_trades.reset_index(drop=True, inplace=True)

    total = len(all_trades)
    day_count = (all_trades['regime'] == 'Day').sum()
    night_count = (all_trades['regime'] == 'Night').sum()
    print(f"    Total trades: {total:,}  (Day: {day_count:,}  |  Night: {night_count:,})")

    # Combined equity
    pnl_arr = all_trades['pnl'].to_numpy()
    equity = STARTING_BALANCE + np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak

    net_profit = equity[-1] - STARTING_BALANCE
    max_dd = drawdown.min()
    wins = (pnl_arr > 0).sum()
    wr = wins / total * 100
    gp = pnl_arr[pnl_arr > 0].sum()
    gl = abs(pnl_arr[pnl_arr < 0].sum())
    pf = gp / gl if gl > 0 else float('inf')
    r2d = abs(net_profit / max_dd) if max_dd != 0 else float('inf')
    total_r = net_profit / RISK_PER_TRADE

    # Max consecutive losses
    mcl = 0; s = 0
    for p in pnl_arr:
        if p < 0: s += 1; mcl = max(mcl, s)
        else: s = 0

    # Longest drawdown period (in days)
    in_dd = drawdown < 0
    dd_start = None
    longest_dd_days = 0
    ts_arr = all_trades['timestamp'].values
    for i in range(len(in_dd)):
        if in_dd[i]:
            if dd_start is None:
                dd_start = ts_arr[i]
        else:
            if dd_start is not None:
                dd_dur = (ts_arr[i] - dd_start) / np.timedelta64(1, 'D')
                longest_dd_days = max(longest_dd_days, dd_dur)
                dd_start = None
    # Check if still in drawdown at end
    if dd_start is not None:
        dd_dur = (ts_arr[-1] - dd_start) / np.timedelta64(1, 'D')
        longest_dd_days = max(longest_dd_days, dd_dur)

    # Per-regime stats
    def _regime_stats(df):
        p = df['pnl'].to_numpy()
        w = (p > 0).sum()
        return {"trades": len(df), "wins": w, "wr": w/len(df)*100 if len(df)>0 else 0,
                "net": p.sum(), "gp": p[p>0].sum(), "gl": abs(p[p<0].sum())}

    ds = _regime_stats(all_trades[all_trades['regime']=='Day'])
    ns = _regime_stats(all_trades[all_trades['regime']=='Night'])

    # Print tearsheet
    print(f"\n  {C_B}{C_M}{DIV}{RST}")
    print(f"  {C_B}{C_M}     QUANT LAB 24/7 -- COMBINED MASTER TEARSHEET{RST}")
    print(f"  {C_B}{C_M}{DIV}{RST}")
    print(f"  {THIN}")
    print(f"  {'Metric':<32} {'Day':>12} {'Night':>12} {'COMBINED':>12}")
    print(f"  {THIN}")
    print(f"  {'Trades':<32} {C_W}{ds['trades']:>12,}{RST} {C_W}{ns['trades']:>12,}{RST} {C_B}{C_G}{total:>12,}{RST}")
    print(f"  {'Wins':<32} {C_W}{ds['wins']:>12,}{RST} {C_W}{ns['wins']:>12,}{RST} {C_B}{C_G}{wins:>12,}{RST}")
    print(f"  {'Win Rate':<32} {C_W}{ds['wr']:>11.2f}%{RST} {C_W}{ns['wr']:>11.2f}%{RST} {C_B}{C_G}{wr:>11.2f}%{RST}")
    print(f"  {THIN}")

    nc = C_G if net_profit >= 0 else C_R
    dnc = C_G if ds['net'] >= 0 else C_R
    nnc = C_G if ns['net'] >= 0 else C_R
    print(f"  {'Net Profit ($)':<32} {dnc}${ds['net']:>11,.2f}{RST} {nnc}${ns['net']:>11,.2f}{RST} {C_B}{nc}${net_profit:>11,.2f}{RST}")
    print(f"  {'Final Balance ($)':<32} {'':<12} {'':<12} {C_B}{nc}${equity[-1]:>11,.2f}{RST}")
    print(f"  {THIN}")
    print(f"  {'Max Drawdown ($)':<32} {'':<12} {'':<12} {C_R}${max_dd:>11,.2f}{RST}")
    pfc = C_G if pf > 1 else C_R
    print(f"  {'Profit Factor':<32} {'':<12} {'':<12} {pfc}{pf:>13.2f}{RST}")
    print(f"  {'Max Consec. Losses':<32} {'':<12} {'':<12} {C_R}{mcl:>13}{RST}")
    print(f"  {THIN}")
    r2dc = C_G if r2d > 1 else C_R
    print(f"  {'Reward / Drawdown Ratio':<32} {'':<12} {'':<12} {C_B}{r2dc}{r2d:>12.2f}x{RST}")
    print(f"  {'Total R':<32} {'':<12} {'':<12} {C_B}{nc}{total_r:>12.1f}R{RST}")
    print(f"  {'Longest DD Period (days)':<32} {'':<12} {'':<12} {C_Y}{longest_dd_days:>12.1f}{RST}")
    print(f"  {C_B}{C_M}{DIV}{RST}")

    # Final verdict
    print("")
    if net_profit > 0 and pf > 1:
        print(f"  {C_B}{C_G}  >> QUANT LAB 24/7 IS PROFITABLE ON UNSEEN DATA <<{RST}")
    else:
        print(f"  {C_R}  >> NEGATIVE EXPECTANCY <<{RST}")
    print("")

    # ── Equity Curve Plot ─────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                    gridspec_kw={'hspace': 0.08})
    fig.patch.set_facecolor('#0d1117')

    trade_nums = np.arange(1, total + 1)

    # Top: equity curve with regime shading
    ax1.set_facecolor('#0d1117')
    ax1.plot(trade_nums, equity, color='#58a6ff', linewidth=1.2, zorder=3)
    ax1.fill_between(trade_nums, STARTING_BALANCE, equity,
                     where=equity >= STARTING_BALANCE, alpha=0.15, color='#3fb950', zorder=2)
    ax1.fill_between(trade_nums, STARTING_BALANCE, equity,
                     where=equity < STARTING_BALANCE, alpha=0.15, color='#f85149', zorder=2)

    # Shade day vs night trades
    regimes = all_trades['regime'].values
    for i in range(len(trade_nums)):
        if regimes[i] == 'Day':
            ax1.axvspan(trade_nums[i]-0.5, trade_nums[i]+0.5, alpha=0.03, color='#e3b341', zorder=1)

    ax1.axhline(STARTING_BALANCE, color='#484f58', linewidth=0.8, linestyle='--', alpha=0.7)
    ax1.set_ylabel('Equity ($)', color='#c9d1d9', fontsize=11)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.tick_params(colors='#8b949e')
    ax1.set_xlim(1, total)
    ax1.set_title(f'Quant Lab 24/7 Master Equity  |  Net: ${net_profit:,.0f}  |  DD: ${max_dd:,.0f}  |  R/DD: {r2d:.1f}x',
                  color='#c9d1d9', fontsize=13, fontweight='bold', pad=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_color('#30363d')
    ax1.spines['left'].set_color('#30363d')
    ax1.tick_params(labelbottom=False)

    # Bottom: drawdown
    ax2.set_facecolor('#0d1117')
    ax2.fill_between(trade_nums, drawdown, 0, color='#f85149', alpha=0.4)
    ax2.plot(trade_nums, drawdown, color='#f85149', linewidth=0.8)
    ax2.set_ylabel('Drawdown ($)', color='#c9d1d9', fontsize=11)
    ax2.set_xlabel('Trade #', color='#c9d1d9', fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax2.tick_params(colors='#8b949e')
    ax2.set_xlim(1, total)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color('#30363d')
    ax2.spines['left'].set_color('#30363d')

    out_path = _OUT / "quant_lab_master_equity.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  {C_G}Equity curve saved: {out_path.name}{RST}\n")


if __name__ == "__main__":
    main()
