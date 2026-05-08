"""
vix_trade_analysis.py
=====================
Merges combined 24/7 trade log with daily VIX data.
Analyzes win rate, avg R, and trade count across VIX regimes.
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
_ROOT  = Path(r"c:\Users\chadm\OneDrive\Desktop\Quant Lab")

STARTING_BALANCE = 50_000.0
RISK_PER_TRADE   = 50.0
COOLDOWN_MIN     = 10

DROP_COLS = ['Open','High','Low','Close','Volume','Minutes_From_Open',
             'Day_Of_Week','datetime','ts_event']

def _latest(d, pat):
    m = sorted(d.glob(pat), key=lambda p: p.stat().st_mtime)
    if not m: print(f"  ERROR: no {pat}"); sys.exit(1)
    return m[-1]

def _run_regime(prefix, holdout_pattern, time_filter_fn):
    mp = f"{prefix}_"
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

    dt_col = 'ts_event' if 'ts_event' in df.columns else 'datetime'
    timestamps = pd.to_datetime(df[dt_col])

    target = "Target" if "Target" in df.columns else "target"
    y = df[target].to_numpy()
    X = df.drop(columns=[target])
    X = X.drop(columns=[c for c in DROP_COLS if c in X.columns])
    X = X.select_dtypes(include=[np.number])

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

    direction = np.zeros(len(y), dtype=int)
    direction[ls] = 1; direction[ss] = -1
    t_idx = np.where(ls | ss)[0]

    Xt = X.iloc[t_idx].copy()
    Xt['ens_long_proba'] = ep[t_idx, li]
    Xt['ens_short_proba'] = ep[t_idx, si]
    Xt['ens_chop_proba'] = ep[t_idx, ci]
    Xt['ens_direction'] = direction[t_idx]
    approved = meta.predict(Xt) == 1
    approved_idx = t_idx[approved]

    trade_times = timestamps.iloc[approved_idx].values
    trade_dirs = direction[approved_idx]
    trade_y = y[approved_idx]

    if time_filter_fn is not None:
        ts_s = pd.Series(trade_times)
        tm = pd.to_datetime(ts_s).dt.hour * 60 + pd.to_datetime(ts_s).dt.minute
        mask = time_filter_fn(tm).values
        trade_times = trade_times[mask]
        trade_dirs = trade_dirs[mask]
        trade_y = trade_y[mask]

    cd_ns = np.timedelta64(COOLDOWN_MIN, 'm')
    keep = []; last_t = None
    for i in range(len(trade_times)):
        if last_t is not None and (trade_times[i] - last_t) < cd_ns: continue
        keep.append(i); last_t = trade_times[i]

    trade_times = trade_times[keep]
    trade_dirs = trade_dirs[keep]
    trade_y = trade_y[keep]

    pnl = []
    r_mult = []
    for d, yv in zip(trade_dirs, trade_y):
        if d == 1 and yv == 1:     pnl.append(LWR * RISK_PER_TRADE); r_mult.append(LWR)
        elif d == 1:               pnl.append(LLR * RISK_PER_TRADE); r_mult.append(LLR)
        elif d == -1 and yv == -1: pnl.append(SWR * RISK_PER_TRADE); r_mult.append(SWR)
        else:                      pnl.append(SLR * RISK_PER_TRADE); r_mult.append(SLR)

    return pd.DataFrame({
        'timestamp': trade_times, 'direction': trade_dirs,
        'actual': trade_y, 'pnl': pnl, 'r_multiple': r_mult,
        'win': [1 if p > 0 else 0 for p in pnl]
    })


def main():
    print(f"\n  {C_B}{C_M}{DIV}{RST}")
    print(f"  {C_B}{C_M}   VIX REGIME ANALYSIS  (Trade-Level){RST}")
    print(f"  {C_B}{C_M}{DIV}{RST}")

    # Load VIX
    vix_path = _ROOT / "TVC_VIX, 1D.csv"
    vix = pd.read_csv(vix_path)
    vix['date'] = pd.to_datetime(vix['time']).dt.date
    vix.rename(columns={'close': 'vix_close', 'high': 'vix_high', 'low': 'vix_low'}, inplace=True)
    print(f"  {C_C}VIX data: {len(vix):,} daily bars ({vix['date'].min()} to {vix['date'].max()}){RST}")

    # Build combined trades
    print(f"  {C_C}Loading Day + Night regimes...{RST}")
    day = _run_regime("day_shift", "*day_shift_clean_holdout*", None)
    night = _run_regime("night_shift", "*night_shift_clean_holdout*", lambda tm: (tm < 480) | (tm >= 1080))

    all_trades = pd.concat([day, night], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    all_trades['date'] = pd.to_datetime(all_trades['timestamp']).dt.date
    print(f"  {C_C}Total trades: {len(all_trades):,}{RST}")

    # Merge with VIX (each trade gets the VIX close for that day)
    merged = all_trades.merge(vix[['date', 'vix_close', 'vix_high']], on='date', how='left')
    matched = merged['vix_close'].notna().sum()
    print(f"  {C_C}Matched to VIX: {matched:,} / {len(merged):,} trades{RST}")
    merged.dropna(subset=['vix_close'], inplace=True)

    # ── VIX Bucket Analysis ──────────────────────────────────────────────────
    bins = [0, 15, 20, 25, 30, 40, 100]
    labels = ['<15', '15-20', '20-25', '25-30', '30-40', '40+']
    merged['vix_bucket'] = pd.cut(merged['vix_close'], bins=bins, labels=labels, right=False)

    print(f"\n  {C_B}{C_M}{DIV}{RST}")
    print(f"  {C_B}{C_M}     VIX REGIME BREAKDOWN  (Combined 24/7 Strategy){RST}")
    print(f"  {C_B}{C_M}{DIV}{RST}")
    print(f"  {THIN}")
    print(f"  {'VIX Range':<12} {'Trades':>8} {'Wins':>8} {'Win Rate':>10} {'Avg R':>8} {'Total R':>10} {'Net PnL':>12}")
    print(f"  {THIN}")

    bucket_stats = []
    for label in labels:
        subset = merged[merged['vix_bucket'] == label]
        if len(subset) == 0:
            continue
        trades = len(subset)
        wins = subset['win'].sum()
        wr = wins / trades * 100
        avg_r = subset['r_multiple'].mean()
        total_r = subset['r_multiple'].sum()
        net_pnl = subset['pnl'].sum()

        rc = C_G if avg_r > 0 else C_R
        nc = C_G if net_pnl > 0 else C_R
        print(f"  {C_W}{label:<12}{RST} {trades:>8,} {wins:>8,} {wr:>9.1f}% {rc}{avg_r:>7.2f}R{RST} {rc}{total_r:>9.1f}R{RST} {nc}${net_pnl:>10,.2f}{RST}")

        bucket_stats.append({
            'label': label, 'trades': trades, 'wins': wins,
            'wr': wr, 'avg_r': avg_r, 'total_r': total_r, 'net_pnl': net_pnl
        })

    print(f"  {THIN}")

    # ── Fine-grained: 1-point VIX increments ─────────────────────────────────
    print(f"\n  {C_B}Fine-Grained VIX Analysis (1-pt bins, min 10 trades):{RST}")
    print(f"  {THIN}")
    print(f"  {'VIX Level':<12} {'Trades':>8} {'Win Rate':>10} {'Avg R':>8} {'Total R':>10} {'Verdict':>10}")
    print(f"  {THIN}")

    fine_bins = range(10, 50)
    for vl in fine_bins:
        subset = merged[(merged['vix_close'] >= vl) & (merged['vix_close'] < vl + 1)]
        if len(subset) < 10:
            continue
        trades = len(subset)
        wins = subset['win'].sum()
        wr = wins / trades * 100
        avg_r = subset['r_multiple'].mean()
        total_r = subset['r_multiple'].sum()

        rc = C_G if avg_r > 0 else C_R
        verdict = f"{C_G}EDGE{RST}" if avg_r > 0 else f"{C_R}AVOID{RST}"
        print(f"  {C_W}{vl}-{vl+1:<8}{RST} {trades:>8,} {wr:>9.1f}% {rc}{avg_r:>7.2f}R{RST} {rc}{total_r:>9.1f}R{RST} {verdict}")

    print(f"  {THIN}")

    # ── Optimal VIX Filter Discovery ─────────────────────────────────────────
    print(f"\n  {C_B}{C_Y}OPTIMAL VIX FILTER SEARCH:{RST}")
    print(f"  {THIN}")

    best_filter = None
    best_improvement = 0
    baseline_avg_r = merged['r_multiple'].mean()
    baseline_total = merged['pnl'].sum()

    for vix_max in np.arange(15, 45, 1):
        filtered = merged[merged['vix_close'] < vix_max]
        if len(filtered) < 50:
            continue
        avg_r = filtered['r_multiple'].mean()
        wr = filtered['win'].mean() * 100
        net = filtered['pnl'].sum()
        improvement = avg_r - baseline_avg_r

        if improvement > best_improvement:
            best_improvement = improvement
            best_filter = {'max_vix': vix_max, 'trades': len(filtered), 'wr': wr,
                          'avg_r': avg_r, 'net': net, 'total_r': filtered['r_multiple'].sum()}

    for vix_min in np.arange(15, 45, 1):
        filtered = merged[merged['vix_close'] >= vix_min]
        if len(filtered) < 50:
            continue
        avg_r = filtered['r_multiple'].mean()
        wr = filtered['win'].mean() * 100
        net = filtered['pnl'].sum()
        improvement = avg_r - baseline_avg_r

        if improvement > best_improvement:
            best_improvement = improvement
            best_filter = {'min_vix': vix_min, 'trades': len(filtered), 'wr': wr,
                          'avg_r': avg_r, 'net': net, 'total_r': filtered['r_multiple'].sum()}

    # Window filters
    for vix_lo in np.arange(12, 35, 1):
        for vix_hi in np.arange(vix_lo + 5, 50, 1):
            filtered = merged[(merged['vix_close'] >= vix_lo) & (merged['vix_close'] < vix_hi)]
            if len(filtered) < 50:
                continue
            avg_r = filtered['r_multiple'].mean()
            improvement = avg_r - baseline_avg_r
            if improvement > best_improvement:
                best_improvement = improvement
                wr = filtered['win'].mean() * 100
                net = filtered['pnl'].sum()
                best_filter = {'window': f'{vix_lo}-{vix_hi}', 'trades': len(filtered),
                              'wr': wr, 'avg_r': avg_r, 'net': net,
                              'total_r': filtered['r_multiple'].sum()}

    print(f"  {C_W}Baseline (no filter):{RST}")
    print(f"    Trades: {len(merged):,}  |  Avg R: {baseline_avg_r:.3f}R  |  Net: ${baseline_total:,.2f}")

    if best_filter:
        print(f"\n  {C_G}Best VIX Filter Found:{RST}")
        for k, v in best_filter.items():
            if k == 'net':
                print(f"    {k}: ${v:,.2f}")
            elif k == 'wr':
                print(f"    {k}: {v:.2f}%")
            elif k == 'avg_r' or k == 'total_r':
                print(f"    {k}: {v:.3f}R")
            else:
                print(f"    {k}: {v}")
        print(f"    {C_G}Avg R improvement: +{best_improvement:.3f}R per trade{RST}")
    else:
        print(f"  {C_Y}No VIX filter improved Avg R.{RST}")

    print(f"  {THIN}")
    print(f"  {C_B}{C_M}{DIV}{RST}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('Quant Lab 24/7 -- VIX Regime Analysis', color='#c9d1d9',
                 fontsize=14, fontweight='bold', y=0.98)

    bs = pd.DataFrame(bucket_stats)

    # 1. Win Rate by VIX bucket
    ax = axes[0, 0]; ax.set_facecolor('#161b22')
    colors = [('#3fb950' if r > baseline_avg_r else '#f85149') for r in bs['avg_r']]
    ax.bar(bs['label'], bs['wr'], color=colors, alpha=0.8, edgecolor='#30363d')
    ax.axhline(merged['win'].mean()*100, color='#58a6ff', linestyle='--', linewidth=1, label='Baseline WR')
    ax.set_ylabel('Win Rate (%)', color='#c9d1d9')
    ax.set_xlabel('VIX Range', color='#c9d1d9')
    ax.set_title('Win Rate by VIX Regime', color='#c9d1d9')
    ax.tick_params(colors='#8b949e')
    for sp in ax.spines.values(): sp.set_color('#30363d')
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    # 2. Avg R by VIX bucket
    ax = axes[0, 1]; ax.set_facecolor('#161b22')
    colors2 = [('#3fb950' if r > 0 else '#f85149') for r in bs['avg_r']]
    ax.bar(bs['label'], bs['avg_r'], color=colors2, alpha=0.8, edgecolor='#30363d')
    ax.axhline(0, color='#484f58', linewidth=0.8)
    ax.axhline(baseline_avg_r, color='#58a6ff', linestyle='--', linewidth=1, label='Baseline Avg R')
    ax.set_ylabel('Avg R per Trade', color='#c9d1d9')
    ax.set_xlabel('VIX Range', color='#c9d1d9')
    ax.set_title('R-Expectancy by VIX Regime', color='#c9d1d9')
    ax.tick_params(colors='#8b949e')
    for sp in ax.spines.values(): sp.set_color('#30363d')
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    # 3. Trade Count by VIX bucket
    ax = axes[1, 0]; ax.set_facecolor('#161b22')
    ax.bar(bs['label'], bs['trades'], color='#58a6ff', alpha=0.7, edgecolor='#30363d')
    ax.set_ylabel('Trade Count', color='#c9d1d9')
    ax.set_xlabel('VIX Range', color='#c9d1d9')
    ax.set_title('Trade Distribution by VIX', color='#c9d1d9')
    ax.tick_params(colors='#8b949e')
    for sp in ax.spines.values(): sp.set_color('#30363d')

    # 4. Scatter: VIX vs R-Multiple
    ax = axes[1, 1]; ax.set_facecolor('#161b22')
    wins_m = merged[merged['win'] == 1]
    losses_m = merged[merged['win'] == 0]
    ax.scatter(losses_m['vix_close'], losses_m['r_multiple'], color='#f85149',
               alpha=0.3, s=8, label='Loss', zorder=2)
    ax.scatter(wins_m['vix_close'], wins_m['r_multiple'], color='#3fb950',
               alpha=0.5, s=12, label='Win', zorder=3)
    ax.axhline(0, color='#484f58', linewidth=0.8)
    ax.set_ylabel('R-Multiple', color='#c9d1d9')
    ax.set_xlabel('VIX Close', color='#c9d1d9')
    ax.set_title('Trade Outcomes vs VIX Level', color='#c9d1d9')
    ax.tick_params(colors='#8b949e')
    for sp in ax.spines.values(): sp.set_color('#30363d')
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    plt.tight_layout()
    out_path = _ROOT / "Model Training" / "vix_regime_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"\n  {C_G}Chart saved: {out_path.name}{RST}\n")


if __name__ == "__main__":
    main()
