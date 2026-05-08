"""
vix_filtered_tearsheet.py
=========================
Side-by-side: Standard 24/7 vs VIX-Filtered (17 <= VIX <= 30).
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
VIX_LO, VIX_HI  = 17.0, 30.0

DROP_COLS = ['Open','High','Low','Close','Volume','Minutes_From_Open',
             'Day_Of_Week','datetime','ts_event']

def _latest(d, pat):
    m = sorted(d.glob(pat), key=lambda p: p.stat().st_mtime)
    if not m: print(f"  ERROR: no {pat}"); sys.exit(1)
    return m[-1]

def _run_regime(prefix, holdout_pat, time_filter_fn):
    mp = f"{prefix}_"
    lgbm=joblib.load(_latest(_MODEL,f"{mp}lgbm_model_*.joblib"))
    rf=joblib.load(_latest(_MODEL,f"{mp}rf_model_*.joblib"))
    mlp=joblib.load(_latest(_MODEL,f"{mp}mlp_model_*.joblib"))
    scaler=joblib.load(_latest(_MODEL,f"{mp}mlp_scaler_*.joblib"))
    meta=joblib.load(_latest(_MODEL,f"{mp}meta_model_*.joblib"))
    thresh=joblib.load(_latest(_MODEL,f"{mp}optimal_thresholds_*.joblib"))
    lt=thresh["long_threshold"];st=thresh["short_threshold"]
    LWR=thresh["long_win_r"];LLR=thresh["long_loss_r"]
    SWR=thresh["short_win_r"];SLR=thresh["short_loss_r"]
    hp=_latest(_FEAT,holdout_pat); df=pd.read_parquet(hp)
    dt_col='ts_event' if 'ts_event' in df.columns else 'datetime'
    timestamps=pd.to_datetime(df[dt_col])
    target="Target" if "Target" in df.columns else "target"
    y=df[target].to_numpy()
    X=df.drop(columns=[target])
    X=X.drop(columns=[c for c in DROP_COLS if c in X.columns])
    X=X.select_dtypes(include=[np.number])
    Xs=scaler.transform(X)
    ep=(lgbm.predict_proba(X)+rf.predict_proba(X)+mlp.predict_proba(Xs))/3.0
    ref=lgbm.classes_
    li=int(np.where(ref==1)[0][0]);si=int(np.where(ref==-1)[0][0]);ci=int(np.where(ref==0)[0][0])
    lp=ep[:,li];sp=ep[:,si]
    lm=lp>=lt;sm=sp>=st;both=lm&sm
    ls=lm.copy();ss=sm.copy()
    ls[both]=lp[both]>=sp[both];ss[both]=sp[both]>lp[both]
    direction=np.zeros(len(y),dtype=int);direction[ls]=1;direction[ss]=-1
    t_idx=np.where(ls|ss)[0]
    Xt=X.iloc[t_idx].copy()
    Xt['ens_long_proba']=ep[t_idx,li];Xt['ens_short_proba']=ep[t_idx,si]
    Xt['ens_chop_proba']=ep[t_idx,ci];Xt['ens_direction']=direction[t_idx]
    approved=meta.predict(Xt)==1;approved_idx=t_idx[approved]
    tt=timestamps.iloc[approved_idx].values;td=direction[approved_idx];ty=y[approved_idx]
    if time_filter_fn is not None:
        ts_s=pd.Series(tt);tm=pd.to_datetime(ts_s).dt.hour*60+pd.to_datetime(ts_s).dt.minute
        mask=time_filter_fn(tm).values;tt=tt[mask];td=td[mask];ty=ty[mask]
    cd_ns=np.timedelta64(COOLDOWN_MIN,'m');keep=[];last_t=None
    for i in range(len(tt)):
        if last_t is not None and (tt[i]-last_t)<cd_ns: continue
        keep.append(i);last_t=tt[i]
    tt=tt[keep];td=td[keep];ty=ty[keep]
    pnl=[]
    for d,yv in zip(td,ty):
        if d==1 and yv==1: pnl.append(LWR*RISK_PER_TRADE)
        elif d==1: pnl.append(LLR*RISK_PER_TRADE)
        elif d==-1 and yv==-1: pnl.append(SWR*RISK_PER_TRADE)
        else: pnl.append(SLR*RISK_PER_TRADE)
    return pd.DataFrame({'timestamp':tt,'direction':td,'actual':ty,'pnl':pnl,
                         'win':[1 if p>0 else 0 for p in pnl]})

def _calc_metrics(trades_df):
    pnl=trades_df['pnl'].to_numpy()
    if len(pnl)==0:
        return {k:0 for k in ['trades','wins','wr','net','final','dd','pf','mcl','r2d','total_r','longest_dd']}
    eq=STARTING_BALANCE+np.cumsum(pnl);pk=np.maximum.accumulate(eq);ddown=eq-pk
    net=eq[-1]-STARTING_BALANCE;dd=ddown.min()
    gp=pnl[pnl>0].sum();gl=abs(pnl[pnl<0].sum())
    pf=gp/gl if gl>0 else float('inf')
    mcl=0;s=0
    for p in pnl:
        if p<0: s+=1;mcl=max(mcl,s)
        else: s=0
    r2d=abs(net/dd) if dd!=0 else float('inf')
    # Longest DD period
    ts=trades_df['timestamp'].values;dd_start=None;longest=0
    for i in range(len(ddown)):
        if ddown[i]<0:
            if dd_start is None: dd_start=ts[i]
        else:
            if dd_start is not None:
                dur=(ts[i]-dd_start)/np.timedelta64(1,'D');longest=max(longest,dur);dd_start=None
    if dd_start is not None:
        dur=(ts[-1]-dd_start)/np.timedelta64(1,'D');longest=max(longest,dur)
    return {'trades':len(pnl),'wins':int((pnl>0).sum()),'wr':(pnl>0).sum()/len(pnl)*100,
            'net':net,'final':eq[-1],'dd':dd,'pf':pf,'mcl':mcl,'r2d':r2d,
            'total_r':net/RISK_PER_TRADE,'longest_dd':longest,'equity':eq,'drawdown':ddown}

def main():
    print(f"\n  {C_B}{C_M}{DIV}{RST}")
    print(f"  {C_B}{C_M}   QUANT LAB  --  VIX-FILTERED MASTER TEARSHEET{RST}")
    print(f"  {C_B}{C_M}{DIV}{RST}")

    # Load VIX
    vix=pd.read_csv(_ROOT/"TVC_VIX, 1D.csv")
    vix['date']=pd.to_datetime(vix['time']).dt.date
    vix.rename(columns={'close':'vix_close'},inplace=True)

    # Build combined trades
    print(f"  {C_C}Loading regimes...{RST}")
    day=_run_regime("day_shift","*day_shift_clean_holdout*",None)
    night=_run_regime("night_shift","*night_shift_clean_holdout*",lambda tm:(tm<480)|(tm>=1080))
    all_trades=pd.concat([day,night],ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    all_trades['date']=pd.to_datetime(all_trades['timestamp']).dt.date

    # Merge VIX
    merged=all_trades.merge(vix[['date','vix_close']],on='date',how='left')
    has_vix=merged['vix_close'].notna()
    standard=merged[has_vix].copy()
    filtered=standard[(standard['vix_close']>=VIX_LO)&(standard['vix_close']<=VIX_HI)].copy()

    print(f"  {C_C}Standard trades (with VIX): {len(standard):,}{RST}")
    print(f"  {C_C}VIX-Filtered ({VIX_LO}-{VIX_HI}): {len(filtered):,}  ({len(standard)-len(filtered):,} removed){RST}")

    s=_calc_metrics(standard)
    f=_calc_metrics(filtered)

    # Print comparison
    print(f"\n  {C_B}{C_M}{DIV}{RST}")
    print(f"  {C_B}{C_M}        STANDARD 24/7  vs  VIX-FILTERED ({VIX_LO:.0f}-{VIX_HI:.0f}){RST}")
    print(f"  {C_B}{C_M}{DIV}{RST}")
    print(f"  {THIN}")
    print(f"  {'Metric':<28} {'STANDARD':>16} {'VIX-FILTERED':>16} {'DELTA':>10}")
    print(f"  {THIN}")

    rows = [
        ('Trades',            'trades', '{:,}',    False, False),
        ('Wins',              'wins',   '{:,}',    False, False),
        ('Win Rate',          'wr',     '{:.2f}%', True,  False),
        ('Net Profit ($)',    'net',    '${:,.2f}', True,  True),
        ('Final Balance ($)', 'final',  '${:,.2f}', True,  True),
        ('Max Drawdown ($)',  'dd',     '${:,.2f}', False, True),
        ('Profit Factor',    'pf',     '{:.2f}',  True,  False),
        ('Max Consec. Losses','mcl',    '{}',      False, False),
        ('Reward/DD Ratio',  'r2d',    '{:.2f}x', True,  False),
        ('Total R',          'total_r','{:.1f}R',  True,  False),
        ('Longest DD (days)','longest_dd','{:.1f}', False, False),
    ]

    for label, key, fmt, higher_good, is_dollar in rows:
        sv=s[key]; fv=f[key]
        if key in ('equity','drawdown'): continue
        ss=fmt.format(sv); fs=fmt.format(fv)

        if key=='dd':
            delta=fv-sv  # less negative = better
            better=delta>0
        elif key in ('mcl','longest_dd'):
            delta=fv-sv
            better=delta<0
        elif higher_good:
            delta=fv-sv
            better=delta>0
        else:
            delta=fv-sv
            better=True

        if is_dollar:
            ds=f"${delta:+,.2f}"
        elif key=='wr':
            ds=f"{delta:+.2f}%"
        elif key in ('total_r',):
            ds=f"{delta:+.1f}R"
        elif key=='r2d':
            ds=f"{delta:+.2f}x"
        elif key=='pf':
            ds=f"{delta:+.2f}"
        elif key=='longest_dd':
            ds=f"{delta:+.1f}"
        else:
            ds=f"{delta:+,.0f}"

        dc=C_G if better else C_R
        sc=C_W; fc=C_G if better else C_W
        print(f"  {C_W}{label:<28}{RST} {sc}{ss:>16}{RST} {fc}{fs:>16}{RST} {dc}{ds:>10}{RST}")

    print(f"  {THIN}")
    print(f"  {C_B}{C_M}{DIV}{RST}")

    # Critical check
    print("")
    if f['mcl']<10:
        print(f"  {C_B}{C_G}  >> MAX CONSECUTIVE LOSSES DROPPED BELOW 10: {f['mcl']} <<{RST}")
    else:
        print(f"  {C_Y}  Max Consec. Losses: {f['mcl']} (did not drop below 10){RST}")

    if f['net']>0 and f['pf']>1:
        print(f"  {C_B}{C_G}  >> VIX-FILTERED STRATEGY IS PROFITABLE <<{RST}")
    print("")

    # ── Equity Curve ─────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                    gridspec_kw={'hspace': 0.08})
    fig.patch.set_facecolor('#0d1117')

    n_s = np.arange(1, s['trades']+1)
    n_f = np.arange(1, f['trades']+1)

    # Top: dual equity curves
    ax1.set_facecolor('#0d1117')
    ax1.plot(n_s, s['equity'], color='#484f58', linewidth=1.0, alpha=0.6, label=f"Standard ({s['trades']:,} trades)", zorder=2)
    ax1.plot(n_f, f['equity'], color='#58a6ff', linewidth=1.4, label=f"VIX {VIX_LO:.0f}-{VIX_HI:.0f} ({f['trades']:,} trades)", zorder=3)
    ax1.fill_between(n_f, STARTING_BALANCE, f['equity'],
                     where=f['equity']>=STARTING_BALANCE, alpha=0.12, color='#3fb950', zorder=1)
    ax1.axhline(STARTING_BALANCE, color='#484f58', linewidth=0.8, linestyle='--', alpha=0.5)
    ax1.set_ylabel('Equity ($)', color='#c9d1d9', fontsize=11)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
    ax1.tick_params(colors='#8b949e'); ax1.tick_params(labelbottom=False)
    ax1.set_title(f"Quant Lab 24/7  |  Standard vs VIX-Filtered ({VIX_LO:.0f}-{VIX_HI:.0f})  |  Net: ${f['net']:,.0f}  |  DD: ${f['dd']:,.0f}  |  R/DD: {f['r2d']:.1f}x",
                  color='#c9d1d9', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=10)
    for sp in ['top','right']: ax1.spines[sp].set_visible(False)
    ax1.spines['bottom'].set_color('#30363d'); ax1.spines['left'].set_color('#30363d')

    # Bottom: filtered drawdown
    ax2.set_facecolor('#0d1117')
    ax2.fill_between(n_f, f['drawdown'], 0, color='#f85149', alpha=0.4)
    ax2.plot(n_f, f['drawdown'], color='#f85149', linewidth=0.8)
    ax2.set_ylabel('Drawdown ($)', color='#c9d1d9', fontsize=11)
    ax2.set_xlabel('Trade #', color='#c9d1d9', fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
    ax2.tick_params(colors='#8b949e')
    for sp in ['top','right']: ax2.spines[sp].set_visible(False)
    ax2.spines['bottom'].set_color('#30363d'); ax2.spines['left'].set_color('#30363d')

    out_path = _ROOT / "Model Training" / "quant_lab_vix_optimized.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  {C_G}Equity curve saved: {out_path.name}{RST}\n")


if __name__ == "__main__":
    main()
