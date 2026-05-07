"""
holdout_backtest.py
===================
Prop-Firm-Compliant Backtest on Sealed Holdout Data
Uses Backtrader with Weighted Soft Vote Ensemble signals.

Pipeline Stage : Backtesting (Phase 5)
Inputs         : Feature Engineering/Outputs/long_clean_holdout.parquet
                 Model Training/Outputs/rf_model_*.joblib
                 Model Training/Outputs/lgbm_model_*.joblib
                 Model Training/Outputs/logreg_pipeline_*.joblib
Outputs        : Model Training/Outputs/backtest_equity_curve.png

NQ Contract Specs
-----------------
  Multiplier : $20 / point
  Tick size  : 0.25 points  ($5 / tick)
  Slippage   : 1 tick = 0.25 points (entry only)
  Commission : $2.00 / contract / side  ($4.00 round trip)
  Margin     : $15,000 (approx CME day-session requirement)

Entry Rules
-----------
  Signal     : Weighted_Vote (Class 1 probability) > 0.65
  Weights    : LGBM x0.50  |  RF x0.35  |  LogReg x0.15
  Size       : 1 contract
  Target     : Close + (6.0 x ATR)
  Stop       : Close - (4.0 x ATR)
  EOD Flatten: Forcefully close any open position at 15:55 EST
"""

import argparse
import logging
import sys
from pathlib import Path

import backtrader as bt
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ── Configuration ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_HERE               = Path(__file__).parent.resolve()
_FEATURE_OUTPUT_DIR = _HERE.parent / "Feature Engineering" / "Outputs"
_MODEL_OUTPUT_DIR   = _HERE / "Outputs"

TARGET_COL  = "Target"
DROP_COLS   = ["Open", "High", "Low", "Close", "Volume", "Minutes_From_Open", "Day_Of_Week"]
CANONICAL   = np.array([-1, 0, 1])

STARTING_CASH    = 50_000.0
COMMISSION_SIDE  = 2.00        # $ per contract per side
NQ_MULTIPLIER    = 20.0        # $ per point
NQ_TICK          = 0.25        # points per tick (slippage)
NQ_MARGIN        = 15_000.0    # approx day-trading margin
VOTE_THRESHOLD   = 0.65
TP_MULT          = 6.0
SL_MULT          = 4.0
EOD_HOUR         = 15
EOD_MINUTE       = 55

# Ensemble weights
W_LGBM   = 0.50
W_RF     = 0.35
W_LOGREG = 0.15

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
DIVIDER = "-" * 60

def _section(title):
    log.info(DIVIDER)
    log.info(f"  {title}")
    log.info(DIVIDER)

# ---------------------------------------------------------------------------
# ── Helper: Find Latest Model ────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _latest(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        log.error(f"[MISSING] No '{pattern}' in {directory}")
        sys.exit(1)
    log.info(f"  {label:<12}: {matches[-1].name}")
    return matches[-1]

# ---------------------------------------------------------------------------
# ── Helper: Align predict_proba to canonical [-1, 0, 1] order ───────────────
# ---------------------------------------------------------------------------

def _align(proba: np.ndarray, model_classes: np.ndarray) -> np.ndarray:
    out = np.zeros((proba.shape[0], 3), dtype=np.float64)
    for i, cls in enumerate(CANONICAL):
        col = np.where(model_classes == cls)[0]
        if col.size:
            out[:, i] = proba[:, col[0]]
    return out

# ---------------------------------------------------------------------------
# ── Phase 1: Math -- Weighted Soft Vote ─────────────────────────────────────
# ---------------------------------------------------------------------------

def math_phase(holdout_path: Path) -> pd.DataFrame:
    _section("Phase 1 / 3  -  Math: Weighted Soft Vote")

    # Load models
    log.info("  Loading models ...")
    lgbm_model   = joblib.load(_latest(_MODEL_OUTPUT_DIR, "lgbm_model_*.joblib",     "LGBM"))
    rf_model     = joblib.load(_latest(_MODEL_OUTPUT_DIR, "rf_model_*.joblib",       "RF"))
    logreg_pipe  = joblib.load(_latest(_MODEL_OUTPUT_DIR, "logreg_pipeline_*.joblib","LogReg"))
    logreg_inner = logreg_pipe.named_steps["model"]

    # Load holdout
    log.info(f"  Loading holdout: {holdout_path.name}")
    df = pd.read_parquet(holdout_path)
    log.info(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Feature matrix (14 features only)
    drop = [c for c in [TARGET_COL] + DROP_COLS if c in df.columns]
    X = df.drop(columns=drop)
    log.info(f"  Feature matrix: {X.shape[1]} features  -> {list(X.columns)}")

    # predict_proba for each model, extract Class 1 probability (index 2)
    log.info("  Running predict_proba() ...")
    lgbm_p1   = _align(lgbm_model.predict_proba(X),    lgbm_model.classes_)[:, 2]
    rf_p1     = _align(rf_model.predict_proba(X),      rf_model.classes_)[:, 2]
    logreg_p1 = _align(logreg_pipe.predict_proba(X),   logreg_inner.classes_)[:, 2]

    # Weighted vote
    df["weighted_vote"] = lgbm_p1 * W_LGBM + rf_p1 * W_RF + logreg_p1 * W_LOGREG

    # ATR in raw price units for bracket calculation
    df["atr_norm"] = df["ATR_Norm"]   # kept normalized; strategy computes ATR = atr_norm * close

    signals_fired = (df["weighted_vote"] > VOTE_THRESHOLD).sum()
    log.info(f"  Weighted Vote computed. Bars above {VOTE_THRESHOLD} threshold: {signals_fired:,}")
    log.info(f"  Vote range: [{df['weighted_vote'].min():.4f}, {df['weighted_vote'].max():.4f}]")

    return df

# ---------------------------------------------------------------------------
# ── Phase 2: Backtrader Components ──────────────────────────────────────────
# ---------------------------------------------------------------------------

class NQFeed(bt.feeds.PandasData):
    """Custom 1-minute NQ feed with weighted_vote and atr_norm lines."""
    lines = ("weighted_vote", "atr_norm",)
    params = (
        ("weighted_vote", -1),  # -1 = auto-match column name
        ("atr_norm",      -1),
    )


class NQCommission(bt.CommInfoBase):
    """NQ futures commission: $2/contract/side, $20/point multiplier."""
    params = (
        ("commission", COMMISSION_SIDE),
        ("mult",       NQ_MULTIPLIER),
        ("commtype",   bt.CommInfoBase.COMM_FIXED),
        ("stocklike",  False),
        ("margin",     NQ_MARGIN),
    )

    def getcommission(self, size, price):
        return abs(size) * self.p.commission


class TradeLogger(bt.Analyzer):
    """Records closed trade P&L for win-rate and stats reporting."""
    def start(self):
        self.trades = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({
                "pnl":     trade.pnl,
                "pnlcomm": trade.pnlcomm,
                "won":     trade.pnlcomm > 0,
            })

    def get_analysis(self):
        return self.trades


class EquityRecorder(bt.Analyzer):
    """Records (datetime, portfolio_value) at each bar."""
    def start(self):
        self._dates  = []
        self._values = []

    def next(self):
        self._dates.append(self.data.datetime.datetime(0))
        self._values.append(self.strategy.broker.getvalue())

    def get_analysis(self):
        return self._dates, self._values


class NQEnsembleStrategy(bt.Strategy):
    params = (
        ("vote_threshold", VOTE_THRESHOLD),
        ("tp_mult",        TP_MULT),
        ("sl_mult",        SL_MULT),
        ("eod_hour",       EOD_HOUR),
        ("eod_minute",     EOD_MINUTE),
    )

    def __init__(self):
        self.bracket = None   # holds [main, stop, limit] order refs

    def _is_eod(self, dt):
        return dt.hour > self.p.eod_hour or (
            dt.hour == self.p.eod_hour and dt.minute >= self.p.eod_minute
        )

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            # Any completed SELL (TP or SL bracket) closes the position
            if order.issell():
                self.bracket = None

        elif order.status in [order.Cancelled, order.Expired, order.Rejected]:
            # If a bracket leg was cancelled and position is flat, clean up ref
            if self.bracket and order in self.bracket:
                if not self.position:
                    self.bracket = None

    def next(self):
        dt = self.data.datetime.datetime(0)

        # ── EOD Rule: flatten & cancel at 15:55 ──────────────────────────────
        if self._is_eod(dt):
            if self.position:
                self.close()
                if self.bracket:
                    for o in self.bracket[1:]:   # cancel TP & SL legs
                        self.cancel(o)
                    self.bracket = None
            elif self.bracket:
                # Main order pending but not yet filled -- cancel everything
                for o in self.bracket:
                    self.cancel(o)
                self.bracket = None
            return

        # ── Entry Rule ────────────────────────────────────────────────────────
        if not self.position and self.bracket is None:
            if self.data.weighted_vote[0] > self.p.vote_threshold:
                close = self.data.close[0]
                atr   = self.data.atr_norm[0] * close   # ATR in price units

                tp = close + self.p.tp_mult * atr
                sl = close - self.p.sl_mult * atr

                self.bracket = self.buy_bracket(
                    size       = 1,
                    limitprice = tp,
                    stopprice  = sl,
                )

# ---------------------------------------------------------------------------
# ── Phase 3: Run Cerebro & Report ───────────────────────────────────────────
# ---------------------------------------------------------------------------

def backtest_phase(df: pd.DataFrame) -> None:
    _section("Phase 2 / 3  -  Building Backtrader Feed")

    # Prepare feed DataFrame
    # Backtrader PandasData needs naive datetime index
    feed_df = df.copy()
    if hasattr(feed_df.index, "tz") and feed_df.index.tz is not None:
        feed_df.index = feed_df.index.tz_convert("US/Eastern").tz_localize(None)

    # Rename OHLCV to lowercase as BT expects
    rename_map = {}
    for col in feed_df.columns:
        if col.lower() in ["open", "high", "low", "close", "volume"]:
            rename_map[col] = col.lower()
    feed_df.rename(columns=rename_map, inplace=True)

    # Verify required columns exist
    for req in ["open", "high", "low", "close", "volume", "weighted_vote", "atr_norm"]:
        if req not in feed_df.columns:
            log.error(f"[MISSING COLUMN] '{req}' not found in feed DataFrame.")
            sys.exit(1)

    log.info(f"  Feed shape  : {feed_df.shape[0]:,} rows")
    log.info(f"  Date range  : {feed_df.index[0]}  to  {feed_df.index[-1]}")

    data_feed = NQFeed(
        dataname  = feed_df,
        datetime  = None,      # use index
        timeframe = bt.TimeFrame.Minutes,
        compression = 1,
    )

    # Cerebro
    _section("Phase 3 / 3  -  Running Cerebro")
    cerebro = bt.Cerebro()
    cerebro.addstrategy(NQEnsembleStrategy)
    cerebro.adddata(data_feed)

    cerebro.broker.setcash(STARTING_CASH)
    cerebro.broker.addcommissioninfo(NQCommission())
    cerebro.broker.set_slippage_fixed(NQ_TICK)   # 1-tick slippage on market fills

    cerebro.addanalyzer(TradeLogger,    _name="trades")
    cerebro.addanalyzer(EquityRecorder, _name="equity")

    log.info(f"  Starting portfolio value : ${STARTING_CASH:,.2f}")
    log.info(f"  Commission               : ${COMMISSION_SIDE:.2f}/contract/side")
    log.info(f"  Slippage                 : {NQ_TICK} points (1 tick)")
    log.info(f"  NQ multiplier            : ${NQ_MULTIPLIER:.0f}/point")
    log.info(f"  Vote threshold           : {VOTE_THRESHOLD}")
    log.info(f"  TP / SL multipliers      : {TP_MULT}x / {SL_MULT}x ATR")
    log.info(f"  EOD flatten              : {EOD_HOUR}:{EOD_MINUTE:02d} EST")
    log.info("")
    log.info("  Running backtest (this may take several minutes on 354K bars) ...")

    results = cerebro.run()
    strat   = results[0]

    ending_value = cerebro.broker.getvalue()
    net_profit   = ending_value - STARTING_CASH

    trade_log = strat.analyzers.trades.get_analysis()
    eq_dates, eq_values = strat.analyzers.equity.get_analysis()

    total_trades = len(trade_log)
    won_trades   = sum(1 for t in trade_log if t["won"])
    win_rate     = (won_trades / total_trades * 100) if total_trades else 0.0
    avg_win      = np.mean([t["pnlcomm"] for t in trade_log if t["won"]])    if won_trades else 0.0
    avg_loss     = np.mean([t["pnlcomm"] for t in trade_log if not t["won"]]) if (total_trades - won_trades) else 0.0

    # ── Print Results ─────────────────────────────────────────────────────────
    _section("Backtest Results  --  Sealed Holdout")
    log.info(f"  Starting Value   : ${STARTING_CASH:>12,.2f}")
    log.info(f"  Ending Value     : ${ending_value:>12,.2f}")
    log.info(f"  Net Profit       : ${net_profit:>+12,.2f}")
    log.info(f"  Return           : {net_profit / STARTING_CASH * 100:>+.2f}%")
    log.info("")
    log.info(f"  Total Trades     : {total_trades:>6}")
    log.info(f"  Winning Trades   : {won_trades:>6}")
    log.info(f"  Losing  Trades   : {total_trades - won_trades:>6}")
    log.info(f"  Win Rate         : {win_rate:>6.1f}%")
    log.info(f"  Avg Win  (net)   : ${avg_win:>+10,.2f}")
    log.info(f"  Avg Loss (net)   : ${avg_loss:>+10,.2f}")
    if avg_loss != 0:
        log.info(f"  Profit Factor    : {abs(avg_win / avg_loss):.2f}x")
    log.info("")

    # ── Equity Curve Plot ─────────────────────────────────────────────────────
    if eq_dates and eq_values:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(eq_dates, eq_values, linewidth=1.2, color="#2196F3", label="Portfolio Value")
        ax.axhline(STARTING_CASH, color="#9E9E9E", linewidth=0.8, linestyle="--", label="Starting Capital")
        ax.fill_between(
            eq_dates, STARTING_CASH, eq_values,
            where=[v >= STARTING_CASH for v in eq_values],
            alpha=0.15, color="#4CAF50",
        )
        ax.fill_between(
            eq_dates, STARTING_CASH, eq_values,
            where=[v < STARTING_CASH for v in eq_values],
            alpha=0.15, color="#F44336",
        )
        ax.set_title(
            f"Equity Curve -- NQ Ensemble Backtest (Holdout: {eq_dates[0].date()} to {eq_dates[-1].date()})\n"
            f"Net P&L: ${net_profit:+,.0f}  |  Trades: {total_trades}  |  Win Rate: {win_rate:.1f}%  |  "
            f"Weights: LGBM {W_LGBM:.0%} / RF {W_RF:.0%} / LogReg {W_LOGREG:.0%}",
            fontsize=11,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        plt.tight_layout()

        out_path = _MODEL_OUTPUT_DIR / "backtest_equity_curve.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Equity curve saved -> {out_path}")
    else:
        log.warning("  No equity data recorded -- plot skipped.")

    log.info("")
    _section("Done  [v]  holdout_backtest.py completed successfully")
    log.info("")

# ---------------------------------------------------------------------------
# ── Entry Point ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prop-firm-compliant backtest of ensemble signals on sealed holdout data.",
    )
    parser.add_argument(
        "--holdout_path", type=Path, default=None,
        help="Override path to holdout .parquet file.",
    )
    args = parser.parse_args()

    holdout_path = args.holdout_path or (_FEATURE_OUTPUT_DIR / "long_clean_holdout.parquet")
    if not holdout_path.exists():
        log.error(f"[MISSING] Holdout file not found: {holdout_path}")
        sys.exit(1)

    log.info("")
    _section("Holdout Backtest  -  Quantitative Trading Pipeline  |  Phase 5")
    log.info(f"  Holdout file : {holdout_path.name}")
    log.info(f"  Initial cash : ${STARTING_CASH:,.0f}")
    log.info("")

    df = math_phase(holdout_path)
    backtest_phase(df)


if __name__ == "__main__":
    main()
