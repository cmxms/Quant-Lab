# Quant Lab: NQ Futures Ensemble Trading System
## Project Overview & Strategic Audit (May 7, 2026)

This document provides a comprehensive breakdown of the NQ futures quantitative trading pipeline, summarizing the architecture, today's specific enhancements, and the results of our final "Proof of Concept" audit on sealed holdout data.

---

## 1. System Architecture & Data Flow

The system operates as a modular "factory" pipeline where raw market data is transformed into audited trading signals.

1.  **Data Ingestion (`update_and_stitch.py`)**: 
    *   **Automated Pipeline**: Coordinates `update_nq_data.py` (located in `Data Ingestion/Raw/`) and `rollover_stitch.py` (Continuous EST).
    *   **Raw Data Management**: Centralizes all master Parquet source files and daily update scripts within the `Data Ingestion/Raw/` directory.
    *   **Synchronization**: Automatically appends Yahoo Finance data (`NQ=F`) to the master Databento series, handles back-adjustments, and synchronizes both the raw and continuous datasets daily.
2.  **Feature Generation (`feature_gen.py`)**: 
    *   Calculates 14+ stationary technical indicators (RSI, MACD, Bollinger, etc.).
    *   **Regime State Logic**: Engineers macro-context features like `RVOL_60m` (liquidity) and `Body_Wick_Ratio_10` (chop filter) to help the model distinguish between trend and mean-reversion environments.
3.  **Labeling (`label_targets_long.py`)**: 
    *   Uses the **Triple Barrier Method** with Numba-accelerated code.
    *   Labels every bar as **1** (Profit Hit), **-1** (Stop Hit), or **0** (No Hit/Vertical Barrier).
4.  **Feature Selection (`mi_processing.py`)**: 
    *   Calculates **Mutual Information (MI)** scores between features and targets.
    *   Prunes features with zero or near-zero predictive alpha to prevent over-fitting.
5.  **Ensemble Training**: 
    *   Trains three distinct architectures: **LightGBM** (Gradient Boosting), **Random Forest** (Bagging), and **Logistic Regression** (Linear).
    *   Saves models and scalers into the `Model Training/Outputs/` directory.
6.  **Orchestration & Audit**: 
    *   Combines model probabilities using a **Weighted Soft Vote**.
    *   Audits model logic via **SHAP Values** to ensure they are learning real market relationships, not just chasing noise.

---

## 2. Today's Key Enhancements

We focused on lifting the system out of "noise" and into a more robust, regime-aware state:

*   **ATR Target Revision**: Moved from high-frequency targets (2.0x/1.5x ATR) to more significant structural targets (**6.0x ATR Profit / 4.0x ATR Stop**). This reduced label noise and increased model accuracy.
*   **Regime Feature Integration**: Added `RVOL_60m` and `Body_Wick_Ratio_10`. A SHAP audit confirmed that `RVOL_60m` is now the #1 most important feature for both tree models.
*   **Ensemble Expansion**: Successfully integrated a 3-model voting engine.
*   **Prop-Firm Backtest Engine**: Built a realistic backtest (`holdout_backtest.py`) that includes:
    *   1-tick slippage penalty on entries.
    *   $4.00 round-trip commission.
    *   15:55 EST Force-Flatten (no overnight risk).

---

## 3. Backtest & Audit Ledger

### A. Model Performance (Sealed Holdout Set)
*Tested on 354,110 bars of unseen data from May 2025 to May 2026.*

| Model | Accuracy | Precision (Weighted) | Verdict |
| :--- | :--- | :--- | :--- |
| **LightGBM** | 45.67% | 0.5369 | Strongest single-model generalization. |
| **Ensemble** | 43.69% | **0.5443** | Highest precision (lowest false signals). |
| **Random Forest**| 43.36% | 0.5419 | Consistent with LGBM. |
| **LogReg** | 36.22% | 0.5435 | Underperformed on accuracy. |

### B. Threshold Sensitivity Analysis
We analyzed how the **Weighted Ensemble Vote** (LGBM 50% / RF 35% / LR 15%) performed across different conviction levels:

| Threshold | Trades/Yr | Win % | Net P&L | Max Drawdown |
| :--- | :--- | :--- | :--- | :--- |
| 0.45 | 886 | 37.9% | -$33,005 | $53,699 |
| 0.50 | 345 | 36.5% | -$14,429 | $34,364 |
| 0.55 | 127 | 37.0% | -$6,432 | $10,753 |
| 0.60 | 18 | 27.8% | -$6,702 | $8,174 |

### C. Winning Trade DNA Audit
We isolated the "Winning DNA" of the signals to find the **Power 3** indicators that best separate a Win from a Loss:
1.  **RVOL_60m** (Must be < 0.59): Wins occur in lower macro-volume environments.
2.  **VWAP_Dist** (Must be < 0.0006): Wins occur when price is tighter to the VWAP.
3.  **BB_PctB** (Must be > 0.63): Wins prefer a slight bullish extension.

**Audit Re-Test Result**: Applying these filters and a 1:1 R:R (2.0x ATR TP/SL) lifted the Win Rate to **46.1%**, but still fell short of the **52% break-even mark** needed to overcome friction (commissions/slippage).

---

## 4. Current Conclusion & Next Steps

The system is mathematically sound and the pipeline is robust, but **the "Edge" is currently negative** as a long-only system. The model is correctly identifying market "movement" but lacks enough conviction to overcome the friction of trading a 1-minute NQ time series.

**Potential Solutions for Collaboration**:
1.  **Short-Side Integration**: The model may have a stronger edge identifying short signals in the current NQ regime.
2.  **Higher Timeframe Features**: Incorporating 5m or 15m context into the 1m signals to improve the "DNA" separation.
3.  **Asymmetric R:R**: Testing 8:1 or higher Reward-to-Risk ratios to capitalize on the 36-40% win rate.
