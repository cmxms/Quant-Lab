import argparse
import pandas as pd
import numpy as np
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Stitch individual futures contracts into a back-adjusted continuous series.")
    parser.add_argument("--input", required=True, help="Input raw Parquet with multiple contracts")
    parser.add_argument("--output", required=True, help="Output continuous Parquet file")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_parquet(args.input)
    except Exception as e:
        print(f"Error reading Parquet: {e}")
        sys.exit(1)

    print("Cleaning and filtering symbols...")
    # Keep only outright contracts (e.g. NQM1, NQZ4). Ignore spreads like NQM1-NQU1 or NQ=F
    df = df[df['symbol'].str.match(r'^NQ[HMUZ]\d+$', na=False)].copy()
    
    # Identify datetime column
    datetime_cols = [c for c in df.columns if c.lower() in ['ts_event', 'ts', 'date', 'datetime', 'timestamp', 'time']]
    if not datetime_cols:
        print("Error: Could not find datetime column.")
        sys.exit(1)
        
    dt_col = datetime_cols[0]
    # Ensure it's parsed as datetime
    df[dt_col] = pd.to_datetime(df[dt_col])
    
    # Sort chronologically by time and then symbol
    df.sort_values([dt_col, 'symbol'], inplace=True)
    
    # Extract just the date for daily aggregation
    # Note: Using the timestamp directly. For accurate daily grouping, we might convert to US/Eastern first
    # if it's currently UTC. The input file seems to be UTC ("...000Z"). 
    # Let's convert to US/Eastern to match trading days
    
    if df[dt_col].dt.tz is None:
        df[dt_col] = df[dt_col].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    else:
        df[dt_col] = df[dt_col].dt.tz_convert('US/Eastern')
        
    df['date_only'] = df[dt_col].dt.date
    
    print("Determining active contracts by daily volume...")
    # Group by date and symbol to get daily volume
    daily_vol = df.groupby(['date_only', 'symbol'])['volume'].sum().reset_index()
    
    # Find the symbol with max volume for each day
    idx = daily_vol.groupby('date_only')['volume'].idxmax()
    active_contracts = daily_vol.loc[idx, ['date_only', 'symbol']].set_index('date_only')
    
    active_series = active_contracts['symbol']
    
    # We want to enforce forward rollover only, preventing flapping back and forth.
    # We can do this by taking a cumulative maximum on contract year/month, but for simplicity
    # and typical clean data, finding transition dates is usually sufficient.
    # To be robust against 1-day volume anomalies, we can forward-fill the active contract once it rolls.
    
    transitions = active_series[active_series != active_series.shift(1)]
    print(f"Detected {len(transitions)} potential contract periods.")
    
    # Clean up flapping (if it rolls to B, then back to A for a day, we ignore the roll back to A)
    # Since symbols sort roughly chronologically (NQM1 -> NQU1 -> NQZ1), we can filter.
    # A robust way is to just accept the first time a new contract becomes active.
    
    clean_transitions = []
    seen_symbols = set()
    for date, sym in transitions.items():
        if sym not in seen_symbols:
            clean_transitions.append((date, sym))
            seen_symbols.add(sym)
            
    # Rebuild the active_series cleanly based on transitions
    clean_active_series = pd.Series(index=active_series.index, dtype='object')
    for i in range(len(clean_transitions)):
        start_date, sym = clean_transitions[i]
        if i < len(clean_transitions) - 1:
            end_date = clean_transitions[i+1][0]
            clean_active_series.loc[start_date:end_date] = sym
        else:
            clean_active_series.loc[start_date:] = sym
            
    # Forward fill to handle any gaps
    clean_active_series.ffill(inplace=True)
    
    print("Final Rollover Schedule:")
    for date, sym in clean_transitions:
        print(f"  {date}: Rolled to {sym}")
        
    print("Stitching 1-minute data...")
    # Add active symbol info to main df
    df['symbol_active'] = df['date_only'].map(clean_active_series)
    
    # Filter to only keep rows where the symbol matches the active symbol for that day
    continuous_df = df[df['symbol'] == df['symbol_active']].copy()
    
    print("Calculating back-adjustments...")
    adjustment = 0.0
    # Create a series of adjustments mapping from date_only to adjustment value
    adjustments_series = pd.Series(0.0, index=continuous_df['date_only'].unique())
    
    # Iterate backwards through transition dates to accumulate adjustments
    for i in range(len(clean_transitions) - 1, 0, -1):
        curr_date, new_sym = clean_transitions[i]
        prev_date, old_sym = clean_transitions[i-1]
        
        # Find the gap on the transition date (first available overlapping minute)
        overlap_day_data = df[df['date_only'] == curr_date]
        
        old_data = overlap_day_data[overlap_day_data['symbol'] == old_sym]
        new_data = overlap_day_data[overlap_day_data['symbol'] == new_sym]
        
        if not old_data.empty and not new_data.empty:
            # Match the first common timestamp
            merged = pd.merge(old_data, new_data, on=dt_col, suffixes=('_old', '_new'))
            if not merged.empty:
                # Use the open price of the first overlapping minute
                old_price = merged.iloc[0]['open_old']
                new_price = merged.iloc[0]['open_new']
                gap = new_price - old_price
                print(f"  Gap on {curr_date} ({old_sym} -> {new_sym}): {gap:.2f}")
            else:
                # No exact minute overlap, use first price of the day
                old_price = old_data.iloc[0]['open']
                new_price = new_data.iloc[0]['open']
                gap = new_price - old_price
                print(f"  Gap on {curr_date} ({old_sym} -> {new_sym}) [no precise minute overlap]: {gap:.2f}")
        else:
            print(f"  Warning: No overlap on {curr_date} between {old_sym} and {new_sym}. Using 0 gap.")
            gap = 0.0
            
        adjustment += gap
        
        # Apply this accumulated adjustment to all dates in the old_sym period
        # The old period is from prev_date to curr_date (exclusive)
        period_mask = (adjustments_series.index >= prev_date) & (adjustments_series.index < curr_date)
        adjustments_series[period_mask] = adjustment
        
    # The very first period also needs the final accumulated adjustment (if there are days before the first transition)
    first_date = clean_transitions[0][0]
    period_mask = (adjustments_series.index < first_date)
    adjustments_series[period_mask] = adjustment
    
    # Map the adjustments back to the continuous dataframe
    continuous_df['adjustment'] = continuous_df['date_only'].map(adjustments_series)
    
    print("Applying cumulative back-adjustments to prices...")
    for col in ['open', 'high', 'low', 'close']:
        continuous_df[col] = continuous_df[col] + continuous_df['adjustment']
        
    # Clean up temporary columns
    continuous_df.drop(columns=['date_only', 'symbol_active', 'adjustment'], inplace=True)
    
    # Sort just in case
    continuous_df.sort_values(dt_col, inplace=True)
    
    print(f"Saving continuous back-adjusted series to {args.output}...")
    continuous_df.to_parquet(args.output, index=False)
    print(f"Successfully generated {len(continuous_df)} continuous 1-minute bars!")

if __name__ == "__main__":
    main()
