import pandas as pd
from pathlib import Path

# Default aggregation functions for known column types
DEFAULT_AGG_FUNCS = {
    'PV(W)': 'sum',
    'SOC(%)': 'mean',
    'Battery(W)': 'sum',
    'Meter(W)': 'sum',
    'Load(W)': 'sum',
    'MPP1(A)': 'mean',
    'MPP2(A)': 'mean',
    'MPP1(V)': 'mean',
    'MPP2(V)': 'mean',
    'IT1(°C)': 'mean'
}


def aggregate_logs(raw_path, agg_path=None, verbose_output=False, store_results=True):
    """Aggregate CSV log files from hourly operational data"""
    
    raw_path = Path(raw_path)
    csv_files = list(raw_path.glob("*.csv"))
    
    if verbose_output:
        print(f"Found {len(csv_files)} CSV files to process in {raw_path}")
    
    results = {}

    if store_results:
        agg_path = Path(agg_path)
        agg_path.mkdir(parents=True, exist_ok=True)
        if verbose_output:
            print(f"Output directory ready: {agg_path}")

    # Process each CSV file
    for csv_file in csv_files:
        if verbose_output:
            print(f"Processing: {csv_file.name}")
        
        # Read CSV file with datetime index
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        # Create aggregation functions based on available columns
        agg_funcs = {}
        for col in df.columns:
            if col in DEFAULT_AGG_FUNCS:
                agg_funcs[col] = DEFAULT_AGG_FUNCS[col]
            else:
                # Default to mean for unknown columns
                agg_funcs[col] = 'mean'
                if verbose_output:
                    print(f"  Unknown column '{col}' in {csv_file.name}, using 'mean' aggregation")

        # Aggregate the data by hour
        df_hourly = df.resample('1h').agg(agg_funcs)
        
        plant_id = csv_file.stem
        results[plant_id] = df_hourly

        # Save the aggregated data to a new CSV file
        if store_results:
            out_file = agg_path / f'{plant_id}_agg-1h.csv'
            df_hourly.to_csv(out_file)
            if verbose_output:
                print(f"  Aggregated data saved to: {out_file}")
        elif verbose_output:
            print(f"  Aggregated data for {csv_file.name} computed (not saved).")

    if verbose_output:
        print(f"Processing complete. Successfully processed {len(results)} files.")

    if store_results:
        return None
    else:
        return results
