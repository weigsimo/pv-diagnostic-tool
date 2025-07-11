import pandas as pd
from pathlib import Path


def compute_differences(hourly_averages, agg_data, out_dir=None, verbose_output=False, store_results=True):
    """Compute differences between PVGIS reference hourly averages and actual aggregated operational data"""
    
    results = {}

    if store_results:
        output_path = Path(out_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if verbose_output:
            print(f"Output directory ready: {output_path}")

    for plant_id, avg_values in hourly_averages.items():
        if verbose_output:
            print(f"Processing plant: {plant_id}")

        # Prepare operational data DataFrame
        df = agg_data[plant_id].copy()
        
        # Handle various timestamp column formats
        if 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)
        if 'Timestamp' not in df.columns:
            df.reset_index(inplace=True)
            df.rename(columns={df.columns[0]: 'Timestamp'}, inplace=True)
        
        # Ensure timestamp is datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Create DataFrame for reference values (8760 hours = 365 days × 24 hours)
        reference_df = pd.DataFrame(avg_values, columns=['AVG'])
        timestamps = pd.date_range(start='2023-01-01 00:00:00', periods=8760, freq='H')
        reference_df['Timestamp'] = timestamps
        reference_df = reference_df[['Timestamp', 'AVG']]

        # Extract month, day, hour for matching
        for frame in [df, reference_df]:
            frame['month'] = frame['Timestamp'].dt.month
            frame['day'] = frame['Timestamp'].dt.day
            frame['hour'] = frame['Timestamp'].dt.hour

        # Handle leap year: map February 29th to February 28th
        feb_29_mask = (df['month'] == 2) & (df['day'] == 29)
        if feb_29_mask.any():
            df.loc[feb_29_mask, 'day'] = 28
            if verbose_output:
                print(f"  Mapped {feb_29_mask.sum()} Feb 29th entries to Feb 28th for {plant_id}")

        # Merge operational data with reference averages
        merged_df = df.merge(reference_df, on=['month', 'day', 'hour'], suffixes=('', '_ref'))

        # Clean up helper columns
        columns_to_drop = ['month', 'day', 'hour']
        if 'Timestamp_ref' in merged_df.columns:
            columns_to_drop.append('Timestamp_ref')
        merged_df = merged_df.drop(columns=columns_to_drop)

        # Compute differences (positive = underperforming, negative = overperforming)
        merged_df['Difference'] = merged_df['AVG'] - merged_df['PV(W)']
        
        # Add some basic statistics
        if verbose_output:
            mean_diff = merged_df['Difference'].mean()
            std_diff = merged_df['Difference'].std()
            print(f"  Mean difference: {mean_diff:.2f}W, Std: {std_diff:.2f}W")

        results[plant_id] = merged_df

        # Save results if requested
        if store_results:
            out_path = Path(out_dir) / f"{plant_id}_differences.csv"
            merged_df.to_csv(out_path, index=False)
            if verbose_output:
                print(f"  Differences saved to: {out_path}")
        elif verbose_output:
            print(f"  Differences computed for {plant_id} (not saved).")

    if verbose_output:
        print(f"Processing complete. Successfully processed {len(results)} plants.")

    return results
