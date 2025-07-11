import json
import numpy as np
from pathlib import Path

LEAP_YEARS = ["2008", "2012", "2016", "2020"]


def get_hourly_averages(results_dict, out_dir=None, verbose_output=False, store_results=False):
    """Calculate average hourly values across multiple years for each plant"""
    
    averages = {}

    if store_results:
        output_path = Path(out_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    for plant_id, plant_data in results_dict.items():
        years = sorted(plant_data.keys())
        
        if verbose_output:
            print(f"Processing plant {plant_id} with {len(years)} years: {years}")
        
        first_year = years[0]
        first_year_values = plant_data[first_year]
        
        # Handle leap year adjustment for the first year
        if first_year in LEAP_YEARS:
            first_year_values = first_year_values[:-24]  # Remove last day for leap years
            
        sum_arr = np.zeros_like(first_year_values, dtype=float)

        for year in years:
            values = plant_data[year]
            
            # Handle leap year adjustment
            if year in LEAP_YEARS:
                values = values[:-24]  # Remove last day for leap years
                if verbose_output:
                    print(f"  Adjusted leap year {year}: removed last 24 hours")
            
            sum_arr += np.array(values, dtype=float)

        # Calculate averages
        avg_values = (sum_arr / len(years)).tolist()
        averages[plant_id] = avg_values

        if store_results:
            out_path = Path(out_dir) / f"{plant_id}_avg.json"
            with open(out_path, "w") as out_file:
                json.dump({plant_id: avg_values}, out_file, indent=4)
            if verbose_output:
                print(f"Average values for {plant_id} written to {out_path}")
        elif verbose_output:
            print(f"Average values for {plant_id} computed (not saved).")

    return averages