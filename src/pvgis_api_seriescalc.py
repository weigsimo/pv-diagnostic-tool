import csv
import json
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

PVGIS_API_SERIESCALC = "https://re.jrc.ec.europa.eu/api/seriescalc"
REQUEST_DELAY = 0.5
REQUEST_TIMEOUT = 30  # seconds
DEFAULT_LOSS = 14
DEFAULT_ANGLE = 35
DEFAULT_ASPECT = 0  # azimuth (facing south)
YEAR_RANGE_START = 2007
YEAR_RANGE_END = 2024

def pvgis_request(metadata_csv, result_dir=None, store_results=False, verbose_output=False):
    """Query PVGIS API for all plants in metadata_csv"""
    
    metadata_path = Path(metadata_csv)
    
    # Create result directory if storing results
    if store_results and result_dir:
        result_path = Path(result_dir)
        result_path.mkdir(parents=True, exist_ok=True)
        if verbose_output:
            print(f"Directory {result_path} created.")

    # --- Read metadata CSV and prepare data ---
    with open(metadata_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        plants = []

        for row in reader:
            plant = {
                "plant_id": row["plant_id"],
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "peakpower": float(row["kw_peak"]),
                "loss": DEFAULT_LOSS,
                "angle": DEFAULT_ANGLE,
                "aspect": DEFAULT_ASPECT,
                "has_battery": row["has_battery"].lower() == "true",
                "battery_capacity": float(row["battery_capacity"]) if row["battery_capacity"] else 0.0,
                "installation_date": datetime.strptime(row["installation_date"], "%Y-%m-%d").date(),
                "has_pv": row["has_pv"].lower() == "true"
            }
            plants.append(plant)

    # --- Query PVGIS API and save results ---
    results = []
    for plant in plants:
        
        if verbose_output:
            print(f"Processing plant: {plant['plant_id']}")
        
        if store_results:
            plant_dir = Path(result_dir) / plant["plant_id"]
            plant_dir.mkdir(parents=True, exist_ok=True)
        
        # Get data for every available year
        for year in range(YEAR_RANGE_START, YEAR_RANGE_END):
        
            params = {
                "lat": plant["lat"],
                "lon": plant["lon"],
                "peakpower": plant["peakpower"],
                "loss": plant["loss"],
                "usehorizon": 1,  # Verschattung durch Horizont berücksichtigen
                "startyear": year, 
                "endyear": year,
                "pvcalculation": 1,  # PV-Leistung zusätzlich zu Einstrahlung berechnen
                "angle": plant["angle"], 
                "aspect": plant["aspect"], 
                "mountingplace": "free",  # Freistehend (opt.)
                "outputformat": "json"
            }

            if verbose_output:
                print(f"Requesting data for {plant['plant_id']} for year {year}...")
            
            response = requests.get(
                PVGIS_API_SERIESCALC, 
                params=params, 
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()

            result = response.json()

            # Add plant_id and year to result for downstream processing
            result.setdefault("inputs", {})["plant_id"] = plant["plant_id"]
            result["inputs"]["year"] = year

            if store_results:
                file_name = Path(result_dir) / plant["plant_id"] / f"{plant['plant_id']}_{year}.json"
                with open(file_name, "w") as f:
                    json.dump(result, f, indent=4)
                if verbose_output:
                    print(f"Saved: {file_name}")
            else:
                results.append(result)
                if verbose_output:
                    print(f"Data for {plant['plant_id']} for year {year} processed successfully.")
        
    if store_results:
        if verbose_output:
            print("All data processed and saved to disk.")
        return None
    else:
        return results
