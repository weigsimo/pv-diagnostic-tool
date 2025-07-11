import json
from pathlib import Path


def extract_hourly_data(json_list, output_dir=None, verbose_output=False, store_results=False):
    """Extract hourly power values from PVGIS API response data"""
    
    results = {}

    if store_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    for data in json_list:
        # Extract plant_id and year from the JSON structure
        plant_id = data["inputs"].get("plant_id", None)
        year = str(data["inputs"]["year"]) if "year" in data["inputs"] else None

        # Fallback: Try to extract from meta if not present in inputs
        if plant_id is None and "meta" in data and "plant_id" in data["meta"]:
            plant_id = data["meta"]["plant_id"]
        if year is None and "meta" in data and "year" in data["meta"]:
            year = str(data["meta"]["year"])

        if plant_id is None or year is None:
            if verbose_output:
                print("Warning: Could not determine plant_id or year for one entry, skipping.")
            continue

        if verbose_output:
            print(f"Processing plant: {plant_id}, year: {year}")

        # Extract hourly values
        hourly_vals = []
        for h in data["outputs"]["hourly"]:
            hourly_vals.append(h["P"])
        
        if plant_id not in results:
            results[plant_id] = {}
        results[plant_id][year] = hourly_vals

        if verbose_output:
            print(f"  Extracted {len(hourly_vals)} hourly values.")

    # Save the extracted hourly values to a new JSON file for each plant
    if store_results:
        for plant_id, plant_vals in results.items():
            output_file = Path(output_dir) / f"{plant_id}.json"
            with open(output_file, 'w') as f:
                json.dump({plant_id: plant_vals}, f, indent=4)
            if verbose_output:
                print(f"Hourly data for {plant_id} extracted and saved to {output_file}")

    return results
