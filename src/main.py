import argparse
import json
import shutil
import pandas as pd
from pathlib import Path

from pvgis_api_seriescalc import pvgis_request
from extract_hourly_data import extract_hourly_data
from get_hourly_avg import get_hourly_averages
from aggregate_csv import aggregate_logs
from compute_differences import compute_differences
from create_feature_vectors import create_shading_features, create_pollution_features
from create_clusters import create_shading_clusters, create_pollution_clusters
from pollution_learning import train_pollution_model, load_pollution_model, predict_pollution
from shading_learning import train_shading_model, load_shading_model, predict_shading


def main():
    parser = argparse.ArgumentParser(description="PV Diagnostic Tool")
    parser.add_argument("--input", type=Path, required=True, help="Path to input (dev_data directory for training, single CSV for prediction)")
    parser.add_argument("--output", type=Path, default=Path("model"), help="Path to save results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--store", action="store_true", help="Store intermediate results")
    parser.add_argument("--predict", action="store_true", help="Switch to prediction mode (classify single plant)")
    parser.add_argument("--models", type=Path, help="Path to directory containing trained models (required for --predict)")
    parser.add_argument("--metadata", type=Path, help="Path to metadata CSV file (required for --predict)")
    parser.add_argument("--plots", action="store_true", help="Generate clustering visualization plots (training mode only)")
    
    args = parser.parse_args()
    
    if args.predict:
        if not args.models:
            parser.error("--models is required when using --predict")
        if not args.metadata:
            parser.error("--metadata is required when using --predict")
        predict_mode(args)
    else:
        train_mode(args)


def train_mode(args):
    # Create output directory
    args.output.mkdir(exist_ok=True)
    
    print("=== PV Diagnostic Tool - Training Mode ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    
    # Step 1: PVGIS data
    print("\n1. Getting PVGIS data...")
    seriescalc_values = pvgis_request(metadata_csv=args.input / "goodwe_plant_metadata.csv", result_dir=args.output / "pvgis_results", verbose_output=args.verbose, store_results=args.store)

    # Step 2: Extract hourly data
    print("2. Extracting hourly data...")
    hourly_data = extract_hourly_data(json_list=seriescalc_values, output_dir=args.output / "hourly_data", verbose_output=args.verbose, store_results=args.store)
    
    # Step 3: Get hourly averages
    print("3. Computing hourly averages...")
    hourly_averages = get_hourly_averages(results_dict=hourly_data, out_dir=args.output / "hourly_averages", verbose_output=args.verbose, store_results=args.store)
    
    # Step 4: Aggregate operational data
    print("4. Aggregating operational data...")
    aggregated_data = aggregate_logs(raw_path=args.input / "goodwe_operational-data", agg_path=args.output / "aggregated_data", verbose_output=args.verbose, store_results=args.store)

    # Step 5: Compute differences
    print("5. Computing differences...")
    differences = compute_differences(hourly_averages=hourly_averages, agg_data=aggregated_data, out_dir=args.output / "differences", verbose_output=args.verbose, store_results=args.store)

    # Step 6: Create feature vectors
    print("6. Creating feature vectors...")
    shading_features = create_shading_features(differences_dict=differences, out_dir=args.output / "shading_fvs", verbose_output=args.verbose, store_results=args.store)
    pollution_features = create_pollution_features(differences_dict=differences, out_dir=args.output / "pollution_fvs", verbose_output=args.verbose, store_results=args.store)

    # Step 7: Clustering
    print("7. Clustering...")
    shading_clusters = create_shading_clusters(feature_vectors=shading_features, out_dir=args.output / "shading_clusters", verbose_output=args.verbose, store_results=args.store, create_plots=args.plots)
    pollution_clusters = create_pollution_clusters(feature_vectors=pollution_features, out_dir=args.output / "pollution_clusters", verbose_output=args.verbose, store_results=args.store, create_plots=args.plots)

    # Step 8: Train models
    print("8. Training models...")
    train_pollution_model(cluster_results=pollution_clusters, model_path=args.output / "pollution_model.pth")
    train_shading_model(cluster_results=shading_clusters, model_path=args.output / "shading_model.pth")

    print("\n=== Training Complete ===")
    print(f"Models saved to: {args.output}")


def predict_mode(args):
    print("=== PV Diagnostic Tool - Prediction Mode ===")
    print(f"Input CSV: {args.input}")
    print(f"Models: {args.models}")
    print(f"Metadata: {args.metadata}")
    
    # Step 1: PVGIS data for single plant
    print("\n1. Getting PVGIS data...")
    seriescalc_values = pvgis_request(metadata_csv=args.metadata, result_dir=args.output / "pvgis_results", verbose_output=args.verbose, store_results=args.store)

    # Step 2: Extract hourly data
    print("2. Extracting hourly data...")
    hourly_data = extract_hourly_data(json_list=seriescalc_values, output_dir=args.output / "hourly_data", verbose_output=args.verbose, store_results=args.store)
    
    # Step 3: Get hourly averages
    print("3. Computing hourly averages...")
    hourly_averages = get_hourly_averages(results_dict=hourly_data, out_dir=args.output / "hourly_averages", verbose_output=args.verbose, store_results=args.store)
    
    # Step 4: Process single CSV file
    print("4. Processing operational data...")
    # Create temporary directory structure for single file
    temp_dir = args.output / "temp_operational"
    temp_dir.mkdir(exist_ok=True)
    
    # Copy single CSV to expected structure
    plant_id = args.input.stem  # Get filename without extension
    shutil.copy(args.input, temp_dir / f"{plant_id}.csv")
    
    aggregated_data = aggregate_logs(raw_path=temp_dir, agg_path=args.output / "aggregated_data", verbose_output=args.verbose, store_results=args.store)

    # Step 5: Compute differences
    print("5. Computing differences...")
    differences = compute_differences(hourly_averages=hourly_averages, agg_data=aggregated_data, out_dir=args.output / "differences", verbose_output=args.verbose, store_results=args.store)

    # Step 6: Create feature vectors
    print("6. Creating feature vectors...")
    shading_features = create_shading_features(differences_dict=differences, out_dir=args.output / "shading_fvs", verbose_output=args.verbose, store_results=args.store)
    pollution_features = create_pollution_features(differences_dict=differences, out_dir=args.output / "pollution_fvs", verbose_output=args.verbose, store_results=args.store)

    # Step 7: Load models and predict
    print("7. Loading models and predicting...")
    pollution_model = load_pollution_model(args.models / "pollution_model.pth")
    shading_model = load_shading_model(args.models / "shading_model.pth")
    
    # Get predictions for each day
    pollution_predictions = predict_pollution(pollution_model, pollution_features)
    shading_predictions = predict_shading(shading_model, shading_features)
    
    # Step 8: Create output JSON
    print("8. Creating output...")
    output_data = create_prediction_output(
        plant_id=plant_id,
        metadata_csv=args.metadata,
        pollution_predictions=pollution_predictions,
        shading_predictions=shading_predictions,
        differences=differences
    )
    
    # Save results
    output_file = args.output / f"{plant_id}_predictions.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n=== Prediction Complete ===")
    print(f"Results saved to: {output_file}")


def create_prediction_output(plant_id, metadata_csv, pollution_predictions, shading_predictions, differences):
    
    # Read metadata for plant info
    metadata = pd.read_csv(metadata_csv)
    plant_info = metadata.iloc[0].to_dict()  # Assuming single plant metadata
    
    output = {
        "plant_id": plant_id,
        "metadata": {
            "latitude": plant_info.get("latitude", None),
            "longitude": plant_info.get("longitude", None),
            "kw_peak": plant_info.get("kw_peak", None)
        },
        "daily_classifications": {}
    }
    
    # Combine predictions by date
    for plant_pred in pollution_predictions.values():
        for date_str, pollution_class in plant_pred.items():
            output["daily_classifications"][date_str] = {
                "pollution": pollution_class,  # 0=no, 1=yes, -1=error
                "shading": -1  # Default to error if no prediction
            }
    
    for plant_pred in shading_predictions.values():
        for date_str, shading_class in plant_pred.items():
            if date_str in output["daily_classifications"]:
                output["daily_classifications"][date_str]["shading"] = shading_class
            else:
                output["daily_classifications"][date_str] = {
                    "pollution": -1,  # Default to error if no prediction
                    "shading": shading_class
                }
    
    return output


if __name__ == "__main__":
    main()