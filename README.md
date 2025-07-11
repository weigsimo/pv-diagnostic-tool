# PV Diagnostic Tool

An AI-powered tool for automatic detection of shading and pollution/soiling issues in photovoltaic (PV) solar installations. This tool uses machine learning to analyze operational data and identify performance degradation patterns.

## Features

- **Dual Detection**: Identifies both shading events and pollution/soiling issues
- **Two Operating Modes**: Training mode for model development and prediction mode for real-time classification
- **PVGIS Integration**: Automatically fetches theoretical solar irradiance data for comparison
- **Neural Network Models**: Uses MLP (Multi-Layer Perceptron) networks for binary classification
- **Clustering Analysis**: HDBSCAN for shading anomaly detection and Gaussian Mixture Models for pollution state modeling
- **JSON Output**: Standardized daily classification results with metadata

## Architecture

The tool implements a complete machine learning pipeline:

1. **Data Preprocessing**: PVGIS API integration → hourly data extraction → operational data aggregation
2. **Feature Engineering**: Creates specialized feature vectors for shading and pollution detection
3. **Clustering**: Unsupervised learning to identify patterns in the data
4. **Supervised Learning**: Neural networks trained on clustered data for classification
5. **Prediction**: Daily classification of new plant data

## Installation

### Prerequisites

- Python 3.8+
- Required packages: `torch`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `hdbscan`

### Setup

```bash
git clone https://github.com/weigsimo/pv-diagnostic-tool.git
cd pv-diagnostic-tool
pip install -r requirements.txt
```

## Usage

The tool supports two main operating modes:

### Training Mode

Train models on a dataset of PV plants:

```bash
cd src
python main.py --input ../dev_data --output ../model --verbose --store
```

**Arguments:**

- `--input`: Path to directory containing training data
- `--output`: Directory to save trained models
- `--verbose`: Enable detailed logging
- `--store`: Save intermediate processing results

**Required Input Structure:**

```
dev_data/
├── goodwe_plant_metadata.csv      # Plant metadata (lat, lon, kw_peak, etc.)
└── goodwe_operational-data/       # Operational CSV files
    ├── gw-0001.csv
    ├── gw-0002.csv
    └── ...
```

### Prediction Mode

Classify a single plant using trained models:

```bash
cd src
python main.py --predict \
    --input gw-0001.csv \
    --models ../model \
    --metadata single_plant_metadata.csv \
    --output predictions \
    --verbose
```

**Arguments:**

- `--predict`: Switch to prediction mode
- `--input`: Single CSV file with operational data
- `--models`: Directory containing trained models (`pollution_model.pth`, `shading_model.pth`)
- `--metadata`: CSV file with plant metadata for PVGIS queries
- `--output`: Directory to save prediction results

## Data Format

### Plant Metadata CSV

```csv
plant_id,latitude,longitude,kw_peak,has_battery,battery_capacity,installation_date,has_pv
gw-0001,48.8046,11.3775,5.74,True,7.7,2023-04-01,True
```

### Operational Data CSV

```csv
Timestamp,PV(W),Battery(W),SOC(%),Load(W),MPP1(A),MPP2(A),MPP1(V),MPP2(V)
2023-01-01 08:00:00,1250.5,150.2,85.3,300.1,5.2,4.8,240.1,238.9
```

## Output Format

### Training Mode

- Saves trained PyTorch models: `pollution_model.pth`, `shading_model.pth`
- Optional intermediate results (feature vectors, clusters, etc.)

### Prediction Mode

Generates JSON output with daily classifications:

```json
{
  "plant_id": "gw-0001",
  "metadata": {
    "latitude": 48.8046,
    "longitude": 11.3775,
    "kw_peak": 5.74
  },
  "daily_classifications": {
    "2023-01-01": {
      "pollution": 0,
      "shading": 1
    },
    "2023-01-02": {
      "pollution": 1,
      "shading": 0
    }
  }
}
```

**Classification Values:**

- `0`: No issue detected
- `1`: Issue detected (pollution/shading)
- `-1`: Prediction error or unavailable

## Technical Details

### Models

- **Pollution Detection**: Gaussian Mixture Model clustering + MLP neural network
- **Shading Detection**: HDBSCAN clustering + MLP neural network
- **Features**: Daily aggregated power patterns, voltage/current variability, temporal characteristics

### Key Components

- `main.py`: Entry point and workflow orchestration
- `pollution_learning.py`: Pollution detection model training/prediction
- `shading_learning.py`: Shading detection model training/prediction
- `create_feature_vectors.py`: Feature engineering for both detection types
- `create_clusters.py`: Unsupervised clustering algorithms
- `pvgis_api_seriescalc.py`: PVGIS API integration for theoretical data

## Examples

### Quick Start - Training

```bash
# Train models on sample dataset
cd src
python main.py --input ../dev_data --output ../models --verbose

# Results: pollution_model.pth and shading_model.pth saved to ../models/
```

### Quick Start - Prediction

```bash
# Classify single plant
cd src
python main.py --predict \
    --input ../dev_data/goodwe_operational-data/gw-0001.csv \
    --models ../models \
    --metadata ../dev_data/goodwe_plant_metadata.csv \
    --output ../predictions

# Results: gw-0001_predictions.json saved to ../predictions/
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{pv_diagnostic_tool,
  title={PV Diagnostic Tool: AI-powered Shading and Pollution Detection},
  author={Simon Weigl},
  year={2025},
  url={https://github.com/weigsimo/pv-diagnostic-tool}
}
```

## Contact

Simon Weigl - simon.weigl@tum.de

Project Link: [https://github.com/weigsimo/pv-diagnostic-tool](https://github.com/weigsimo/pv-diagnostic-tool)
