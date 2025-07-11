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
python main.py --input ../dev_data --output ../model --verbose --store --plots
```

**Arguments:**

- `--input`: Path to directory containing training data
- `--output`: Directory to save trained models
- `--verbose`: Enable detailed logging
- `--store`: Save intermediate processing results
- `--plots`: Generate clustering visualization plots (training mode only)
  - Creates PCA scatter plots showing HDBSCAN (shading) and GMM (pollution) clustering results
  - Plots are saved as PNG files in the respective cluster output directories
  - Helps visualize how well the clustering algorithms separate different plant behaviors

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
- Displays comprehensive performance metrics (precision, recall, F1-score, confusion matrix)
- Optional intermediate results (feature vectors, clusters, etc.)
- With `--plots`: PCA clustering visualizations (`shading_clusters_pca.png`, `pollution_clusters_pca.png`)

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

# Train models with clustering visualization plots
cd src
python main.py --input ../dev_data --output ../models --verbose --plots

# Results: pollution_model.pth and shading_model.pth saved to ../models/
# With --plots: PCA cluster visualizations saved to ../models/shading_clusters/ and ../models/pollution_clusters/
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

## License

This project is licensed under the MIT License.

**Copyright 2025 Simon Maximilian Weigl**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
