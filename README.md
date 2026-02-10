# Causal Feature Tokenizer Transformer (CFT)

## Groundwater Level Prediction with Causally-Informed Deep Learning

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Training](#model-training)
7. [Available Models](#available-models)
8. [Evaluation and Visualization](#evaluation-and-visualization)
9. [Configuration](#configuration)
10. [Output Files](#output-files)
11. [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a novel **Causal Feature Tokenizer Transformer (CFT)** architecture for groundwater level prediction. The model integrates prior causal knowledge from causal discovery algorithms (LPCMCI) into the attention mechanism of a feature tokenizer transformer, enabling more interpretable and physically-consistent predictions.

The framework supports multiple model architectures for comparative studies, including:
- **Causal-FT-Transformer**: Primary model with integrated causal priors
- **FT-Transformer**: Feature Tokenizer baseline
- **Causal-Transformer**: Standard transformer with causal masking
- **Base Transformer**: Vanilla tabular transformer
- **LSTM**: Sequential baseline model

---

## Key Features

- **Causal-Informed Learning**: Incorporates LPCMCI-derived causal relationships as prior constraints
- **Feature Tokenization**:learnable token embeddings for numerical features
- **Location-Aware Modeling**: Optional geographic coordinate encoding
- **Hydrological Clustering**: K-means clustering based on environmental fingerprints
- **Comprehensive Evaluation**: RMSE, MAE, R² metrics with spatial and temporal visualizations
- **Modular Design**: Easy extension for new models and data sources

---

## Project Structure

```
CFT/
├── data/
│   ├── data_preprocessing.py    # GroundwaterDataPreprocessor class
│   ├── data_manager.py          # Data I/O management
│   └── run_data_preprocessing.py  # Preprocessing entry point
├── model/
│   ├── causal_ft_transformer.py # Causal-FT-Transformer architecture
│   ├── ft_transformer.py        # FT-Transformer baseline
│   ├── causal_transformer.py    # Causal standard transformer
│   ├── transformer.py           # Base tabular transformer
│   └── lstm.py                  # LSTM baseline
├── train/
│   ├── train_causal_ft_transformer.py
│   ├── train_ft_transformer.py
│   ├── train_causal_transformer.py
│   ├── train_transformer.py
│   └── train_lstm.py
├── train_model.py               # Unified training launcher
├── utils/
│   ├── evaluation_utils.py      # Metrics computation
│   └── visualization_utils.py   # Plotting functions
├── Datasets/
│   └── Australia(bgyh2).csv     # Raw data (Australian groundwater)
├── preprocessed/                # Preprocessed data storage
├── models/                      # Saved model checkpoints
├── results/                     # Evaluation results
└── visualizations/             # Generated figures
```

---

## Installation

### Requirements

```bash
Python >= 3.8
PyTorch >= 1.9.0
NumPy >= 1.19.0
Pandas >= 1.2.0
Scikit-learn >= 0.24.0
Matplotlib >= 3.4.0
Tigramite >= 5.0.0 (optional, for causal discovery)
Cartopy >= 0.20.0 (optional, for spatial visualization)
```

### Setup

```bash
# Clone or navigate to the project directory
cd CFT

# Install dependencies (optional but recommended)
pip install torch numpy pandas scikit-learn matplotlib
pip install tigramite  # For LPCMCI causal discovery
pip install cartopy    # For spatial visualizations
```

---

## Data Preprocessing

**IMPORTANT**: Data preprocessing MUST be performed before model training. This step normalizes features, performs hydrological clustering, and splits the dataset.

### Quick Start

```bash
# Navigate to project directory
cd CFT

# Run preprocessing with default configuration
python data/run_data_preprocessing.py
```

### What the Preprocessing Does

1. **Feature Normalization**
   - Groundwater features: Separate normalization for positive/negative values
   - Environmental features: MinMax scaling to [0,1]
   - Static features: Standard scaling

2. **Hydrological Clustering** (Optional)
   - K-means clustering based on environmental fingerprints
   - Features: temperature, precipitation, population, evapotranspiration, drought index, soil moisture, etc.
   - Silhouette score evaluation

3. **Dataset Splitting**
   - Training set: First ~80% of years
   - Validation set: Next ~10% of years
   - Test set: Remaining ~10% of years

4. **Output Files**
   - `preprocessed/*.pkl`: Normalized data matrices
   - `preprocessed/*_preprocessor.pkl`: Preprocessor state (scalers, cluster models)
   - `metadata/*_config.json`: Configuration metadata

### Customizing Preprocessing

Edit the `DEFAULT_CONFIG` dictionary in `data/run_data_preprocessing.py`:

```python
DEFAULT_CONFIG = {
    'data_file': get_data_path('Australia(bgyh2).csv'),
    'target_name': 'Groundwater_',
    'n_clusters': 5,           # 0 to disable clustering
    'train_split_ratio': 0.8,
    'val_split_ratio': 0.1,
    'save_results': True,
    'generate_visualizations': True
}
```

---

## Model Training

### Basic Usage

```bash
# From project root directory
python train_model.py --model <model_name>
```

### Examples

```bash
# Train the primary Causal-FT-Transformer model
python train_model.py --model causal_ft_transformer

# Train FT-Transformer baseline
python train_model.py --model ft_transformer

# Train LSTM baseline
python train_model.py --model lstm

# List all available models
python train_model.py --list
```

---

## Available Models

| Model | Description | Key Parameters |
|-------|-------------|-----------------|
| `causal_ft_transformer` | Causal Feature Tokenizer Transformer with LPCMCI priors | window_size, d_model, n_heads, causal_consistency_lambda |
| `ft_transformer` | FT-Transformer with location tokens | window_size, d_model, n_heads |
| `causal_transformer` | Standard Transformer with causal masking | window_size, d_model, n_heads |
| `transformer` | Base tabular transformer baseline | window_size, d_model, n_heads |
| `lstm` | LSTM baseline for time-series | window_size, hidden_size |

### Default Hyperparameters

```python
# Transformer models
{
    'd_model': 64,
    'n_heads': 4,
    'num_layers': 2,
    'dim_feedforward': 128,
    'dropout': 0.1,
    'window_size': 4,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
}

# LSTM model
{
    'window_size': 4,
    'hidden_size': 64,
    'dropout': 0.01,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 50
}
```

### Causal Consistency Loss

For `causal_ft_transformer`, a causal consistency loss term encourages learned attention weights to respect LPCMCI-derived causal relationships:

```python
causal_consistency_lambda = 0.1  # Weight for causal consistency loss
```

---

## Evaluation and Visualization

### Metrics Computed

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination

### Generated Visualizations

1. **Training Curves**: Loss and validation loss over epochs
2. **Prediction Scatter Plots**: True vs. Predicted values
3. **Spatial Maps**: Australia-wide prediction error distributions
4. **Causal Weight Comparisons**: LPCMCI priors vs. learned weights

### Saved to

```
visualizations/
├── causal_ft_transformer/
│   ├── training_history.png
│   ├── test_prediction_scatter.png
│   ├── test_spatial_error_map.png
│   └── causal_weight_comparison.png
├── ft_transformer/
├── lstm_baseline/
└── ...
```

---

## Configuration

### Environment Variables

No environment variables required. All paths are relative to the project root.

### Random Seed

All experiments use `RANDOM_SEED = 42` for reproducibility:

```python
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
```

---

## Output Files

### Model Checkpoints

```
models/
├── causal_ft_transformer/
│   ├── causal_ft_transformer.pth    # Model weights
│   └── causal_ft_transformer.pkl    # Full model state
├── ft_transformer/
├── causal_transformer/
└── ...
```

### Evaluation Results

```
results/
├── causal_ft_transformer/
│   ├── evaluation_results.json
│   ├── training_config.json
│   ├── causal_feature_importance_summary.csv
│   └── test_predictions.csv
├── lstm_baseline/
└── ...
```

### Preprocessed Data

```
preprocessed/
├── groundwater_YYYYMMDD_HHMMSS_preprocessed.pkl
├── groundwater_YYYYMMDD_HHMMSS_preprocessed_preprocessor.pkl
└── ...
```

---

## Troubleshooting

### Common Issues

**1. No preprocessed data found**
```bash
# Solution: Run preprocessing first
python data/run_data_preprocessing.py
```

**2. CUDA out of memory**
```python
# Reduce batch size in configuration
batch_size = 16  # Instead of 32
```

**3. Tigramite not available**
```bash
# Install tigramite for LPCMCI causal discovery
pip install tigramite

# The system will fall back to correlation-based causal weights
```

**4. Cartopy not available**
```bash
# Install cartopy for spatial visualizations
pip install cartopy

# Without cartopy, spatial maps will be skipped
```

### Checking Available Data

```python
from data.data_manager import DataManager

dm = DataManager()
available = dm.list_available_data()
print(available)
```

---

## Data Format

### Expected Input Format (CSV)

| Column | Description |
|-------|-------------|
| Site ID | Unique site identifier |
| Latitude | Decimal degrees |
| Longitude | Decimal degrees |
| Groundwater | Groundwater level  |
| precipitation | Annual precipitation |
| tasmax | Maximum temperature |
| tasmin | Minimum temperature |
| tas | Mean temperature |
| E | Evapotranspiration |
| pdsi | Palmer Drought Severity Index |
| q | Surface Runoff |
| soil | Soil moisture |
| population | Population density |


---

## Data Sources

### Explanatory Variables

Explanatory variables data can be downloaded from the following sources:

| Variable | Format | Resolution/Period | Source URL |
|----------|--------|-------------------|------------|
| population | Grid | 30arc-sec / (1990-2022) | [Zenodo](https://zenodo.org/records/11179644) |
| precipitation | nc | 1°×1° / Monthly (1991-2020) | [NOAA GPCP](https://downloads.psl.noaa.gov/Datasets/gpcp/precip.mon.ltm.1991-2020.nc) |
| tasmax | nc | 2.0°×2.5° / Monthly (2001-2020) | [NASA GISS](https://dpesgf03.nccs.nasa.gov/thredds/fileServer/CMIP6/DAMIP/NASA-GISS/GISS-E2-1-G/hist-sol/r1i1p1f2/Amon/tasmax/gn/v20181011/tasmax_Amon_GISS-E2-1-G_hist-sol_r1i1p1f2_gn_200101-202012.nc) |
| tasmin | nc | 2.0°×2.5° / Monthly (2001-2020) | [NASA GISS](https://dpesgf03.nccs.nasa.gov/thredds/fileServer/CMIP6/DAMIP/NASA-GISS/GISS-E2-1-G/hist-sol/r1i1p1f2/Amon/tasmin/gn/v20181011/tasmin_Amon_GISS-E2-1-G_hist-sol_r1i1p1f2_gn_200101-202012.nc) |
| tas | nc | 2.0°×2.5° / Monthly (2001-2020) | [NorCPM1 CMIP6](http://noresg.nird.sigma2.no/thredds/fileServer/esg_dataroot/cmor/CMIP6/CMIP/NCC/NorCPM1/historical-ext/r25i1p1f1/Amon/tas/gn/v20200724/tas_Amon_NorCPM1_historical-ext_r25i1p1f1_gn_201501-201812.nc) |
| E (Evapotranspiration) | nc | 0.1°×0.1° / Yearly (1980-2023) | [GLEAM](https://www.gleam.eu/) |
| pdsi | nc | 0.04°×0.04° / Yearly (1958-2024) | [TerraClimate](https://climate.northwestknowledge.net/TERRACLIMATE/index_directDownloads.php) |
| q (Surface Runoff) | nc | 0.04°×0.04° / Yearly (1958-2024) | [TerraClimate](https://climate.northwestknowledge.net/TERRACLIMATE/index_directDownloads.php) |
| soil | nc | 0.04°×0.04° / Yearly (1958-2024) | [TerraClimate](https://climate.northwestknowledge.net/TERRACLIMATE/index_directDownloads.php) |

### Main Dataset

The primary groundwater dataset (Australian groundwater levels with environmental features) can be downloaded from:

📥 **[Google Drive Download](https://download.example.com)** or **[Direct Link](https://drive.google.com/file/d/1VRwqiD6UlxHASB6IlMlIHK9L2Mqgh5AL/view?usp=sharing)**

**Instructions:**
1. Download the dataset from the link above
2. Place the file `Australia(bgyh2).csv` in the `Datasets/` folder:
   ```
   CFT/
   └── Datasets/
       └── Australia(bgyh2.csv)
   ```
3. Run data preprocessing:
   ```bash
   python data/run_data_preprocessing.py
   ```

**Note**: TerraClimate variables (pdsi, q, soil) share the same download page.

---

## Contact

For questions or issues, please open a GitHub issue or contact the repository maintainer.

**Email**: 2024218552@mail.hfut.edu.cn
