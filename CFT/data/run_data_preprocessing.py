# -*- coding: utf-8 -*-
"""
Main Data Preprocessing Script.

Functionality:
1. Load raw data
2. Execute data preprocessing (normalization, clustering, dataset splitting)
3. Save processing results
"""

import os
import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.data_preprocessing import GroundwaterDataPreprocessor
from data.data_manager import DataManager


# Get data file path (relative to project root or Datasets folder)
def get_data_path(filename):
    """Get absolute path for data files."""
    # First check if file exists in project root
    project_root_file = os.path.join(PROJECT_ROOT, filename)
    if os.path.exists(project_root_file):
        return project_root_file
    
    # Then check in Datasets folder
    datasets_folder = os.path.join(PROJECT_ROOT, 'Datasets')
    datasets_file = os.path.join(datasets_folder, filename)
    if os.path.exists(datasets_file):
        return datasets_file
    
    # Return default path (will trigger file not found error)
    return datasets_file


# ============== Default Configuration ==============
DEFAULT_CONFIG = {
    # Data file (checks project root first, then Datasets folder)
    'data_file': get_data_path('Australia(bgyh2).csv'),
    'target_name': 'Groundwater_',

    # Clustering settings
    'n_clusters': 0,  # 0 indicates clustering disabled

    # Dataset splitting
    'train_split_ratio': 0.8,
    'val_split_ratio': 0.1,

    # Output settings
    'save_results': True,
    'generate_visualizations': True
}


def validate_config(config):
    """Validate configuration parameters."""
    assert 0.5 <= config['train_split_ratio'] <= 0.9, "Training set ratio must be between 0.5-0.9"
    val_ratio = config.get('val_split_ratio', 0.1)
    assert 0.05 <= val_ratio <= 0.3, "Validation set ratio must be between 0.05-0.3"
    assert 1 - config['train_split_ratio'] - val_ratio >= 0.05, "Test set ratio too small"
    assert os.path.exists(config['data_file']), f"Data file not found: {config['data_file']}"
    print("Configuration validation passed")


def run_preprocessing(config=None):
    """
    Execute data preprocessing.

    Args:
        config: Configuration dictionary, uses default config if None

    Returns:
        dict: Processing results
    """
    config = config or DEFAULT_CONFIG.copy()

    print("=" * 60)
    print("Groundwater Data Preprocessing System")
    print("=" * 60)

    start_time = time.time()

    # 1. Initialize components
    data_manager = DataManager()
    preprocessor = GroundwaterDataPreprocessor(
        target_name=config['target_name'],
        n_clusters=config['n_clusters']
    )

    # 2. Load data
    print(f"\n[1/3] Loading data: {os.path.basename(config['data_file'])}")
    data = pd.read_csv(config['data_file'])
    print(f"  Data: {data.shape[0]} sites x {data.shape[1]} features")

    # 3. Data preprocessing
    print(f"\n[2/3] Data preprocessing...")
    preprocess_start = time.time()

    preprocessed_results = preprocessor.process_data(
        data,
        train_split_ratio=config['train_split_ratio'],
        val_split_ratio=config.get('val_split_ratio', 0.1)
    )
    preprocess_time = time.time() - preprocess_start

    print(f"  Complete: {preprocess_time:.2f}s | "
          f"{len(preprocessed_results['site_ids'])} sites, "
          f"{len(preprocessed_results['years'])} years")

    # 4. Save results
    if config.get('save_results', True):
        print(f"\n[3/3] Saving results...")
        saved_files = _save_results(
            data_manager, preprocessor, preprocessed_results, start_time
        )
    else:
        saved_files = {'preprocessed_results': preprocessed_results}

    # Print summary
    total_time = time.time() - start_time
    _print_summary(preprocessed_results, total_time)

    return {
        'preprocessed_results': preprocessed_results,
        'processing_time': total_time
    }


def _save_results(data_manager, preprocessor, preprocessed_results, start_time):
    """Save processing results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"groundwater_{timestamp}"

    saved = {}

    # Save preprocessed data
    preprocessed_filename = data_manager.save_preprocessed_data(
        preprocessed_results, preprocessor, f"{base_filename}_preprocessed"
    )
    saved['preprocessed_filename'] = preprocessed_filename

    # Save configuration
    config_filename = data_manager.create_processing_pipeline_config({
        'preprocessing_config': DEFAULT_CONFIG,
        'preprocessed_filename': saved.get('preprocessed_filename'),
        'processing_time': time.time() - start_time,
        'data_info': {
            'n_sites': len(preprocessed_results['site_ids']),
            'n_years': len(preprocessed_results['years']),
            'train_split_year': preprocessed_results['train_split_year'],
            'val_split_year': preprocessed_results.get('val_split_year')
        }
    }, f"{base_filename}_config.json")
    saved['config_filename'] = config_filename

    print(f"  Results saved: {base_filename}")
    return saved


def _print_summary(preprocessed_results, total_time):
    """Print processing summary."""
    print("\n" + "=" * 60)
    print(f"Processing complete! Time: {total_time:.2f} seconds")
    print(f"Sites: {len(preprocessed_results['site_ids'])}")
    print(f"Years: {len(preprocessed_results['years'])} "
          f"({preprocessed_results['years'][0]}-{preprocessed_results['years'][-1]})")
    print(f"Train/Val/Test: "
          f"{len(preprocessed_results.get('train_years', []))}/"
          f"{len(preprocessed_results.get('val_years', []))}/"
          f"{len(preprocessed_results.get('test_years', []))} years")
    print("=" * 60)


def load_processed_data(data_type='preprocessed', filename=None):
    """
    Load processed data.

    Args:
        data_type: 'preprocessed' or 'gridded'
        filename: Filename, None to list available files

    Returns:
        Processing results
    """
    data_manager = DataManager()

    if filename is None:
        available = data_manager.list_available_data()
        print("Available data files:")
        for dtype, files in available.items():
            if files:
                print(f"\n{dtype}:")
                for f in files[:5]:  # Display only first 5
                    info = data_manager.get_data_info(f, dtype)
                    if info:
                        print(f"  - {f} ({info.get('timestamp', 'unknown')[:10]})")
                if len(files) > 5:
                    print(f"  ... Total {len(files)} files")
        return None

    if data_type == 'preprocessed':
        return data_manager.load_preprocessed_data(filename)
    elif data_type == 'gridded':
        print("Note: Gridded data functionality has been removed, please use preprocessed data")
        return None
    else:
        print(f"Unsupported data type: {data_type}")
        return None


if __name__ == "__main__":
    # Run with default configuration
    results = run_preprocessing()

    # Example: Load processed data
    # load_processed_data('preprocessed')
