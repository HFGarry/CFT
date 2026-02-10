#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FT-Transformer Training Pipeline.

This module implements the training, evaluation, and inference pipeline
for the FT-Transformer baseline model.

Key Components:
    - TabularTransformerPredictor: End-to-end pipeline for training and evaluation
"""

import os
import sys
import time
import random
import math
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Optuna integration
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[Warning] Optuna not available.")

# Import from model folder
from model.ft_transformer import (
    SimpleFTTransformer,
    TabularDataset
)

from data.data_manager import DataManager
from utils.visualization_utils import (
    plot_training_history,
    generate_combined_australia_map,
    plot_prediction_comparison,
)
from utils.evaluation_utils import compute_evaluation_metrics


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")


class TabularTransformerPredictor:
    """FT-Transformer Baseline Predictor with Location Token Support."""

    def __init__(self, window_size=4, target_feature='Groundwater_', use_location_token=True):
        self.window_size = window_size
        self.target_feature = target_feature
        self.use_location_token = use_location_token
        self.model = None
        self.history = None
        self.preprocessed_data = None
        self.site_ids_list = None
        self.n_sites = 0
        self.scaler_info = None
        self.location_coords_train = None
        self.location_coords_val = None
        self.location_coords_test = None
        self.spatial_data = None  # Original spatial coordinates [lat, lon]

    def load_preprocessed_data(self, preprocessed_filename):
        data_manager = DataManager()
        self.preprocessed_data, preprocessor, metadata = data_manager.load_preprocessed_data(preprocessed_filename)
        print(f"[Data] Loaded {len(self.preprocessed_data['site_ids'])} sites")
        return True

    def create_sequences(self):
        """Construct time-window sequences with location coordinates."""
        print(f"[Data] Building sequences (window_size={self.window_size}, use_location_token={self.use_location_token})...")
        dynamic_data = self.preprocessed_data['dynamic_data_normalized']
        years = self.preprocessed_data['years']
        feature_names = self.preprocessed_data['dynamic_features']
        self.site_ids_list = self.preprocessed_data.get('site_ids', [])
        n_sites_total, n_years = dynamic_data.shape[0], len(years)

        # Extract spatial coordinates (Latitude, Longitude) for each site
        if self.use_location_token:
            spatial_data = self.preprocessed_data.get('spatial_data_normalized')
            if spatial_data is None:
                spatial_data = self.preprocessed_data.get('spatial_data_original')
            if spatial_data is not None and len(spatial_data) > 0:
                if isinstance(spatial_data, np.ndarray) and spatial_data.shape[1] >= 2:
                    self.spatial_data = spatial_data[:, :2].copy()
                    print(f"[Data] Location coordinates loaded: {self.spatial_data.shape}")
                else:
                    self.spatial_data = None
                    print("[Warning] Could not extract location coordinates from spatial_data")
            else:
                self.spatial_data = None
                print("[Warning] No spatial_data found in preprocessed data")
        else:
            self.spatial_data = None

        # Parse features by prefix and year
        feature_types = {}
        for idx, fname in enumerate(feature_names):
            match = re.search(r'^(.*?)_?(\d{4})$', fname)
            if match:
                prefix = match.group(1).rstrip('_')
                year = int(match.group(2))
            else:
                prefix = fname.rstrip('_')
                year = None
            feature_types.setdefault(prefix, []).append((year, idx))

        for prefix in feature_types:
            feature_types[prefix].sort(key=lambda x: x[0] if x[0] is not None else 0)

        n_feature_types = len(feature_types)
        data_by_year = np.zeros((n_years, n_sites_total, n_feature_types))

        for feat_idx, (prefix, year_col_list) in enumerate(feature_types.items()):
            is_static = all(y is None for y, _ in year_col_list)
            if is_static:
                col_idx = year_col_list[0][1]
                for y_idx in range(n_years):
                    data_by_year[y_idx, :, feat_idx] = dynamic_data[:, col_idx]
            else:
                for year, col_idx in year_col_list:
                    if year in years:
                        y_idx = years.index(year)
                        data_by_year[y_idx, :, feat_idx] = dynamic_data[:, col_idx]

        clean_target = self.target_feature.rstrip('_')
        target_prefix = next((p for p in feature_types if clean_target == p or clean_target in p), None)
        target_feat_idx = list(feature_types.keys()).index(target_prefix)
        self.target_feat_idx = target_feat_idx
        self.n_sites = n_sites_total

        # Split data by year (to prevent leakage)
        train_split_year = self.preprocessed_data['train_split_year']
        val_split_year = self.preprocessed_data.get('val_split_year', None)
        years_arr = np.array(years)

        # Determine dataset type for each year
        def get_dataset_type(year):
            if year <= train_split_year:
                return 'train'
            elif val_split_year and year <= val_split_year:
                return 'val'
            else:
                return 'test'

        # Create sequences ensuring no leakage across windows
        n_samples_per_site = n_years - self.window_size
        if n_samples_per_site <= 0:
            raise ValueError(f"[Error] Insufficient time span: years={n_years}, window={self.window_size}")

        valid_samples = []

        for i in range(n_samples_per_site):
            # Prediction year
            pred_year_idx = i + self.window_size
            pred_year = years[pred_year_idx]
            pred_dataset = get_dataset_type(pred_year)

            # Window years
            window_years = years[i:i+self.window_size]
            window_datasets = [get_dataset_type(y) for y in window_years]

            # Check for leakage
            is_valid = True
            if pred_dataset == 'train':
                if not all(ds == 'train' for ds in window_datasets):
                    is_valid = False
            elif pred_dataset == 'val':
                if any(ds == 'test' for ds in window_datasets):
                    is_valid = False

            if is_valid:
                valid_samples.append(i)

        if len(valid_samples) == 0:
            raise ValueError("[Error] No valid samples found.")

        # Build valid sequences
        X_windows = np.array([data_by_year[i: i + self.window_size] for i in valid_samples])
        y_targets = data_by_year[[i + self.window_size for i in valid_samples], :, target_feat_idx]

        X_flattened = X_windows.transpose(0, 2, 1, 3).reshape(-1, self.window_size, n_feature_types)
        y_flattened = y_targets.flatten()
        site_ids_flattened = np.tile(np.arange(n_sites_total), len(valid_samples))
        year_indices_flattened = np.repeat([i + self.window_size for i in valid_samples], n_sites_total)

        # Final split
        sample_years = years_arr[year_indices_flattened]

        train_mask = sample_years <= train_split_year
        if val_split_year:
            val_mask = (sample_years > train_split_year) & (sample_years <= val_split_year)
            test_mask = sample_years > val_split_year
        else:
            remaining = ~train_mask
            val_mask = np.zeros_like(train_mask, dtype=bool)
            test_mask = remaining

        self.X_train, self.site_train, self.y_train = X_flattened[train_mask], site_ids_flattened[train_mask], \
            y_flattened[train_mask]
        self.X_val, self.site_val, self.y_val = X_flattened[val_mask], site_ids_flattened[val_mask], y_flattened[val_mask]
        self.X_test, self.site_test, self.y_test = X_flattened[test_mask], site_ids_flattened[test_mask], y_flattened[test_mask]

        # Extract location coordinates for each sample based on site_ids
        if self.use_location_token and self.spatial_data is not None:
            site_to_coords = self.spatial_data
            self.location_coords_train = site_to_coords[self.site_train]
            self.location_coords_val = site_to_coords[self.site_val]
            self.location_coords_test = site_to_coords[self.site_test]
            print(f"  Location coords: Train={self.location_coords_train.shape}, Val={self.location_coords_val.shape}, Test={self.location_coords_test.shape}")
        else:
            self.location_coords_train = None
            self.location_coords_val = None
            self.location_coords_test = None

        self.scaler_info = {
            'target_feature': target_prefix,
            'normalization_metadata': self.preprocessed_data.get('normalization_metadata', {}),
            'feature_strategies': self.preprocessed_data.get('normalization_metadata', {}).get('feature_strategies', {})
        }
        print(f"[Data] Scaler info initialized with target: {target_prefix}")

        print(f"[Data] Sequences ready:")
        print(f"  Train: {len(self.X_train)} samples (Years: {sample_years[train_mask].min()}-{sample_years[train_mask].max()})")
        if len(sample_years[val_mask]) > 0:
            print(f"  Val: {len(self.X_val)} samples (Years: {sample_years[val_mask].min()}-{sample_years[val_mask].max()})")
        print(f"  Test: {len(self.X_test)} samples (Years: {sample_years[test_mask].min()}-{sample_years[test_mask].max()})")
        return True

    def build_model(self, d_model=64, n_heads=4, num_layers=2, dim_feedforward=128, dropout=0.1,
                    location_embed_dim=32):
        """Build the FT-Transformer model with optional location token."""
        n_features = self.X_train.shape[2]
        self.model = SimpleFTTransformer(
            window_size=self.window_size,
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_location_token=self.use_location_token,
            location_embed_dim=location_embed_dim
        ).to(device)
        print(f"[Model] Built FT-Transformer: d_model={d_model}, layers={num_layers}, use_location_token={self.use_location_token}")
        return True

    def train_model(self, epochs=100, batch_size=32, learning_rate=0.001, weight_decay=1e-5, use_callbacks=True,
                    callback_params=None, verbose=1):
        train_dataset = TabularDataset(self.X_train, self.site_train, self.y_train, self.location_coords_train)
        val_dataset = TabularDataset(self.X_val, self.site_val, self.y_val, self.location_coords_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = callback_params.get('early_stop_patience', 15) if callback_params else 15

        use_loc = self.use_location_token and self.location_coords_train is not None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                if use_loc:
                    X_b, _, y_b, loc_b = batch
                    loc_b = loc_b.to(device)
                else:
                    X_b, _, y_b = batch
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                if use_loc:
                    output = self.model(X_b, location_coords=loc_b)
                else:
                    output = self.model(X_b)
                loss = criterion(output, y_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_b.size(0)

            train_loss /= len(train_dataset)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if use_loc:
                        X_b, _, y_b, loc_b = batch
                        loc_b = loc_b.to(device)
                    else:
                        X_b, _, y_b = batch
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    if use_loc:
                        val_loss += criterion(self.model(X_b, location_coords=loc_b), y_b).item() * X_b.size(0)
                    else:
                        val_loss += criterion(self.model(X_b), y_b).item() * X_b.size(0)
            val_loss /= len(val_dataset)

            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)

            if verbose == 1 and (epoch % 5 == 0):
                print(f"Epoch {epoch} - loss: {train_loss:.5f} - val_loss: {val_loss:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_baseline_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            if use_callbacks and patience_counter >= early_stop_patience:
                break

        if os.path.exists('best_baseline_model.pth'):
            self.model.load_state_dict(torch.load('best_baseline_model.pth'))
            os.remove('best_baseline_model.pth')
        self.history = history
        return True

    def predict(self, X, site_ids, location_coords=None):
        """Predict with optional location coordinates."""
        self.model.eval()
        dataset = TabularDataset(X, site_ids, np.zeros(len(X)), location_coords)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        preds = []
        use_loc = self.use_location_token and location_coords is not None
        with torch.no_grad():
            for batch in loader:
                if use_loc:
                    X_b, _, _, loc_b = batch
                    loc_b = loc_b.to(device)
                    out = self.model(X_b.to(device), location_coords=loc_b)
                else:
                    X_b, _, _ = batch
                    out = self.model(X_b.to(device))
                preds.append(out.cpu().numpy())
        return np.vstack(preds).flatten()

    def inverse_normalize(self, data, feature_name=None):
        """Inverse normalization of predictions to original scale."""
        if not self.scaler_info:
            return data

        feature_strategies = self.scaler_info.get('feature_strategies', {})
        if not feature_strategies:
            meta = self.scaler_info.get('normalization_metadata', {})
            feature_strategies = meta.get('feature_strategies', {})

        if not feature_strategies:
            print("[Warning] No normalization strategies found")
            return data

        target_prefix = self.scaler_info.get('target_feature', '')
        if not feature_name:
            feature_name = target_prefix

        stats = None
        if feature_name in feature_strategies:
            stats = feature_strategies[feature_name]
        else:
            for key in feature_strategies:
                if key.startswith(feature_name):
                    stats = feature_strategies[key]
                    break

        if not stats:
            print(f"[Warning] No normalization stats found for feature: {feature_name}")
            return data

        method = stats.get('method', 'minmax')
        data = np.array(data)
        original = np.zeros_like(data)

        if method == 'separate_groundwater':
            pos = stats.get('positive', {})
            neg = stats.get('negative', {})
            pos_mask = data >= 0
            if np.any(pos_mask) and pos.get('max', 0) > pos.get('min', 0):
                original[pos_mask] = data[pos_mask] * (pos['max'] - pos['min']) + pos['min']
            neg_mask = (data < 0) & (data >= -1)
            if np.any(neg_mask) and neg.get('max', 0) > neg.get('min', 0):
                original[neg_mask] = (data[neg_mask] + 1) * (neg['max'] - neg['min']) + neg['min']
        elif method == 'minmax':
            d_min = stats.get('min', 0)
            d_max = stats.get('max', 1)
            if d_max > d_min:
                original = data * (d_max - d_min) + d_min
            else:
                original = data
        elif method == 'separate':
            pos = stats.get('positive', {})
            neg = stats.get('negative', {})
            pos_mask = (data >= 0) & (data <= 1)
            if np.any(pos_mask) and pos.get('max', 0) > pos.get('min', 0):
                original[pos_mask] = data[pos_mask] * (pos['max'] - pos['min']) + pos['min']
            neg_mask = (data < 0) & (data >= -1)
            if np.any(neg_mask) and neg.get('max', 0) > neg.get('min', 0):
                original[neg_mask] = (data[neg_mask] + 1) * (neg['max'] - neg['min']) + neg['min']
            zero_mask = (data > -1e-10) & (data < 1e-10)
            if np.any(zero_mask):
                original[zero_mask] = 0
        elif method == 'zero_fill':
            original = data
        else:
            print(f"[Warning] Unknown normalization method: {method}")
            original = data

        return original

    def evaluate_model(self):
        """Evaluate and visualize with location token support."""
        print("\n" + "=" * 60 + f"\n[Eval] Evaluating FT-Transformer Model (use_location_token={self.use_location_token})\n" + "=" * 60)
        results = {}
        coords = self.preprocessed_data.get('spatial_data_original', None)

        test_datasets = []
        if hasattr(self, 'X_val') and self.X_val is not None and len(self.X_val) > 0:
            test_datasets.append(("Validation", self.X_val, self.site_val, self.y_val, self.location_coords_val))
        if hasattr(self, 'X_test') and self.X_test is not None and len(self.X_test) > 0:
            test_datasets.append(("Test", self.X_test, self.site_test, self.y_test, self.location_coords_test))

        if not test_datasets:
            print("[Warning] No data available for evaluation")
            return results

        for name, X, site, y, loc_coords in test_datasets:
            if len(X) == 0:
                print(f"[Warning] {name} set is empty, skipping...")
                continue

            pred_norm = self.predict(X, site, location_coords=loc_coords)
            pred_real = self.inverse_normalize(pred_norm)
            true_real = self.inverse_normalize(y)

            if len(true_real) == 0 or len(pred_real) == 0:
                print(f"[Warning] {name} set has no valid data after inverse normalization")
                continue

            metrics = compute_evaluation_metrics(true_real, pred_real, prefix="")
            print(f"{name} | RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R2: {metrics['r2']:.4f} | MAPE: {metrics['mape']:.2f}%")

            results[name.lower()] = {
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'mape': metrics['mape'],
                'pred_real': pred_real,
                'true_real': true_real
            }
            plot_prediction_comparison(true_real, pred_real, f"{name} Set", model_name="FT_Transformer_Location",
                                       save_subdir="ft_transformer")

            if name == "Test" and coords is not None:
                try:
                    n_test_years = len(X) // self.n_sites
                    test_pred_reshaped = pred_real.reshape(n_test_years, self.n_sites)
                    test_true_reshaped = true_real.reshape(n_test_years, self.n_sites)
                    generate_combined_australia_map(
                        coords,
                        test_true_reshaped[-1, :],
                        test_pred_reshaped[-1, :],
                        "Test_FT_Location_Year",
                        "ft_transformer"
                    )
                except Exception as e:
                    print(f"[Vis] Spatial vis failed: {e}")
        return results


# Legacy main function removed.
# Use train_model.py in the project root for training:
#   python train_model.py --model ft_transformer
