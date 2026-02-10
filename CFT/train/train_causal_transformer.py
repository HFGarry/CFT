#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal-Transformer Training Pipeline.

This module implements the training, evaluation, and inference pipeline
for the Causal-Standard-Transformer model.

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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Tigramite Integration (LPCMCI)
try:
    from tigramite import data_processing as pp
    from tigramite.lpcmci import LPCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    TIGRAMITE_AVAILABLE = True
except ImportError:
    TIGRAMITE_AVAILABLE = False
    print("[Warning] Tigramite not found. Causal weights will fall back to Correlation.")

# Import from model folder
from model.causal_transformer import (
    StandardCausalTransformer,
    TabularDataset
)

from data.data_manager import DataManager
from utils.visualization_utils import (
    plot_training_history,
    generate_combined_australia_map,
    plot_prediction_comparison,
    plot_causal_weight_comparison
)
from utils.evaluation_utils import (
    compute_evaluation_metrics,
    analyze_causal_interpretability,
    extract_causal_weights_from_model
)


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")


class TabularTransformerPredictor:
    """Causal-Transformer Predictor with training and evaluation logic."""

    def __init__(self, window_size=4, target_feature='Groundwater_'):
        self.window_size = window_size
        self.target_feature = target_feature
        self.model = None
        self.history = None
        self.preprocessed_data = None
        self.causal_strengths = None
        self._raw_time_series_matrix = None

    def load_preprocessed_data(self, preprocessed_filename):
        data_manager = DataManager()
        self.preprocessed_data, preprocessor, metadata = data_manager.load_preprocessed_data(preprocessed_filename)
        print(f"[Data] Loaded {len(self.preprocessed_data['site_ids'])} sites")
        return True

    def create_sequences(self):
        """Construct temporal sliding windows for training."""
        print(f"[Data] Building sequences (window_size={self.window_size})...")
        dynamic_data = self.preprocessed_data['dynamic_data_normalized']
        years = self.preprocessed_data['years']
        feature_names = self.preprocessed_data['dynamic_features']
        self.site_ids_list = self.preprocessed_data.get('site_ids', [])
        n_sites_total, n_years = dynamic_data.shape[0], len(years)

        # Feature alignment
        feature_types = {}
        for idx, fname in enumerate(feature_names):
            match = re.search(r'^(.*?)_?(\d{4})$', fname)
            prefix = match.group(1).rstrip('_') if match else fname.rstrip('_')
            year = int(match.group(2)) if match else None
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
                        data_by_year[years.index(year), :, feat_idx] = dynamic_data[:, col_idx]

        clean_target = self.target_feature.rstrip('_')
        target_prefix = next((p for p in feature_types if clean_target == p or clean_target in p), None)
        self.target_feat_idx = list(feature_types.keys()).index(target_prefix)
        self.n_sites = n_sites_total
        self._raw_time_series_matrix = data_by_year

        train_split_year = self.preprocessed_data['train_split_year']
        val_split_year = self.preprocessed_data.get('val_split_year', None)

        # Define valid sample indices for sliding window construction
        valid_samples = []
        for i in range(n_years - self.window_size):
            valid_samples.append(i)

        # Construct sliding windows
        X_windows = np.array([data_by_year[i: i + self.window_size] for i in valid_samples])
        y_targets = data_by_year[[i + self.window_size for i in valid_samples], :, self.target_feat_idx]

        X_flattened = X_windows.transpose(0, 2, 1, 3).reshape(-1, self.window_size, n_feature_types)
        y_flattened = y_targets.flatten()
        site_ids_flattened = np.tile(np.arange(n_sites_total), len(valid_samples))

        years_arr = np.array(years)
        sample_years = np.repeat(years_arr[[i + self.window_size for i in valid_samples]], n_sites_total)

        train_mask = sample_years <= train_split_year
        if val_split_year:
            val_mask = (sample_years > train_split_year) & (sample_years <= val_split_year)
            test_mask = sample_years > val_split_year
        else:
            val_mask = np.zeros_like(train_mask, dtype=bool)
            test_mask = ~train_mask

        self.X_train, self.site_train, self.y_train = X_flattened[train_mask], site_ids_flattened[train_mask], \
            y_flattened[train_mask]
        self.X_val, self.site_val, self.y_val = X_flattened[val_mask], site_ids_flattened[val_mask], y_flattened[val_mask]
        self.X_test, self.site_test, self.y_test = X_flattened[test_mask], site_ids_flattened[test_mask], y_flattened[test_mask]

        self.scaler_info = {
            'target_feature': target_prefix,
            'normalization_metadata': self.preprocessed_data.get('normalization_metadata', {}),
            'feature_strategies': self.preprocessed_data.get('normalization_metadata', {}).get('feature_strategies', {})
        }
        print(f"[Data] Scaler info initialized with target: {target_prefix}")
        print(f"[Data] Sequences ready. Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        return True

    def _calculate_causal_strengths(self):
        """Compute lagged causal strengths using LPCMCI algorithm."""
        n_features = self._raw_time_series_matrix.shape[2]
        w = self.window_size

        def fallback_corr():
            print("[Causal] Using correlation fallback...")
            causal_matrix = np.zeros((w, n_features))
            for f in range(n_features):
                for t in range(w):
                    c = np.abs(np.corrcoef(self.X_train[:, t, f], self.y_train)[0, 1])
                    causal_matrix[t, f] = c if not np.isnan(c) else 0.0
            return causal_matrix / (np.max(causal_matrix) + 1e-9)

        if not TIGRAMITE_AVAILABLE:
            self.causal_strengths = fallback_corr()
            return self.causal_strengths

        print(f"[Causal] Running Tigramite LPCMCI (Lags: 1 to {w})...")
        try:
            avg_series = np.mean(self._raw_time_series_matrix, axis=1)
            dataframe = pp.DataFrame(avg_series)
            cond_ind_test = ParCorr()
            lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
            results = lpcmci.run_lpcmci(tau_max=w, pc_alpha=0.8)

            val_matrix = results['val_matrix']
            causal_matrix = np.zeros((w, n_features))
            for f in range(n_features):
                for t in range(1, w + 1):
                    strength = np.abs(val_matrix[f, self.target_feat_idx, t])
                    causal_matrix[w - t, f] = strength

            max_val = np.max(causal_matrix)
            if max_val > 0:
                self.causal_strengths = causal_matrix / max_val
                print("[Causal] Causal weights computed successfully.")
            else:
                raise ValueError("No significant causal links found")
        except Exception as e:
            print(f"[Causal] LPCMCI failed: {e}. Falling back to correlation.")
            self.causal_strengths = fallback_corr()
        return self.causal_strengths

    def build_model(self, d_model=64, n_heads=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        causal_prior = self._calculate_causal_strengths()
        n_features = self.X_train.shape[2]
        self.model = StandardCausalTransformer(
            window_size=self.window_size, n_features=n_features,
            causal_strengths=causal_prior, d_model=d_model, n_heads=n_heads,
            num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout
        ).to(device)
        return True

    def train_model(self, epochs=100, batch_size=32, learning_rate=0.001, weight_decay=1e-5, use_callbacks=True,
                    callback_params=None, verbose=1):
        train_dataset = TabularDataset(self.X_train, self.site_train, self.y_train)
        val_dataset = TabularDataset(self.X_val, self.site_val, self.y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = callback_params.get('early_stop_patience', 15) if callback_params else 15

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for X_b, _, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                output = self.model(X_b)
                loss = criterion(output, y_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_b.size(0)

            train_loss /= len(train_dataset)
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, _, y_b in val_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    val_loss += criterion(self.model(X_b), y_b).item() * X_b.size(0)

            val_loss /= len(val_dataset)
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)

            if verbose == 1 and (epoch % 5 == 0):
                print(f"Epoch {epoch} - loss: {train_loss:.5f} - val_loss: {val_loss:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_causal_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            if use_callbacks and patience_counter >= early_stop_patience:
                break

        if os.path.exists('best_causal_model.pth'):
            self.model.load_state_dict(torch.load('best_causal_model.pth', weights_only=False))
            os.remove('best_causal_model.pth')
        self.history = history
        return True

    def predict(self, X, site_ids):
        self.model.eval()
        dataset = TabularDataset(X, site_ids, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        preds = []
        with torch.no_grad():
            for X_b, _, _ in loader:
                out = self.model(X_b.to(device))
                preds.append(out.cpu().numpy())
        return np.vstack(preds).flatten()

    def inverse_normalize(self, data, feature_name=None):
        """Inverse normalization of predictions."""
        if not self.scaler_info:
            return data
        feature_strategies = self.scaler_info.get('feature_strategies', {})
        if not feature_strategies:
            meta = self.scaler_info.get('normalization_metadata', {})
            feature_strategies = meta.get('feature_strategies', {})
        if not feature_strategies:
            return data

        target_prefix = self.scaler_info.get('target_feature', '').rstrip('_')
        if not feature_name:
            feature_name = target_prefix

        stats = None
        if feature_name in feature_strategies:
            stats = feature_strategies[feature_name]
        elif f"{feature_name}_" in feature_strategies:
            stats = feature_strategies[f"{feature_name}_"]
        else:
            for key in sorted(feature_strategies.keys()):
                if key.startswith(feature_name):
                    stats = feature_strategies[key]
                    break

        if not stats:
            return data

        method = stats.get('method', 'minmax')
        data = np.array(data)
        original = np.zeros_like(data)

        if method in ['separate_groundwater', 'separate']:
            pos, neg = stats.get('positive', {}), stats.get('negative', {})
            pos_mask = data >= 0
            if np.any(pos_mask) and pos.get('max', 0) > pos.get('min', 0):
                original[pos_mask] = data[pos_mask] * (pos['max'] - pos['min']) + pos['min']
            neg_mask = data < 0
            if np.any(neg_mask) and neg.get('max', 0) > neg.get('min', 0):
                original[neg_mask] = (data[neg_mask] + 1) * (neg['max'] - neg['min']) + neg['min']
        elif method == 'minmax':
            d_min, d_max = stats.get('min', 0), stats.get('max', 1)
            original = data * (d_max - d_min) + d_min if d_max > d_min else data
        else:
            original = data
        return original

    def evaluate_model(self):
        """Evaluate model and generate visualizations."""
        print("\n" + "=" * 60 + "\n[Eval] Evaluating Causal-Standard-Transformer\n" + "=" * 60)
        results = {}
        coords = self.preprocessed_data.get('spatial_data_original', None)
        eval_datasets = []
        if hasattr(self, 'X_val') and self.X_val is not None and len(self.X_val) > 0:
            eval_datasets.append(("Validation", self.X_val, self.site_val, self.y_val))
        if hasattr(self, 'X_test') and self.X_test is not None and len(self.X_test) > 0:
            eval_datasets.append(("Test", self.X_test, self.site_test, self.y_test))

        for name, X, site, y in eval_datasets:
            pred_norm = self.predict(X, site)
            pred_real = self.inverse_normalize(pred_norm)
            true_real = self.inverse_normalize(y)

            metrics = compute_evaluation_metrics(true_real, pred_real, prefix="")
            print(f"{name} | RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R2: {metrics['r2']:.4f}")

            results[name.lower()] = {
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'pred_real': pred_real,
                'true_real': true_real
            }
            plot_prediction_comparison(true_real, pred_real, f"{name} Set", model_name="Causal_Standard",
                                       save_subdir="causal_transformer")

            if name == "Test" and coords is not None:
                try:
                    n_test_years = len(X) // self.n_sites
                    test_pred_reshaped = pred_real.reshape(n_test_years, self.n_sites)
                    test_true_reshaped = true_real.reshape(n_test_years, self.n_sites)
                    generate_combined_australia_map(
                        coords, test_true_reshaped[-1, :], test_pred_reshaped[-1, :],
                        "Test_Causal_Year", "causal_transformer"
                    )
                except Exception as e:
                    print(f"[Vis] Spatial visualization failed: {e}")
        return results

    def analyze_causal_interpretability(self, save_dir='causal_transformer', target_lag=None):
        """Analyze causal interpretability: Compare LPCMCI prior weights with learned weights."""
        if self.model is None:
            print("[Warning] Model not trained yet. Cannot analyze causal interpretability.")
            return None

        feature_names = self.preprocessed_data.get('dynamic_features', [])
        prior_2d, learned_2d, window_size, n_features = extract_causal_weights_from_model(self.model)

        df_causal = analyze_causal_interpretability(
            model=self.model,
            feature_names=feature_names,
            model_name="Causal_Transformer",
            save_dir=save_dir,
            target_lag=target_lag
        )

        if df_causal is not None:
            dm = DataManager()
            dm.save_causal_results(
                model_type='causal_transformer',
                df_causal=df_causal,
                prior_weights=prior_2d,
                learned_weights=learned_2d,
                filename='causal_feature_importance_summary.csv'
            )

        return df_causal


# Legacy main function removed.
# Use train_model.py in the project root for training:
#   python train_model.py --model causal_transformer
