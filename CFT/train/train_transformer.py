#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Transformer Training Pipeline.

This module implements the training, evaluation, and inference pipeline
for the baseline Tabular Transformer model.

Key Components:
    - TabularTransformerPredictor: End-to-end pipeline for training and evaluation
    - HyperParamOptimizer: Optuna-based hyperparameter optimization
"""

import os
import re
import time
import random
import math
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from data.data_manager import DataManager
    from utils.visualization_utils import (
        plot_training_history,
        generate_combined_australia_map,
        plot_prediction_comparison,
        plot_feature_importance
    )
    from utils.evaluation_utils import compute_evaluation_metrics
except ImportError:
    print("[Warning] Local modules not found.")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Global Seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")


class TabularTransformerPredictor:
    """Baseline Tabular Transformer Predictor."""

    def __init__(self, window_size=4, target_feature='Groundwater_'):
        self.window_size = window_size
        self.target_feature = target_feature
        self.model = None
        self.history = None
        self.preprocessed_data = None

        self.X_train = self.site_train = self.y_train = None
        self.X_val = self.site_val = self.y_val = None
        self.X_test = self.site_test = self.y_test = None

        self.site_ids_list = None
        self.n_sites = 0
        self.feature_types = []
        self.scaler_info = None

    def load_preprocessed_data(self, preprocessed_filename):
        """Load data."""
        data_manager = DataManager()
        self.preprocessed_data, preprocessor, metadata = data_manager.load_preprocessed_data(preprocessed_filename)
        print(f"[Data] Loaded {len(self.preprocessed_data['site_ids'])} sites")
        return True

    def create_sequences(self):
        """Construct time-window samples."""
        print(f"[Data] Building sequences (window_size={self.window_size})...")

        dynamic_data = self.preprocessed_data['dynamic_data_normalized']
        years = self.preprocessed_data['years']
        feature_names = self.preprocessed_data['dynamic_features']
        self.site_ids_list = self.preprocessed_data.get('site_ids', [])
        n_sites_total, n_years = dynamic_data.shape[0], len(years)

        # 1. Parse features
        feature_types = {}
        for idx, fname in enumerate(feature_names):
            match = re.search(r'^(.*?)_?(\d{4})$', fname)
            if match:
                prefix = match.group(1)
                year = int(match.group(2))
                if prefix.endswith('_'):
                    prefix = prefix[:-1]
            else:
                prefix = fname
                year = None
            feature_types.setdefault(prefix, []).append((year, idx))

        for prefix in feature_types:
            feature_types[prefix].sort(key=lambda x: x[0] if x[0] is not None else 0)

        clean_target = self.target_feature.rstrip('_')
        target_prefix = next((p for p in feature_types if p == clean_target), None)
        if target_prefix is None:
            target_prefix = next((p for p in feature_types if clean_target in p), None)
        if target_prefix is None:
            raise ValueError(f"[Error] Target feature not found: {self.target_feature}")

        target_feat_idx = list(feature_types.keys()).index(target_prefix)
        self.target_feat_idx = target_feat_idx
        self.feature_types = list(feature_types.keys())
        self.n_sites = n_sites_total

        print("-" * 40)
        print(f"Features ({len(self.feature_types)} types):")
        print(self.feature_types)
        print("-" * 40)

        # 2. Build 3D Matrix
        n_feature_types = len(feature_types)
        data_by_year = np.zeros((n_years, n_sites_total, n_feature_types))

        for feat_idx, (prefix, year_col_list) in enumerate(feature_types.items()):
            for year, col_idx in year_col_list:
                if year is None:
                    data_by_year[:, :, feat_idx] = dynamic_data[:, col_idx]
                elif year in years:
                    y_idx = years.index(year)
                    data_by_year[y_idx, :, feat_idx] = dynamic_data[:, col_idx]

        # 3. Split Data (Prevent Leakage)
        train_split_year = self.preprocessed_data['train_split_year']
        val_split_year = self.preprocessed_data.get('val_split_year', None)
        years_arr = np.array(years)

        def get_dataset_type(year):
            if year <= train_split_year:
                return 'train'
            elif val_split_year and year <= val_split_year:
                return 'val'
            else:
                return 'test'

        # 4. Create Sequences
        valid_samples = []

        for i in range(n_years - self.window_size):
            pred_year_idx = i + self.window_size
            pred_year = years[pred_year_idx]
            pred_dataset = get_dataset_type(pred_year)

            window_years = years[i:i+self.window_size]
            window_datasets = [get_dataset_type(y) for y in window_years]

            # Check leakage
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
            raise ValueError("[Error] No valid samples.")

        # Build valid sequences
        X_windows = np.array([data_by_year[i: i + self.window_size] for i in valid_samples])
        y_targets = data_by_year[[i + self.window_size for i in valid_samples], :, target_feat_idx]

        # 5. Flatten & Align
        X_flattened = X_windows.transpose(0, 2, 1, 3).reshape(-1, self.window_size, n_feature_types)
        y_flattened = y_targets.flatten()
        site_ids_flattened = np.tile(np.arange(n_sites_total), len(valid_samples))
        year_indices_flattened = np.repeat([i + self.window_size for i in valid_samples], n_sites_total)

        # 6. Split
        sample_years = years_arr[year_indices_flattened]

        train_mask = sample_years <= train_split_year
        if val_split_year:
            val_mask = (sample_years > train_split_year) & (sample_years <= val_split_year)
            test_mask = sample_years > val_split_year
        else:
            remaining = ~train_mask
            val_mask = np.zeros_like(train_mask)
            test_mask = remaining

        self.X_train = X_flattened[train_mask]
        self.site_train = site_ids_flattened[train_mask]
        self.y_train = y_flattened[train_mask]
        self.X_val = X_flattened[val_mask]
        self.site_val = site_ids_flattened[val_mask]
        self.y_val = y_flattened[val_mask]
        self.X_test = X_flattened[test_mask]
        self.site_test = site_ids_flattened[test_mask]
        self.y_test = y_flattened[test_mask]

        # Summary
        print(f"[Data] Split Summary:")
        print(f"  Train: {len(self.X_train)} samples (Years: {sample_years[train_mask].min()}-{sample_years[train_mask].max()})")
        if len(sample_years[val_mask]) > 0:
            print(f"  Val: {len(self.X_val)} samples (Years: {sample_years[val_mask].min()}-{sample_years[val_mask].max()})")
        print(f"  Test: {len(self.X_test)} samples (Years: {sample_years[test_mask].min()}-{sample_years[test_mask].max()})")

        self.scaler_info = {
            'target_feature': target_prefix,
            'normalization_metadata': self.preprocessed_data.get('normalization_metadata', {}),
            'feature_strategies': self.preprocessed_data.get('normalization_metadata', {}).get('feature_strategies', {})
        }
        print(f"[Data] Scaler info initialized with target: {target_prefix}")
        print(f"[Data] Baseline samples ready.")
        return True

    def build_model(self, d_model=64, n_heads=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        """Build Baseline Model without Site Embedding."""
        from model.transformer import BaseTabularTransformer
        n_features = self.X_train.shape[2]
        self.model = BaseTabularTransformer(
            window_size=self.window_size,
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(device)
        print(f"[Model] Built Baseline (No Site Embedding): Input Feats={n_features}, d_model={d_model}")
        return True

    def train_model(self, epochs=100, batch_size=32, learning_rate=0.001,
                    weight_decay=1e-5, use_callbacks=True, callback_params=None, verbose=1):
        """Training loop."""
        from model.transformer import TabularDataset
        if self.model is None:
            raise ValueError("[Error] Model not built")

        train_dataset = TabularDataset(self.X_train, self.y_train)
        val_dataset = TabularDataset(self.X_val, self.y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=(verbose > 0)
        )

        history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = callback_params.get('early_stop_patience', 15) if callback_params else 15

        print(f"[Train] Starting Baseline Training (Epochs={epochs}, Batch={batch_size})...")
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                output = self.model(X_b)
                loss = criterion(output, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * X_b.size(0)

            train_loss /= len(train_dataset)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    output = self.model(X_b)
                    loss = criterion(output, y_b)
                    val_loss += loss.item() * X_b.size(0)

            val_loss /= len(val_dataset)
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)

            if verbose == 1 and (epoch % 5 == 0 or epoch == 0):
                print(f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.5f} - val_loss: {val_loss:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_base_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1

            if use_callbacks and patience_counter >= early_stop_patience:
                if verbose > 0:
                    print(f"[Train] Early stopping at epoch {epoch + 1}")
                break

        if os.path.exists('best_base_model.pth'):
            self.model.load_state_dict(torch.load('best_base_model.pth'))
            os.remove('best_base_model.pth')

        self.history = history
        return True

    def predict(self, X):
        """Inference."""
        from model.transformer import TabularDataset
        self.model.eval()
        dataset = TabularDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        preds = []
        with torch.no_grad():
            for X_b, _ in loader:
                X_b = X_b.to(device)
                out = self.model(X_b)
                preds.append(out.cpu().numpy())
        return np.vstack(preds).flatten()

    def inverse_normalize(self, data, feature_name=None):
        """Inverse normalization of predictions to original data scale."""
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
            pos_mask = data >= 0
            if np.any(pos_mask) and pos.get('max', 0) > pos.get('min', 0):
                original[pos_mask] = data[pos_mask] * (pos['max'] - pos['min']) + pos['min']
            neg_mask = data < 0
            if np.any(neg_mask) and neg.get('max', 0) > neg.get('min', 0):
                original[neg_mask] = (data[neg_mask] + 1) * (neg['max'] - neg['min']) + neg['min']
        elif method == 'zero_fill':
            original = data
        else:
            print(f"[Warning] Unknown normalization method: {method}")
            original = data

        return original

    def evaluate_model(self):
        """Evaluate."""
        print("\n" + "=" * 60)
        print("[Eval] Evaluating Baseline Model...")
        print("=" * 60)
        results = {}
        coords = self.preprocessed_data.get('spatial_data_original', None)

        def calc_metrics(name, X, y_true_norm):
            if len(X) == 0:
                print(f"[Warning] {name} set is empty, skipping...")
                return None
            pred_norm = self.predict(X)
            pred_real = self.inverse_normalize(pred_norm)
            true_real = self.inverse_normalize(y_true_norm)

            metrics = compute_evaluation_metrics(true_real, pred_real, prefix="")

            print(f"\n[{name} Evaluation]")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE:  {metrics['mae']:.4f}")
            print(f"  R2:   {metrics['r2']:.4f}")
            return {'rmse': metrics['rmse'], 'mae': metrics['mae'], 'r2': metrics['r2'],
                    'pred_real': pred_real, 'true_real': true_real}

        if len(self.X_val) > 0:
            val_res = calc_metrics("Validation", self.X_val, self.y_val)
            if val_res:
                results['validation'] = {k: v for k, v in val_res.items() if k not in ['pred_real', 'true_real']}

        test_res = None
        if len(self.X_test) > 0:
            test_res = calc_metrics("Test", self.X_test, self.y_test)
            if test_res:
                results['test'] = {
                    'rmse': test_res['rmse'],
                    'mae': test_res['mae'],
                    'r2': test_res['r2'],
                    'pred_real': test_res['pred_real'],
                    'true_real': test_res['true_real']
                }
        else:
            print("[Warning] Test set is empty, skipping evaluation...")
            return {
                'test': {
                    'rmse': float('nan'),
                    'mae': float('nan'),
                    'r2': float('nan'),
                    'pred_real': np.array([]),
                    'true_real': np.array([])
                }
            }

        if test_res is None:
            return results

        try:
            plot_prediction_comparison(
                test_res['true_real'], test_res['pred_real'],
                "Test Set", model_name="BaseTransformer", save_subdir="transformer_tabular"
            )

            if coords is not None:
                n_test_samples = len(self.X_test)
                if n_test_samples % self.n_sites == 0:
                    n_years_test = n_test_samples // self.n_sites
                    pred_reshaped = test_res['pred_real'].reshape(n_years_test, self.n_sites)
                    true_reshaped = test_res['true_real'].reshape(n_years_test, self.n_sites)

                    last_pred = pred_reshaped[-1, :]
                    last_true = true_reshaped[-1, :]
                    last_year = self.preprocessed_data['years'][-1]

                    generate_combined_australia_map(
                        coords, last_true, last_pred,
                        last_year, save_subdir="transformer_tabular"
                    )
        except Exception as e:
            print(f"[Vis] Visualization skipped: {e}")
        return test_res


class HyperParamOptimizer:
    """Optuna Optimization."""

    def __init__(self, predictor_cls, base_config):
        self.predictor_cls = predictor_cls
        self.base_config = base_config
        self.base_predictor = predictor_cls()
        self.base_predictor.load_preprocessed_data(base_config['data_path'])
        self.base_predictor.create_sequences()

    def objective(self, trial):
        d_model = trial.suggest_categorical('d_model', [32, 64, 128])
        valid_heads = [h for h in [2, 4, 8] if d_model % h == 0]
        if not valid_heads:
            valid_heads = [1]
        params = {
            'd_model': d_model,
            'n_heads': trial.suggest_categorical('n_heads', valid_heads),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'dim_feedforward': trial.suggest_int('dim_feedforward', 64, 256),
            'dropout': trial.suggest_float('dropout', 0.1, 0.4),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
        }
        predictor = self.predictor_cls()
        predictor.window_size = self.base_predictor.window_size
        predictor.n_sites = self.base_predictor.n_sites
        predictor.X_train = self.base_predictor.X_train
        predictor.site_train = self.base_predictor.site_train
        predictor.y_train = self.base_predictor.y_train
        predictor.X_val = self.base_predictor.X_val
        predictor.site_val = self.base_predictor.site_val
        predictor.y_val = self.base_predictor.y_val
        predictor.scaler_info = self.base_predictor.scaler_info
        predictor.build_model(
            d_model=params['d_model'], n_heads=params['n_heads'],
            num_layers=params['num_layers'], dim_feedforward=params['dim_feedforward'],
            dropout=params['dropout']
        )
        predictor.train_model(epochs=15, batch_size=params['batch_size'],
                              learning_rate=params['learning_rate'], verbose=0)
        val_pred = predictor.predict(predictor.X_val)
        return mean_squared_error(predictor.y_val, val_pred)

    def optimize(self, n_trials=20):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params


# Legacy main function removed.
# Use train_model.py in the project root for training:
#   python train_model.py --model transformer
