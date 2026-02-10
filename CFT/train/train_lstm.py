#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Training Pipeline.

This module implements the training, evaluation, and inference pipeline
for the LSTM baseline model.

Key Components:
    - LSTMBaselinePredictor: End-to-end pipeline for training and evaluation
"""

import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model.lstm import LSTMModel
from data.data_manager import DataManager
from utils.visualization_utils import (
    plot_training_history,
    plot_prediction_comparison,
)
from utils.evaluation_utils import compute_evaluation_metrics


# Random Seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")


class LSTMBaselinePredictor:
    """LSTM Baseline Predictor for groundwater level prediction."""

    def __init__(self, window_size=4, target_feature='Groundwater_', hidden_size=64, dropout=0.2):
        self.window_size = window_size
        self.target_feature = target_feature
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.model = None
        self.preprocessed_data = None
        self.history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
        self.feature_names = None
        self.years = None
        self.n_features = None
        self.scaler_info = None

    def load_data(self):
        """
        Load preprocessed dataset and initialize normalization parameters.

        Retrieves the most recent preprocessed data file and extracts
        metadata required for inverse normalization during evaluation.
        """
        dm = DataManager()
        available = dm.list_available_data('preprocessed')
        if not available['preprocessed']:
            raise FileNotFoundError("[Error] No preprocessed data found.")
        latest_file = available['preprocessed'][0]
        print(f"[Data] Loading: {latest_file} ...")
        self.preprocessed_data, _, _ = dm.load_preprocessed_data(latest_file)
        self.scaler_info = {
            'feature_normalizers': self.preprocessed_data.get('normalization_metadata', {}),
            'target_feature': self.target_feature
        }

    def create_sequences(self):
        """
        Construct temporal sliding window sequences for LSTM input.

        Transforms the 3D data structure (Year × Site × Feature) into
        input-output pairs using a sliding window approach with specified
        temporal horizon (window_size). Labels are derived from the
        target feature at the prediction time step.
        """
        data = self.preprocessed_data
        dynamic_data = data['dynamic_data_normalized']
        years = data['years']
        feature_names = data['dynamic_features']
        n_sites, _ = dynamic_data.shape

        self.feature_names = feature_names
        self.years = years

        # Extract feature prefixes and corresponding years via regex pattern matching
        feature_types = {}
        for idx, fname in enumerate(feature_names):
            m = re.match(r'^(.*?)[_]?(\d{4})$', fname)
            if m:
                prefix, year = m.group(1), int(m.group(2))
                if prefix.endswith('_'): prefix = prefix[:-1]
                feature_types.setdefault(prefix, []).append((year, idx))

        for p in feature_types: feature_types[p].sort(key=lambda x: x[0])

        # Identify target variable index for prediction task
        target_prefix = None
        clean_target = self.target_feature.rstrip('_')
        for p in feature_types:
            if p == self.target_feature or p == clean_target:
                target_prefix = p
                break
        target_idx = list(feature_types.keys()).index(target_prefix)

        # Restructure data into 3D array: (n_years, n_sites, n_features)
        n_feats = len(feature_types)
        data_3d = np.zeros((len(years), n_sites, n_feats), dtype=np.float32)
        for f_idx, (prefix, year_list) in enumerate(feature_types.items()):
            for year, col_idx in year_list:
                if year in years:
                    data_3d[years.index(year), :, f_idx] = dynamic_data[:, col_idx]

        # Generate sliding window input sequences and corresponding targets
        X_temp, y_temp, year_indices = [], [], []
        for i in range(len(years) - self.window_size):
            # Input: window of consecutive time steps
            X_temp.append(data_3d[i:i + self.window_size])
            # Target: target feature at prediction time step
            y_temp.append(data_3d[i + self.window_size, :, target_idx])
            year_indices.append(i + self.window_size)

        X_all = np.array(X_temp).transpose(0, 2, 1, 3)
        y_all = np.array(y_temp)

        n_windows = X_all.shape[0]
        X_flat = X_all.reshape(n_windows * n_sites, self.window_size, n_feats)
        y_flat = y_all.reshape(n_windows * n_sites, 1)

        sample_years = np.repeat(np.array(years)[year_indices], n_sites)

        train_year = data['train_split_year']
        val_year = data.get('val_split_year')
        train_mask = sample_years <= train_year
        if val_year:
            val_mask = (sample_years > train_year) & (sample_years <= val_year)
            test_mask = sample_years > val_year
        else:
            val_mask = np.zeros_like(train_mask, dtype=bool)
            test_mask = ~train_mask

        self.X_train, self.y_train = X_flat[train_mask], y_flat[train_mask]
        self.X_val, self.y_val = X_flat[val_mask], y_flat[val_mask]
        self.X_test, self.y_test = X_flat[test_mask], y_flat[test_mask]

        print(f"[Data] Sequences ready: Train: {self.X_train.shape}, Val: {self.X_val.shape}, Test: {self.X_test.shape}")

    def build_model(self, input_dim=None):
        """
        Instantiate the LSTM model architecture.

        Constructs the LSTM network with specified hidden size and dropout
        regularization, then transfers to the computational device (GPU/CPU).
        """
        if input_dim is None:
            input_dim = self.X_train.shape[-1] if hasattr(self, 'X_train') else 10

        self.model = LSTMModel(
            input_size=input_dim,
            hidden_size=self.hidden_size,
            dropout=self.dropout
        ).to(device)
        self.n_features = input_dim
        print(self.model)

    def train(self, epochs=50, batch_size=256, lr=0.001):
        """
        Execute model training with mini-batch gradient descent.

        Implements the training loop including:
        - Forward pass and loss computation
        - Backpropagation and parameter optimization
        - Validation monitoring for hyperparameter tuning
        - Early stopping based on validation loss
        - Learning rate scheduling
        """
        # Convert data to FloatTensor to ensure type compatibility with model parameters
        train_dataset = TensorDataset(torch.from_numpy(self.X_train).float(), torch.from_numpy(self.y_train).float())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if len(self.X_val) > 0:
            val_dataset = TensorDataset(torch.from_numpy(self.X_val).float(), torch.from_numpy(self.y_val).float())
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            val_loader = None

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # Early Stopping mechanism: track best validation performance
        best_val_loss = float('inf')
        best_weights = self.model.state_dict()  # Initial weights
        patience = 10
        patience_counter = 0

        print("[Train] Starting LSTM training (PyTorch)...")

        for epoch in range(epochs):
            self.model.train()
            train_loss, train_mae = 0.0, 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                train_mae += torch.abs(outputs - targets).sum().item()

            train_loss /= len(train_loader.dataset)
            train_mae /= len(train_loader.dataset)

            # Validation phase: evaluate model on held-out data
            val_loss, val_mae = 0.0, 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = self.model(inputs)
                        v_loss = criterion(outputs, targets)
                        val_loss += v_loss.item() * inputs.size(0)
                        val_mae += torch.abs(outputs - targets).sum().item()
                val_loss /= len(val_loader.dataset)
                val_mae /= len(val_loader.dataset)

            self.history['loss'].append(train_loss)
            self.history['mae'].append(train_mae)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)

            scheduler.step(val_loss)
            print(f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")

            # Early Stopping check: terminate if no improvement observed
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break

        # Restore model weights from best performing epoch
        self.model.load_state_dict(best_weights)

    def inverse_normalize(self, data):
        """
        Transform normalized predictions back to original data scale.

        Applies the inverse of the preprocessing transformation using
        stored normalization metadata. Supports MinMax and separate
        positive/negative scaling strategies.
        """
        if self.scaler_info is None or not data.size:
            return data

        target = self.scaler_info.get('target_feature')
        norms = self.scaler_info.get('feature_normalizers', {})
        feature_strategies = norms.get('feature_strategies', {})

        # Retrieve normalization parameters for the target feature
        params = None
        if target:
            for k, v in feature_strategies.items():
                if target in k or target.rstrip('_') in k:
                    params = v
                    break

        if not params:
            return data

        data_arr = np.asarray(data)
        original = np.zeros_like(data_arr)
        method = params.get('method', 'minmax')

        if method in ['separate_groundwater', 'separate']:
            # Separate scaling for positive and negative values
            pos, neg = params.get('positive', {}), params.get('negative', {})
            pos_mask = data_arr >= 0
            if np.any(pos_mask) and pos.get('max', 0) > pos.get('min', 0):
                original[pos_mask] = data_arr[pos_mask] * (pos['max'] - pos['min']) + pos['min']
            neg_mask = (data_arr < 0)
            if np.any(neg_mask) and neg.get('max', 0) > neg.get('min', 0):
                original[neg_mask] = (data_arr[neg_mask] + 1) * (neg['max'] - neg['min']) + neg['min']
        elif method == 'minmax':
            d_min, d_max = params.get('min', 0), params.get('max', 1)
            original = data_arr * (d_max - d_min) + d_min if d_max > d_min else data_arr
        else:
            original = data_arr

        return original

    def evaluate(self, save_results=True):
        """
        Assess model performance on test dataset.

        Computes evaluation metrics (RMSE, MAE, R²) and generates
        diagnostic visualizations including training history and
        prediction scatter plots.
        """
        if not hasattr(self, 'X_test') or self.X_test is None or len(self.X_test) == 0:
            print("[Warning] Test set empty.")
            return {}

        self.model.eval()
        with torch.no_grad():
            # Prepare test inputs in FloatTensor format
            inputs = torch.from_numpy(self.X_test).float().to(device)
            y_pred = self.model(inputs).cpu().numpy()

        y_true_inv = self.inverse_normalize(self.y_test).flatten()
        y_pred_inv = self.inverse_normalize(y_pred).flatten()

        # Compute evaluation metrics using utility function
        metrics = compute_evaluation_metrics(y_true_inv, y_pred_inv, prefix="test_")

        print(f"Test Results | RMSE = {metrics['test_rmse']:.4f}, MAE = {metrics['test_mae']:.4f}, R2 = {metrics['test_r2']:.4f}")

        results = {
            'test': {
                'rmse': metrics['test_rmse'],
                'mae': metrics['test_mae'],
                'r2': metrics['test_r2'],
                'n_valid': metrics['test_n_valid'],
                'pred_real': y_pred_inv,
                'true_real': y_true_inv
            }
        }

        os.makedirs('visualizations/lstm_baseline', exist_ok=True)
        class HistoryShim:
            def __init__(self, history): self.history = history
        plot_training_history(HistoryShim(self.history), "LSTM_Baseline", "lstm_baseline")
        plot_prediction_comparison(y_true_inv, y_pred_inv, "test_lstm", "LSTM_Baseline", "lstm_baseline")

        return results


# Legacy main function removed.
# Use train_model.py in the project root for training:
#   python train_model.py --model lstm
