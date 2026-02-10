#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Utilities for Groundwater Prediction Models.

Features:
- Residual analysis and reporting
- Large residual point identification
- Evaluation metrics computation
- Causal interpretability analysis
"""

import re
import numpy as np
import pandas as pd


def compute_evaluation_metrics(y_true, y_pred, prefix=""):
    """
    Compute evaluation metrics including RMSE, MAE, R2, and MAPE.

    Args:
        y_true: True values
        y_pred: Predicted values
        prefix: Metric name prefix

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    y_true_arr = np.asarray(y_true).flatten()
    y_pred_arr = np.asarray(y_pred).flatten()

    # Remove NaN values
    valid_mask = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
    y_true_clean = y_true_arr[valid_mask]
    y_pred_clean = y_pred_arr[valid_mask]

    if len(y_true_clean) == 0:
        return {f'{prefix}rmse': np.nan, f'{prefix}mae': np.nan, f'{prefix}r2': np.nan, f'{prefix}mape': np.nan}

    # Compute metrics
    rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))

    # R2 score
    ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
    ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # MAPE (avoid division by zero)
    non_zero_mask = y_true_clean != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true_clean[non_zero_mask] - y_pred_clean[non_zero_mask]) / y_true_clean[non_zero_mask])) * 100
    else:
        mape = np.nan

    return {
        f'{prefix}rmse': rmse,
        f'{prefix}mae': mae,
        f'{prefix}r2': r2,
        f'{prefix}mape': mape,
        f'{prefix}n_valid': int(len(y_true_clean))
    }


def clean_feature_names(feature_names):
    """
    Clean feature names by removing year suffixes to obtain base feature types.

    Args:
        feature_names: List of feature names with potential year suffixes

    Returns:
        list: Cleaned feature names
    """
    if feature_names is None:
        return []

    clean_names = []
    for fname in feature_names:
        match = re.match(r'^(.*?)[_]?(\d{4})$', fname)
        clean = match.group(1).rstrip('_') if match else fname.rstrip('_')
        if clean not in clean_names:
            clean_names.append(clean)
    return clean_names


def extract_causal_weights_from_model(model):
    """
    Extract prior and learned causal weights from a causal model.

    Supports models with the following causal unit structures:
    - PaperStyleCausalUnit (CausalFTTransformer): causal_prior flattened, learnable_w per feature
    - PaperStyleCausalUnit (CausalTransformer): causal_prior 2D, learnable_w 2D

    Args:
        model: PyTorch model containing causal_unit attribute

    Returns:
        tuple: (prior_weights_2d, learned_weights_2d, window_size, n_features)
    """
    import torch

    causal_unit = model.causal_unit
    prior = causal_unit.causal_prior

    # Determine prior shape and reshape if needed
    if prior.ndim == 1:
        # Flattened prior from CausalFTTransformer
        # Need to infer 2D shape from learnable_w
        if causal_unit.learnable_w.ndim == 1:
            # CausalFTTransformer style: n_features
            n_features = causal_unit.learnable_w.shape[0]
            # Infer window_size from flattened prior
            total_size = prior.shape[0]
            if total_size % n_features == 0:
                window_size = total_size // n_features
            else:
                window_size = 4  # Default fallback
            prior_2d = prior.cpu().numpy().reshape(window_size, n_features)
        else:
            # Standard 2D format
            prior_2d = prior.cpu().numpy()
            window_size, n_features = prior_2d.shape
    else:
        # Already 2D
        prior_2d = prior.cpu().numpy()
        window_size, n_features = prior_2d.shape

    # Compute learned weights: W_causal = Sigmoid(W_learnable * M_prior + Bias)
    with torch.no_grad():
        logic = causal_unit.learnable_w * prior + causal_unit.bias
        learned_weights_flat = torch.sigmoid(logic).cpu().numpy()

    # Reshape learned weights to match prior shape
    if learned_weights_flat.ndim == 1:
        learned_2d = learned_weights_flat.reshape(window_size, n_features)
    else:
        learned_2d = learned_weights_flat

    return prior_2d, learned_2d, window_size, n_features


def analyze_causal_interpretability(model, feature_names=None, model_name="Model",
                                     save_dir='causal_analysis', target_lag=None):
    """
    Analyze causal interpretability by comparing LPCMCI prior weights with learned weights.

    This function evaluates how well the model has learned to incorporate
    prior causal knowledge by comparing the initial causal matrix with the
    final learned attention weights from the causal unit.

    Args:
        model: Trained PyTorch model with causal_unit attribute
        feature_names: List of feature names (optional)
        model_name: Name for logging and plot titles
        save_dir: Directory for saving analysis results
        target_lag: Target lag time (1, 2, ...), None displays all lags

    Returns:
        pd.DataFrame: DataFrame containing causal weight comparison results
    """
    print("\n" + "=" * 60)
    print(f"[Analysis] Causal Interpretability Analysis: {model_name}")
    print(f"[Analysis] Target Lag: {target_lag if target_lag else 'All Lags'}")
    print("=" * 60)

    # Extract weights from model
    prior_2d, learned_2d, window_size, n_features = extract_causal_weights_from_model(model)

    # Clean feature names
    if feature_names is None:
        raw_features = [f"Feature_{i}" for i in range(n_features)]
    else:
        raw_features = clean_feature_names(feature_names)
        # Ensure we have enough names
        while len(raw_features) < n_features:
            raw_features.append(f"Feature_{len(raw_features)}")

    # Generate summary statistics by averaging across lags
    prior_1d = prior_2d.mean(axis=0)  # Average across lags
    learned_1d = learned_2d.mean(axis=0)  # Average across lags

    # Create summary DataFrame
    df_causal = pd.DataFrame({
        'Feature': raw_features[:len(prior_1d)],
        'Causal_Prior_Mean': prior_1d,
        'Learned_Weight_Mean': learned_1d
    }).sort_values(by='Learned_Weight_Mean', ascending=False)

    # Add difference column (learned weights vs. prior)
    df_causal['Weight_Difference'] = df_causal['Learned_Weight_Mean'] - df_causal['Causal_Prior_Mean']

    # Add rank columns
    df_causal['Prior_Rank'] = df_causal['Causal_Prior_Mean'].rank(ascending=False).astype(int)
    df_causal['Learned_Rank'] = df_causal['Learned_Weight_Mean'].rank(ascending=False).astype(int)
    df_causal['Rank_Change'] = df_causal['Prior_Rank'] - df_causal['Learned_Rank']

    # Print summary statistics
    print(f"\n[Analysis] Window Size: {window_size}, Features: {n_features}")
    print(f"[Analysis] Prior Weight Range: [{prior_1d.min():.4f}, {prior_1d.max():.4f}]")
    print(f"[Analysis] Learned Weight Range: [{learned_1d.min():.4f}, {learned_1d.max():.4f}]")

    # Print feature importance summary
    print(f"\n[Analysis] Feature Importance Summary (Top 10):")
    print("-" * 60)
    header = f"{'Feature':<25} {'Prior':>10} {'Learned':>10} {'Diff':>10} {'Rank Chg':>10}"
    print(header)
    print("-" * 60)

    for _, row in df_causal.head(10).iterrows():
        print(f"{row['Feature']:<25} {row['Causal_Prior_Mean']:>10.4f} "
              f"{row['Learned_Weight_Mean']:>10.4f} {row['Weight_Difference']:>10.4f} "
              f"{row['Rank_Change']:>+10}")

    print("-" * 60)

    # Compute overall statistics
    prior_learned_corr = np.corrcoef(prior_1d, learned_1d)[0, 1]
    mean_weight_diff = np.mean(np.abs(df_causal['Weight_Difference']))
    print(f"\n[Analysis] Correlation (Prior vs Learned): {prior_learned_corr:.4f}")
    print(f"[Analysis] Mean Absolute Weight Difference: {mean_weight_diff:.4f}")

    return df_causal


def compute_causal_weight_statistics(prior_weights, learned_weights):
    """
    Compute comprehensive statistics for causal weight comparison.

    Args:
        prior_weights: Prior causal weights (2D: window_size x n_features)
        learned_weights: Learned causal weights (2D: window_size x n_features)

    Returns:
        dict: Dictionary containing various causal weight statistics
    """
    prior_2d = np.asarray(prior_weights)
    learned_2d = np.asarray(learned_weights)

    # Average across lags
    prior_1d = prior_2d.mean(axis=0)
    learned_1d = learned_2d.mean(axis=0)

    stats = {}

    # Basic statistics
    stats['prior_mean'] = float(np.mean(prior_1d))
    stats['prior_std'] = float(np.std(prior_1d))
    stats['prior_max'] = float(np.max(prior_1d))
    stats['prior_min'] = float(np.min(prior_1d))

    stats['learned_mean'] = float(np.mean(learned_1d))
    stats['learned_std'] = float(np.std(learned_1d))
    stats['learned_max'] = float(np.max(learned_1d))
    stats['learned_min'] = float(np.min(learned_1d))

    # Correlation between prior and learned
    if np.std(prior_1d) > 0 and np.std(learned_1d) > 0:
        stats['correlation'] = float(np.corrcoef(prior_1d, learned_1d)[0, 1])
    else:
        stats['correlation'] = 0.0

    # Mean absolute difference
    stats['mean_abs_difference'] = float(np.mean(np.abs(learned_1d - prior_1d)))

    # KL divergence approximation (P || Q)
    eps = 1e-10
    p = prior_1d + eps
    q = learned_1d + eps
    p = p / p.sum()
    q = q / q.sum()
    stats['kl_divergence_prior_to_learned'] = float(np.sum(p * np.log(p / q)))

    # Top features analysis
    top_k = min(5, len(prior_1d))
    top_prior_indices = np.argsort(prior_1d)[-top_k:][::-1]
    top_learned_indices = np.argsort(learned_1d)[-top_k:][::-1]

    stats['top_prior_features'] = [int(i) for i in top_prior_indices]
    stats['top_learned_features'] = [int(i) for i in top_learned_indices]
    stats['top_features_overlap'] = len(set(top_prior_indices) & set(top_learned_indices))

    return stats
