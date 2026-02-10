#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Utilities for Groundwater Prediction Models.

Features:
- Australia spatial map visualization (Actual vs Predicted vs Residuals)
- Prediction comparison plots
- Feature importance analysis
- Training history visualization
- Causal weight comparison (LPCMCI vs Learned)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Optional dependencies
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

def _spatial_interpolate_nan(spatial_data, actual, predicted):
    """
    Interpolate NaN values based on spatial coordinates.

    Args:
        spatial_data: Spatial coordinates [n_sites, 2]
        actual: Actual values [n_sites]
        predicted: Predicted values [n_sites]

    Returns:
        tuple: (spatial_data, actual_clean, predicted_clean)
    """
    try:
        from scipy.interpolate import griddata
    except ImportError:
        print("Warning: scipy not available, using mean replacement")
        actual_mean = np.nanmean(actual)
        predicted_mean = np.nanmean(predicted)
        actual_clean = np.where(np.isnan(actual), actual_mean, actual)
        predicted_clean = np.where(np.isnan(predicted), predicted_mean, predicted)
        return spatial_data, actual_clean, predicted_clean

    actual_clean = actual.copy()
    predicted_clean = predicted.copy()

    # Identify valid data points
    valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
    n_valid = np.sum(valid_mask)

    if n_valid < 3:  # Minimum 3 points required for interpolation
        print(f"   Insufficient valid data points ({n_valid}), using mean replacement")
        actual_clean = np.nan_to_num(actual, nan=np.nanmean(actual))
        predicted_clean = np.nan_to_num(predicted, nan=np.nanmean(predicted))
        return spatial_data, actual_clean, predicted_clean

    valid_coords = spatial_data[valid_mask]

    # Interpolate actual values
    nan_mask_actual = np.isnan(actual)
    if np.any(nan_mask_actual):
        n_nan_actual = np.sum(nan_mask_actual)
        try:
            interpolated_actual = griddata(
                valid_coords, actual[valid_mask],
                spatial_data[nan_mask_actual],
                method='linear', fill_value=np.nanmean(actual)
            )
            actual_clean[nan_mask_actual] = interpolated_actual
            print(f"   Actual values interpolation: {n_nan_actual} NaN values")
        except Exception as e:
            print(f"   Actual values interpolation failed, using mean: {e}")
            actual_clean[nan_mask_actual] = np.nanmean(actual)

    # Interpolate predicted values
    nan_mask_predicted = np.isnan(predicted)
    if np.any(nan_mask_predicted):
        n_nan_predicted = np.sum(nan_mask_predicted)
        try:
            interpolated_predicted = griddata(
                valid_coords, predicted[valid_mask],
                spatial_data[nan_mask_predicted],
                method='linear', fill_value=np.nanmean(predicted)
            )
            predicted_clean[nan_mask_predicted] = interpolated_predicted
            print(f"   Predicted values interpolation: {n_nan_predicted} NaN values")
        except Exception as e:
            print(f"   Predicted values interpolation failed, using mean: {e}")
            predicted_clean[nan_mask_predicted] = np.nanmean(predicted)

    return spatial_data, actual_clean, predicted_clean


def generate_combined_australia_map(spatial_data, actual, predicted, year, save_subdir="visualizations"):
    """
    Generate combined Australia map: spatial comparison (Actual vs Predicted) with residual distribution.

    Args:
        spatial_data: Original spatial coordinates [n_sites, 2] (lat, lon)
        actual: Actual values [n_sites]
        predicted: Predicted values [n_sites]
        year: Year
        save_subdir: Save subdirectory
    """
    if not CARTOPY_AVAILABLE:
        print("Cartopy unavailable, skipping map visualization")
        return

    try:
        print(f"Generating combined Australia map visualization (Actual/Predicted/Residuals) - {year}...")

        # Data preprocessing and interpolation (ensure consistent points across all three plots)
        actual = np.array(actual).flatten()
        predicted = np.array(predicted).flatten()
        spatial_clean, actual_clean, predicted_clean = _spatial_interpolate_nan(spatial_data, actual, predicted)
        residuals_clean = actual_clean - predicted_clean

        # Set plotting extent
        extent = [113, 155, -42, -12]
        projection = ccrs.PlateCarree()

        # Create figure: 1 row, 3 columns
        fig = plt.figure(figsize=(22, 7), dpi=100)
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])

        # Common styling
        point_size = 15
        alpha_val = 0.8

        # Calculate unified colorbar range for first two plots
        v_min = min(np.min(actual_clean), np.min(predicted_clean))
        v_max = max(np.max(actual_clean), np.max(predicted_clean))

        # 1. Actual values subplot
        ax1 = fig.add_subplot(gs[0], projection=projection)
        ax1.set_extent(extent, crs=projection)
        ax1.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.8)
        sc1 = ax1.scatter(spatial_clean[:, 1], spatial_clean[:, 0], c=actual_clean,
                          cmap='viridis', s=point_size, alpha=alpha_val, vmin=v_min, vmax=v_max, transform=projection)
        plt.colorbar(sc1, ax=ax1, shrink=0.6, label='Depth (m)')
        ax1.set_title(f'Actual Groundwater Depth ', fontsize=14)

        # 2. Predicted values subplot
        ax2 = fig.add_subplot(gs[1], projection=projection)
        ax2.set_extent(extent, crs=projection)
        ax2.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.8)
        sc2 = ax2.scatter(spatial_clean[:, 1], spatial_clean[:, 0], c=predicted_clean,
                          cmap='viridis', s=point_size, alpha=alpha_val, vmin=v_min, vmax=v_max, transform=projection)
        plt.colorbar(sc2, ax=ax2, shrink=0.6, label='Depth (m)')
        ax2.set_title(f'Predicted Groundwater Depth ', fontsize=14)

        # 3. Residual values subplot
        ax3 = fig.add_subplot(gs[2], projection=projection)
        ax3.set_extent(extent, crs=projection)
        ax3.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.8)
        # Fixed colorbar range for residuals: -5.0 to 5.0
        sc3 = ax3.scatter(spatial_clean[:, 1], spatial_clean[:, 0], c=residuals_clean,
                          cmap='coolwarm', s=point_size, alpha=0.9, vmin=-5.0, vmax=5.0, transform=projection)
        plt.colorbar(sc3, ax=ax3, shrink=0.6, label='Residual (m)')
        ax3.set_title(f'Prediction Residuals ', fontsize=14)

        # Main title
        fig.suptitle(f'Groundwater Depth Spatial Analysis', fontsize=18, y=0.95)

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'./visualizations/{save_subdir}/australia_combined_{year}_{timestamp}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        print(f"Combined map saved: {save_path}")

    except Exception as e:
        print(f"Failed to generate combined map: {e}")
        import traceback
        traceback.print_exc()
        plt.close()


def plot_prediction_comparison(y_true, y_pred, dataset_name, model_name="Model", save_subdir="visualizations"):
    """
    Generate prediction comparison plot.

    Args:
        y_true: True values
        y_pred: Predicted values
        dataset_name: Dataset name
        model_name: Model name
        save_subdir: Save subdirectory
    """
    print(f"Generating {dataset_name} prediction comparison plot...")

    # Handle different dimensional data
    original_shape = y_true.shape
    if len(y_true.shape) > 1:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        print(f"   Original data shape: {original_shape}, flattened: {y_true_flat.shape}")
    else:
        y_true_flat = y_true
        y_pred_flat = y_pred
        print(f"   Data shape: {y_true.shape}")

    # Remove NaN values
    valid_mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[valid_mask]
    y_pred_clean = y_pred_flat[valid_mask]

    if len(y_true_clean) == 0:
        print("No valid data for prediction comparison plot")
        return

    print(f"   Valid data points: {len(y_true_clean)}/{len(y_true_flat)}")
    print(f"   True values range: [{np.min(y_true_clean):.3f}, {np.max(y_true_clean):.3f}]")
    print(f"   Predicted values range: [{np.min(y_pred_clean):.3f}, {np.max(y_pred_clean):.3f}]")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(y_true_clean, y_pred_clean, alpha=0.6, s=20)

    # Add diagonal line
    min_val = min(np.min(y_true_clean), np.min(y_pred_clean))
    max_val = max(np.max(y_true_clean), np.max(y_pred_clean))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    ax1.set_xlabel('True Values (m)')
    ax1.set_ylabel('Predicted Values (m)')
    ax1.set_title('Predicted vs True Values')
    ax1.grid(True, alpha=0.3)

    # 2. Residual plot (fixed y-axis range: -30 to 30)
    ax2 = axes[0, 1]
    residuals = y_pred_clean - y_true_clean
    ax2.scatter(y_pred_clean, residuals, alpha=0.6, s=20)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values (m)')
    ax2.set_ylabel('Residuals (m)')
    ax2.set_title('Residual Distribution')
    ax2.set_ylim(-30, 30)  # Fixed residual range
    ax2.grid(True, alpha=0.3)

    # 3. Residual histogram (fixed x-axis range: -30 to 30)
    ax3 = axes[1, 0]
    residuals_clipped = np.clip(residuals, -30, 30)
    ax3.hist(residuals_clipped, bins=30, alpha=0.7, color='skyblue', edgecolor='black', range=(-30, 30))
    ax3.set_xlabel('Residuals (m)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Residual Histogram')
    ax3.set_xlim(-30, 30)  # Fixed residual range
    ax3.grid(True, alpha=0.3)

    # 4. Time series plot (if sufficient samples)
    ax4 = axes[1, 1]
    if len(y_true_clean) > 50:
        sample_indices = np.linspace(0, len(y_true_clean)-1, 50, dtype=int)
        ax4.plot(sample_indices, y_true_clean[sample_indices], 'o-', label='True Values', alpha=0.7)
        ax4.plot(sample_indices, y_pred_clean[sample_indices], 's-', label='Predicted Values', alpha=0.7)
    else:
        ax4.plot(y_true_clean, 'o-', label='True Values', alpha=0.7)
        ax4.plot(y_pred_clean, 's-', label='Predicted Values', alpha=0.7)

    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Groundwater Depth (m)')
    ax4.set_title('Time Series Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'{dataset_name} - {model_name} Prediction Comparison', fontsize=16)
    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'./visualizations/{save_subdir}/prediction_comparison_{timestamp}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Free memory, avoid blocking in non-interactive environments

    print(f"Prediction comparison plot saved: {save_path}")

    # Calculate and output basic statistics
    rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean)**2))
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    print(f"   Model: {model_name}, Dataset: {dataset_name}")
    print(f"   Performance metrics: RMSE={rmse:.4f}, MAE={mae:.4f}")


def plot_feature_importance(feature_names, feature_importance, model_name="Model", top_n=20, save_subdir="visualizations"):
    """
    Generate feature importance plot.

    Args:
        feature_names: List of feature names
        feature_importance: Feature importance array
        model_name: Model name
        top_n: Number of top important features to display
        save_subdir: Save subdirectory
    """
    if feature_importance is None or len(feature_names) == 0:
        print("No feature importance data available")
        return

    print(f"Generating feature importance plot...")
    print(f"   Total features: {len(feature_names)}")
    print(f"   Displaying top {top_n} important features")

    # Extract feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    # Output importance statistics
    total_importance = np.sum(feature_importance)
    top_10_importance = np.sum(importance_df.head(10)['importance'])
    print(f"   Top 10 features importance proportion: {top_10_importance/total_importance*100:.1f}%")

    # Select top N features
    top_features = importance_df.head(top_n)

    # Generate plots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # 1. Bar chart
    ax1 = axes[0]
    bars = ax1.barh(range(len(top_features)), top_features['importance'])
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'], fontsize=10)
    ax1.set_xlabel('Feature Importance')
    ax1.set_title(f'Top {top_n} Feature Importance')
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=8)

    # 2. Cumulative importance plot
    ax2 = axes[1]
    cumulative_importance = np.cumsum(importance_df['importance'].values)
    ax2.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'b-', linewidth=2)
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('Cumulative Importance')
    ax2.set_title('Cumulative Feature Importance')
    ax2.grid(True, alpha=0.3)

    # Add 80% and 95% importance lines
    total_importance = cumulative_importance[-1]
    ax2.axhline(y=total_importance * 0.8, color='r', linestyle='--', alpha=0.7, label='80%')
    ax2.axhline(y=total_importance * 0.95, color='orange', linestyle='--', alpha=0.7, label='95%')
    ax2.legend()

    plt.suptitle(f'{model_name} Feature Importance Analysis', fontsize=16)
    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'./visualizations/{save_subdir}/feature_importance_{timestamp}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Free memory, avoid blocking in non-interactive environments

    print(f"Feature importance plot saved: {save_path}")

    # Print important feature information
    print(f"Top 10 Important Features:")
    for i, row in top_features.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.6f}")


def plot_spatial_prediction_comparison(y_true, y_pred, dataset_name, model_name="Model", max_samples=4, save_subdir="visualizations"):
    """
    Generate spatial prediction comparison plot.

    Args:
        y_true: True values [batch, height, width]
        y_pred: Predicted values [batch, height, width]
        dataset_name: Dataset name
        model_name: Model name
        max_samples: Maximum number of samples
        save_subdir: Save subdirectory
    """
    print(f"Generating spatial prediction comparison plot...")

    # Select samples for visualization
    n_samples = min(max_samples, y_true.shape[0])
    sample_indices = np.linspace(0, y_true.shape[0] - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(sample_indices):
        true_sample = y_true[idx]
        pred_sample = y_pred[idx]
        diff_sample = pred_sample - true_sample

        # True values
        im1 = axes[i, 0].imshow(true_sample, cmap='viridis', aspect='auto')
        axes[i, 0].set_title(f'Sample {idx + 1} - True Values')
        axes[i, 0].set_xlabel('Longitude Direction')
        axes[i, 0].set_ylabel('Latitude Direction')
        plt.colorbar(im1, ax=axes[i, 0], shrink=0.8)

        # Predicted values
        im2 = axes[i, 1].imshow(pred_sample, cmap='viridis', aspect='auto')
        axes[i, 1].set_title(f'Sample {idx + 1} - Predicted Values')
        axes[i, 1].set_xlabel('Longitude Direction')
        axes[i, 1].set_ylabel('Latitude Direction')
        plt.colorbar(im2, ax=axes[i, 1], shrink=0.8)

        # Difference
        im3 = axes[i, 2].imshow(diff_sample, cmap='RdBu_r', aspect='auto')
        axes[i, 2].set_title(f'Sample {idx + 1} - Difference (Pred-True)')
        axes[i, 2].set_xlabel('Longitude Direction')
        axes[i, 2].set_ylabel('Latitude Direction')
        plt.colorbar(im3, ax=axes[i, 2], shrink=0.8)

    plt.suptitle(f'{dataset_name} - {model_name} Spatial Prediction Comparison - Groundwater Depth (m)', fontsize=16)
    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'./visualizations/{save_subdir}/spatial_comparison_{timestamp}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Free memory, avoid blocking in non-interactive environments

    print(f"Spatial prediction comparison plot saved: {save_path}")


def plot_training_history(history, model_name="Model", save_subdir="visualizations"):
    """
    Generate training history plot.

    Args:
        history: Keras training history object
        model_name: Model name
        save_subdir: Save subdirectory
    """
    if history is None:
        print("No training history data")
        return

    print(f"Generating training history plot...")

    plt.figure(figsize=(15, 5))

    # Loss curve
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # MAE curve
    plt.subplot(1, 3, 2)
    if 'mae' in history.history:
        plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
        plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
        plt.title('Model MAE', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No MAE Data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Model MAE', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Learning rate curve (if available)
    plt.subplot(1, 3, 3)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], linewidth=2, color='red')
        plt.title('Learning Rate', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
    else:
        plt.text(0.5, 0.5, 'No LR Data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Learning Rate', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.suptitle(f'{model_name} Training History', fontsize=16)
    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'./visualizations/{save_subdir}/training_history_{timestamp}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Free memory, avoid blocking in non-interactive environments

    print(f"Training history plot saved: {save_path}")


def plot_hyperparameter_optimization(study, save_subdir="visualizations"):
    """
    Generate hyperparameter optimization visualization.

    Args:
        study: Optuna study object
        save_subdir: Save subdirectory
    """
    try:
        import optuna
    except ImportError:
        print("Optuna unavailable, skipping hyperparameter optimization visualization")
        return

    print("Generating hyperparameter optimization visualization...")

    # Check if sufficient trials are available
    if len(study.trials) < 2:
        print("Insufficient number of trials, skipping visualization generation")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = f'./visualizations/{save_subdir}/hyperparameter_optimization'
    os.makedirs(viz_dir, exist_ok=True)

    try:
        # 1. Optimization history plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Optimization history
        ax1 = axes[0, 0]
        trial_numbers = [trial.number for trial in study.trials]
        trial_values = [trial.value for trial in study.trials if trial.value is not None]
        trial_nums_valid = [trial.number for trial in study.trials if trial.value is not None]

        ax1.plot(trial_nums_valid, trial_values, 'b-', alpha=0.6, label='Trial Values')

        # Add best value line
        best_values = []
        current_best = float('inf')
        for value in trial_values:
            if value < current_best:
                current_best = value
            best_values.append(current_best)

        ax1.plot(trial_nums_valid, best_values, 'r-', linewidth=2, label='Best Value')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Objective Value (MSE)')
        ax1.set_title('Hyperparameter Optimization History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Parameter importance (if sufficient trials)
        ax2 = axes[0, 1]
        if len(study.trials) >= 10:
            try:
                # Calculate parameter importance
                param_importance = optuna.importance.get_param_importances(study)

                if param_importance:
                    params = list(param_importance.keys())
                    importance_values = list(param_importance.values())

                    y_pos = np.arange(len(params))
                    ax2.barh(y_pos, importance_values)
                    ax2.set_yticks(y_pos)
                    ax2.set_yticklabels(params)
                    ax2.set_xlabel('Importance')
                    ax2.set_title('Parameter Importance Analysis')
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'Cannot compute parameter importance', ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Parameter Importance Analysis')
            except Exception as e:
                ax2.text(0.5, 0.5, f'Parameter importance computation failed:\n{str(e)}', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Parameter Importance Analysis')
        else:
            ax2.text(0.5, 0.5, 'Insufficient trials\n(requires >= 10)', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Parameter Importance Analysis')

        # 3. Trial distribution histogram
        ax3 = axes[1, 0]
        if trial_values:
            ax3.hist(trial_values, bins=min(20, len(trial_values)//2 + 1), alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(study.best_value, color='red', linestyle='--', linewidth=2, label=f'Best Value: {study.best_value:.4f}')
            ax3.set_xlabel('Objective Value (MSE)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Trial Results Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Parameter convergence (select main parameters)
        ax4 = axes[1, 1]
        main_params = ['n_estimators', 'max_depth', 'learning_rate']
        available_params = [p for p in main_params if any(p in trial.params for trial in study.trials)]

        if available_params:
            for param in available_params[:3]:  # Display up to 3 parameters
                param_values = []
                param_trials = []
                for trial in study.trials:
                    if param in trial.params and trial.value is not None:
                        param_values.append(trial.params[param])
                        param_trials.append(trial.number)

                if param_values:
                    ax4.scatter(param_trials, param_values, alpha=0.6, label=param, s=20)

            ax4.set_xlabel('Trial Number')
            ax4.set_ylabel('Parameter Value')
            ax4.set_title('Main Parameter Convergence')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No parameter data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Main Parameter Convergence')

        plt.suptitle(f'Hyperparameter Optimization Analysis (Trials: {len(study.trials)})', fontsize=16)
        plt.tight_layout()

        # Save figure
        save_path = os.path.join(viz_dir, f'hyperparameter_optimization_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Free memory, avoid blocking in non-interactive environments

        print(f"Hyperparameter optimization visualization saved: {save_path}")

        return save_path

    except Exception as e:
        print(f"Visualization generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_causal_weight_comparison(prior_weights, learned_weights, model_name="Model", feature_names=None, window_size=4, save_subdir="visualizations", target_lag=None):
    """
    Generate causal weight comparison plot: displays LPCMCI prior weights vs. model-learned weights.
    Supports displaying weight values for all lag periods.

    Args:
        prior_weights: Prior causal weights (Window, Features) or averaged (Features,)
        learned_weights: Learned weights (Window, Features) or averaged (Features,)
        feature_names: List of feature names
        model_name: Model name
        window_size: Time window size (number of lags)
        save_subdir: Save subdirectory
        target_lag: Target lag time (1, 2, ...), if None displays all lags
    """
    print(f"Generating {model_name} causal weight comparison plot...")

    try:
        # Convert to numpy arrays
        prior = np.array(prior_weights)
        learned = np.array(learned_weights)

        # If input is 1D (averaged), convert to 2D (1, Features) for plotting
        if prior.ndim == 1:
            prior = prior.reshape(1, -1)
            learned = learned.reshape(1, -1)
            # Lag labels only "Average"
            lag_labels = ["Average"]
        else:
            # If 2D, ensure matching with window_size
            if prior.shape[0] != window_size:
                print(f"   Warning: Number of rows in prior_weights ({prior.shape[0]}) does not match window_size ({window_size}).")
                window_size = prior.shape[0]
            lag_labels = [f"Lag {i+1}" for i in range(window_size)]

        n_lags, n_features = prior.shape

        # If target_lag specified, only display weights for that lag
        lags_to_plot = []
        if target_lag is not None:
            if 1 <= target_lag <= n_lags:
                lags_to_plot = [target_lag - 1] # Convert 1-based lag to 0-based index
                print(f"   Note: Displaying weights for lag {target_lag} only.")
            else:
                print(f"   Warning: target_lag ({target_lag}) invalid (valid range: 1-{n_lags}). Displaying all lags.")
                lags_to_plot = list(range(n_lags))
        else:
            lags_to_plot = list(range(n_lags))

        # Clean feature names
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        else:
            clean_feature_names = []
            for fname in feature_names:
                match = re.search(r'^(.*?)_?(\d{4})$', fname)
                clean = match.group(1).rstrip('_') if match else fname.rstrip('_')
                if clean not in clean_feature_names:
                    clean_feature_names.append(clean)
            feature_names = clean_feature_names

        # If too many features, select top 15 for detailed visualization
        # Calculate total importance across all lags and features for ranking
        total_importance = prior.mean(axis=0) + learned.mean(axis=0)
        top_indices = np.argsort(total_importance)[::-1][:min(15, n_features)]
        top_features = [feature_names[i] for i in top_indices]

        # Create figure (dynamically adjusted based on lags_to_plot)
        n_lags_to_plot = len(lags_to_plot)
        fig, axes = plt.subplots(n_lags_to_plot, 1, figsize=(14, 4 * n_lags_to_plot), squeeze=False)
        plt.subplots_adjust(hspace=0.4)

        for i, lag_idx in enumerate(lags_to_plot):
            ax = axes[i, 0]
            p_vals = prior[lag_idx, top_indices]
            l_vals = learned[lag_idx, top_indices]

            x = np.arange(len(top_features))
            width = 0.35

            rects1 = ax.bar(x - width/2, p_vals, width, label='LPCMCI Prior', color='#1f77b4', alpha=0.8)
            rects2 = ax.bar(x + width/2, l_vals, width, label='Learned Weight', color='#ff7f0e', alpha=0.8)

            ax.set_ylabel(f'Score ({lag_labels[lag_idx]})')
            ax.set_title(f'{lag_labels[lag_idx]} Feature Weights')
            ax.set_xticks(x)
            ax.set_xticklabels(top_features, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)

            # Add value labels
            ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=8)
            ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=8)

        plt.suptitle(f'{model_name} Causal Weights: LPCMCI Prior vs. Learned', fontsize=16)
        plt.tight_layout()

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'./visualizations/{save_subdir}/causal_weights_{timestamp}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Causal weight comparison plot saved: {save_path}")

        # Save detailed numerical table (CSV)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f'./visualizations/{save_subdir}/causal_weights_table_{timestamp}.csv'
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Build DataFrame containing all lag information
        detailed_data = []
        for i in range(n_lags):
            for j in range(n_features):
                detailed_data.append({
                    'Lag': lag_labels[i],
                    'Feature': feature_names[j],
                    'LPCMCI_Prior': prior[i, j],
                    'Learned_Weight': learned[i, j],
                    'Importance_Score': prior[i, j] + learned[i, j]
                })

        df_detailed = pd.DataFrame(detailed_data)
        df_detailed.to_csv(csv_path, index=False)
        print(f"Causal weight detailed numerical table saved: {csv_path}")

        return df_detailed

    except Exception as e:
        print(f"Failed to generate causal weight comparison plot: {e}")
        import traceback
        traceback.print_exc()
        return None
