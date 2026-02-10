# -*- coding: utf-8 -*-
"""
Data Management Module.

Handles saving and loading of preprocessed data and model results.
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime


class DataManager:
    """Data Manager for saving and loading processing results."""
    
    # Model type configuration
    MODEL_CONFIGS = {
        'lstm': {
            'name': 'LSTM Baseline',
            'dir': 'lstm',
            'result_dir': 'lstm_baseline',
            'extension': '.pth',  
            'is_pytorch': True
        },
        'transformer': {
            'name': 'Base Tabular Transformer (PyTorch)',
            'dir': 'transformer',
            'result_dir': 'transformer_tabular',
            'extension': '.pth'
        },
        'causal_transformer': {
            'name': 'Causal-Standard-Transformer',
            'dir': 'causal_transformer',
            'result_dir': 'causal_transformer',
            'extension': '.pth'
        },
        'ft_transformer': {
            'name': 'FT-Transformer (PyTorch)',
            'dir': 'ft_transformer',
            'result_dir': 'ft_transformer',
            'extension': '.pth'
        },
        'causal_ft_transformer': {
            'name': 'Causal-FT-Transformer',
            'dir': 'causal_ft_transformer',
            'result_dir': 'causal_ft_transformer',
            'extension': '.pth'
        },
        'test': {
            'name': 'Test Model (Causal-FT-Transformer)',
            'dir': 'test',
            'result_dir': 'test',
            'extension': '.pth',
            'is_pytorch': True
        }
    }

    def __init__(self, base_dir=None):
        # Auto-detect project root directory
        if base_dir is None:
            # Try to find project root by looking for common project markers
            current = os.path.abspath('.')
            potential_roots = [
                current,
                os.path.dirname(current),  # Parent directory
            ]

            # Check for project markers
            for potential_root in potential_roots:
                markers = ['train_model.py', 'requirements.txt', 'README.md', '.git']
                if any(os.path.exists(os.path.join(potential_root, m)) for m in markers):
                    base_dir = potential_root
                    break
            else:
                # Fallback to current directory
                base_dir = current

        self.base_dir = base_dir
        # All directories are relative to base_dir for consistency
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.results_dir = os.path.join(self.base_dir, 'results')
        self.visualizations_dir = os.path.join(self.base_dir, 'visualizations')
        self.ensure_directory_exists()
        self._init_result_savers()

    def ensure_directory_exists(self):
        """Ensure all base directories exist."""
        # Core data directories
        core_dirs = ['preprocessed', 'metadata', 'models', 'results', 'visualizations']
        for dir_name in core_dirs:
            os.makedirs(os.path.join(self.base_dir, dir_name), exist_ok=True)

        # Create model-specific subdirectories
        for model_type, config in self.MODEL_CONFIGS.items():
            # Model directory: models/{model_name}
            os.makedirs(os.path.join(self.models_dir, config['dir']), exist_ok=True)
            # Results directory: results/{result_name}
            os.makedirs(os.path.join(self.results_dir, config['result_dir']), exist_ok=True)
            # Visualization directory: visualizations/{result_name}
            vis_dir = os.path.join(self.visualizations_dir, config['result_dir'])
            os.makedirs(vis_dir, exist_ok=True)
            # Hyperparameter optimization subdirectory
            os.makedirs(os.path.join(vis_dir, 'hyperparameter_optimization'), exist_ok=True)

    def save_preprocessed_data(self, preprocessed_results, preprocessor, filename=None):
        """
        Save preprocessing results.

        Args:
            preprocessed_results: Preprocessing results dictionary
            preprocessor: Preprocessor instance
            filename: Save filename, auto-generated if None
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"preprocessed_data_{timestamp}"

        # Save preprocessing data
        data_filepath = os.path.join(self.base_dir, 'preprocessed', f"{filename}.pkl")
        with open(data_filepath, 'wb') as f:
            pickle.dump(preprocessed_results, f)

        # Save preprocessor state
        preprocessor_filepath = os.path.join(self.base_dir, 'preprocessed', f"{filename}_preprocessor.pkl")
        preprocessor.save_preprocessor(preprocessor_filepath)

        # Save metadata
        metadata = {
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'data_shape': {
                'n_sites': len(preprocessed_results['site_ids']),
                'n_years': len(preprocessed_results['years']),
                'n_features': len(preprocessed_results['dynamic_features']),
                'spatial_shape': preprocessed_results['spatial_data_normalized'].shape,
                'dynamic_shape': preprocessed_results['dynamic_data_normalized'].shape
            },
            'train_split_year': preprocessed_results['train_split_year'],
            'val_split_year': preprocessed_results.get('val_split_year'),
            'years_range': f"{preprocessed_results['years'][0]}-{preprocessed_results['years'][-1]}",
            'train_years': preprocessed_results.get('train_years', []),
            'val_years': preprocessed_results.get('val_years', []),
            'test_years': preprocessed_results.get('test_years', []),
            'n_clusters': preprocessor.n_clusters,
            'n_anomalous_sites': len(preprocessor.anomalous_sites) if preprocessor.anomalous_sites else 0,
            'n_outlier_corrections': len(preprocessor.outlier_records) if preprocessor.outlier_records else 0
        }

        metadata_filepath = os.path.join(self.base_dir, 'metadata', f"{filename}_metadata.json")
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Preprocessed data saved:")
        print(f"  Data: {data_filepath}")
        print(f"  Preprocessor: {preprocessor_filepath}")
        print(f"  Metadata: {metadata_filepath}")

        return filename

    def load_preprocessed_data(self, filename):
        """
        Load preprocessing results.

        Args:
            filename: Filename (without extension)

        Returns:
            preprocessed_results: Preprocessing results dictionary
            preprocessor: Preprocessor instance
            metadata: Metadata dictionary
        """
        from data.data_preprocessing import GroundwaterDataPreprocessor

        # Load preprocessing data
        data_filepath = os.path.join(self.base_dir, 'preprocessed', f"{filename}.pkl")
        with open(data_filepath, 'rb') as f:
            preprocessed_results = pickle.load(f)

        # Load preprocessor
        preprocessor = GroundwaterDataPreprocessor()
        preprocessor_filepath = os.path.join(self.base_dir, 'preprocessed', f"{filename}_preprocessor.pkl")
        preprocessor.load_preprocessor(preprocessor_filepath)

        # Load metadata
        metadata_filepath = os.path.join(self.base_dir, 'metadata', f"{filename}_metadata.json")
        with open(metadata_filepath, 'r') as f:
            metadata = json.load(f)

        print(f"Preprocessed data loaded: {filename}")
        print(f"  Sites: {metadata['data_shape']['n_sites']}")
        print(f"  Years: {metadata['data_shape']['n_years']} ({metadata['years_range']})")
        print(f"  Features: {metadata['data_shape']['n_features']}")

        # Display dataset split information
        if 'train_years' in metadata:
            print(f"  Training years: {len(metadata['train_years'])} ({metadata['train_years'][0] if metadata['train_years'] else 'N/A'}-{metadata['train_split_year']})")
        if 'val_years' in metadata and metadata['val_years']:
            print(f"  Validation years: {len(metadata['val_years'])} ({metadata['val_years'][0]}-{metadata['val_split_year']})")
        if 'test_years' in metadata and metadata['test_years']:
            print(f"  Test years: {len(metadata['test_years'])} ({metadata['test_years'][0]}-{metadata['test_years'][-1]})")

        return preprocessed_results, preprocessor, metadata

    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj

    def _save_keras_model(self, model, filepath):
        """Unified Keras model saving method."""
        # If file already exists, delete it first
        if os.path.exists(filepath):
            print(f"Warning: Model file already exists, will be overwritten: {filepath}")
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Failed to delete old model file: {e}")

        # Attempt to save in Keras format
        try:
            model.save(filepath, save_format='keras')
            print(f"Model saved in Keras format: {filepath}")
            return filepath
        except Exception as e:
            print(f"Keras format save failed, attempting H5 format: {e}")
            # Fallback to H5 format
            h5_filepath = filepath.replace('.keras', '.h5')
            if os.path.exists(h5_filepath):
                os.remove(h5_filepath)
            model.save(h5_filepath, save_format='h5')
            print(f"Model saved in H5 format: {h5_filepath}")
            return h5_filepath
    
    def save_model_results(self, model_type, model_data, evaluation_results, training_config, filename=None, y_true=None, y_pred=None):
        """
        Generic model saving method.

        Args:
            model_type: Model type
            model_data: Model data dictionary
            evaluation_results: Evaluation results
            training_config: Training configuration
            filename: Save filename
            y_true: True value array (optional)
            y_pred: Predicted value array (optional)
        """
        import joblib

        # Validate model type
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(self.MODEL_CONFIGS.keys())}")

        config = self.MODEL_CONFIGS[model_type]

        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{config['result_dir']}_{timestamp}"

        # Directory paths
        model_dir = os.path.join(self.models_dir, config['dir'])
        results_dir = os.path.join(self.results_dir, config['result_dir'])

        # Save model file
        # Check if PyTorch model (via configuration or model type)
        is_pytorch_model = config.get('is_pytorch', False) or model_type in [
            'transformer', 'causal_transformer',
            'ft_transformer', 'lstm', 'causal_ft_transformer', 'test'
        ]

        if is_pytorch_model:
            # PyTorch model
            import torch
            model_filepath = os.path.join(model_dir, f"{filename}{config['extension']}")
            if 'model_state_dict' in model_data:
                torch.save(model_data['model_state_dict'], model_filepath)
            elif 'model' in model_data and hasattr(model_data['model'], 'state_dict'):
                torch.save(model_data['model'].state_dict(), model_filepath)

            # Save other model data (excluding PyTorch objects)
            model_metadata = {}
            for k, v in model_data.items():
                if k not in ['model_state_dict', 'model']:
                    model_metadata[k] = v
                elif k == 'model':
                    # Retain model class name for reference
                    model_metadata['model_class'] = type(v).__name__

            metadata_filepath = os.path.join(model_dir, f"{filename}_data.pkl")
            joblib.dump(model_metadata, metadata_filepath)
        else:
            # Keras model
            model_filepath = os.path.join(model_dir, f"{filename}{config['extension']}")
            if 'model' in model_data and hasattr(model_data['model'], 'save'):
                model_filepath = self._save_keras_model(model_data['model'], model_filepath)

            # Save other model data
            model_metadata = {k: v for k, v in model_data.items() if k != 'model'}
            metadata_filepath = os.path.join(model_dir, f"{filename}_data.pkl")
            joblib.dump(model_metadata, metadata_filepath)

        # Save evaluation results
        evaluation_filepath = os.path.join(results_dir, f"{filename}_evaluation.json")

        # Supplement data_range metric (if not present)
        if 'test' in evaluation_results and 'data_range' not in evaluation_results['test']:
            if y_true is not None:
                try:
                    y_true_arr = np.array(y_true).flatten()
                    if len(y_true_arr) > 0:
                        evaluation_results['test']['data_range'] = {
                            'min': float(np.min(y_true_arr)),
                            'max': float(np.max(y_true_arr)),
                            'mean': float(np.mean(y_true_arr))
                        }
                except Exception as e:
                    print(f"[Warning] Failed to calculate data_range: {e}")
            elif 'metrics_test' in locals(): # For ConvLSTM
                 pass # Already handled locally

        with open(evaluation_filepath, 'w', encoding='utf-8') as f:
            json.dump(self._make_json_serializable(evaluation_results), f, indent=2, ensure_ascii=False)

        # Save predictions and true values
        if y_true is not None and y_pred is not None:
            y_df = pd.DataFrame({
                'y_true': y_true.flatten(),
                'y_pred': y_pred.flatten()
            })
            predictions_filepath = os.path.join(results_dir, f"{filename}_predictions.csv")
            y_df.to_csv(predictions_filepath, index=False)
            print(f"[Save] Predictions saved to: {predictions_filepath}")

        # Save training configuration
        config_filepath = os.path.join(results_dir, f"{filename}_config.json")
        with open(config_filepath, 'w', encoding='utf-8') as f:
            json.dump(self._make_json_serializable(training_config), f, indent=2, ensure_ascii=False)

        # Save metadata
        metadata = {
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'model_type': config['name'],
            'model_filepath': model_filepath,
            'evaluation_filepath': evaluation_filepath,
            'config_filepath': config_filepath,
            'evaluation_metrics': {
                metric_type: {k: v for k, v in metrics.items()
                             if k in ['rmse', 'mae', 'r2', 'mape', 'n_valid', 'data_range']}
                for metric_type, metrics in evaluation_results.items()
                if isinstance(metrics, dict)
            }
        }

        # Add model-specific metadata
        model_specific_keys = ['grid_size', 'window_size', 'target_channel', 'n_channels',
                               'target_feature', 'sequence_length', 'prediction_horizon',
                               'n_features', 'feature_names', 'n_sites', 'sites_h', 'sites_w',
                               'hyperparams']
        for key in model_specific_keys:
            if key in model_data:
                metadata[key] = model_data[key]

        # Convert all metadata values to JSON-serializable types
        metadata = self._make_json_serializable(metadata)

        metadata_filepath_main = os.path.join(self.base_dir, 'metadata', f"{filename}_{model_type}_metadata.json")
        with open(metadata_filepath_main, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Print save information
        print(f"{config['name']} model results saved:")
        print(f"  Model file: {model_filepath}")
        if model_type != 'xgboost':
            print(f"  Model data: {metadata_filepath}")
        print(f"  Evaluation results: {evaluation_filepath}")
        print(f"  Training configuration: {config_filepath}")
        print(f"  Metadata file: {metadata_filepath_main}")

        # Display evaluation metrics summary
        self._print_evaluation_summary(evaluation_results)

        return filename
    
    def _print_evaluation_summary(self, evaluation_results):
        """Print evaluation metrics summary."""
        if not evaluation_results:
            return

        print(f"\nEvaluation Metrics Summary:")
        for dataset_type, metrics in evaluation_results.items():
            if isinstance(metrics, dict) and any(k in metrics for k in ['rmse', 'mae', 'r2', 'mape']):
                print(f"  {dataset_type.upper()} Set:")
                for metric_name in ['rmse', 'mae', 'r2', 'mape', 'n_valid']:
                    if metric_name in metrics:
                        if metric_name == 'mape':
                            print(f"    {metric_name.upper()}: {metrics[metric_name]:.1f}%")
                        elif metric_name == 'n_valid':
                            print(f"    Valid data points: {metrics[metric_name]}")
                        else:
                            print(f"    {metric_name.upper()}: {metrics[metric_name]:.4f}")
    
    def save_convlstm_model_results(self, model_data, evaluation_results, training_config, filename=None, y_true=None, y_pred=None):
        """Save ConvLSTM model results."""
        return self.save_model_results('convlstm', model_data, evaluation_results, training_config, filename, y_true, y_pred)

    def list_available_data(self, data_type='all'):
        """
        List available data files.

        Args:
            data_type: 'preprocessed', 'models', or 'all'
        """
        available_data = {}

        if data_type in ['preprocessed', 'all']:
            preprocessed_dir = os.path.join(self.base_dir, 'preprocessed')
            if os.path.exists(preprocessed_dir):
                files = [f.replace('.pkl', '') for f in os.listdir(preprocessed_dir)
                         if f.endswith('.pkl') and not f.endswith('_preprocessor.pkl')]
                files.sort(key=lambda x: os.path.getmtime(os.path.join(preprocessed_dir, x + '.pkl')), reverse=True)
                available_data['preprocessed'] = files

        if data_type in ['models', 'all']:
            models_files = {}
            for model_type, config in self.MODEL_CONFIGS.items():
                model_dir = os.path.join(self.models_dir, config['dir'])
                if os.path.exists(model_dir):
                    files = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.pth', '.pkl'))]
                    if files:
                        models_files[model_type] = files
            available_data['models'] = models_files

        return available_data

    def get_data_info(self, filename, data_type):
        """
        Get data file information.

        Args:
            filename: Filename (without extension)
            data_type: 'preprocessed', 'gridded', 'features', or 'models'
        """
        metadata_filepath = os.path.join(self.base_dir, 'metadata', f"{filename}_metadata.json")

        if os.path.exists(metadata_filepath):
            with open(metadata_filepath, 'r') as f:
                metadata = json.load(f)
            return metadata
        else:
            return None

    def cleanup_old_data(self, days_old=30):
        """
        Clean up old data files.

        Args:
            days_old: Delete files older than this many days
        """
        import time
        from pathlib import Path

        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)

        for subdir in ['preprocessed', 'metadata']:
            dir_path = os.path.join(self.base_dir, subdir)
            if os.path.exists(dir_path):
                for file_path in Path(dir_path).iterdir():
                    if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        print(f"Deleted old file: {file_path}")

    def create_processing_pipeline_config(self, config_dict, filename=None):
        """
        Create processing pipeline configuration file.

        Args:
            config_dict: Configuration dictionary
            filename: Configuration filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_config_{timestamp}.json"

        config_filepath = os.path.join(self.base_dir, 'metadata', filename)
        with open(config_filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Pipeline configuration saved to: {config_filepath}")
        return filename
    
    # ============== Model Result Saver Wrapper Methods ==============

    @classmethod
    def _model_result_wrapper(cls, model_type):
        """Create wrapper method for model result saving."""
        def save_fn(model_data, evaluation_results, training_config, filename=None, y_true=None, y_pred=None):
            # This is a special wrapper that requires a DataManager instance
            # Usage: Call via instance: self.save_lstm_model_results(...)
            raise NotImplementedError("This method must be called on a DataManager instance")
        save_fn.__doc__ = f"Save {DataManager.MODEL_CONFIGS.get(model_type, {}).get('name', model_type)} model results."
        save_fn._model_type = model_type  # Mark model type
        return save_fn

    def _create_result_saver(self, model_type):
        """Generate actual result saving method at instance creation."""
        config = self.MODEL_CONFIGS.get(model_type, {})
        name = config.get('name', model_type)

        def save_fn(model_data, evaluation_results, training_config, filename=None, y_true=None, y_pred=None):
            return self.save_model_results(model_type, model_data, evaluation_results, training_config, filename, y_true, y_pred)

        save_fn.__doc__ = f"Save {name} model results."
        return save_fn

    def _init_result_savers(self):
        """Initialize all result saving methods."""
        self.save_lstm_model_results = self._create_result_saver('lstm')
        self.save_ft_transformer_model_results = self._create_result_saver('ft_transformer')
        self.save_transformer_model_results = self._create_result_saver('transformer')
        self.save_causal_transformer_model_results = self._create_result_saver('causal_transformer')
        self.save_causal_ft_transformer_model_results = self._create_result_saver('causal_ft_transformer')
        self.save_test_model_results = self._create_result_saver('test')
    
    def save_causal_results(self, model_type, df_causal, prior_weights=None, learned_weights=None, filename=None):
        """
        Save causal analysis results including weight comparison table and statistics.

        This method integrates with evaluation_utils.compute_causal_weight_statistics
        to provide comprehensive causal analysis results.

        Args:
            model_type: Model type ('causal_transformer', 'causal_lpe_transformer',
                       'causal_ft_transformer', 'causal_convlstm')
            df_causal: DataFrame containing causal weights comparison
            prior_weights: Prior causal weights array (optional, for statistics)
            learned_weights: Learned causal weights array (optional, for statistics)
            filename: Save filename

        Returns:
            dict: Dictionary containing saved file paths
        """
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"[Error] Unsupported model type: {model_type}")

        config = self.MODEL_CONFIGS[model_type]
        results_dir = os.path.join(self.results_dir, config['result_dir'])

        # Ensure directory exists
        os.makedirs(results_dir, exist_ok=True)

        saved_files = {}

        # Save causal weights comparison DataFrame
        if df_causal is not None:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = f"causal_analysis_{timestamp}.csv"
            else:
                csv_filename = filename if filename.endswith('.csv') else f"{filename}.csv"

            csv_filepath = os.path.join(results_dir, csv_filename)
            df_causal.to_csv(csv_filepath, index=False)
            print(f"[Save] Causal results saved to: {csv_filepath}")
            saved_files['causal_data'] = csv_filepath

        # Save causal weight statistics (if weights provided)
        if prior_weights is not None and learned_weights is not None:
            try:
                from utils.evaluation_utils import compute_causal_weight_statistics
                stats = compute_causal_weight_statistics(prior_weights, learned_weights)

                stats_filename = f"causal_statistics_{timestamp}.json" if 'timestamp' in locals() else "causal_statistics.json"
                stats_filepath = os.path.join(results_dir, stats_filename)

                with open(stats_filepath, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)

                print(f"[Save] Causal statistics saved to: {stats_filepath}")
                saved_files['causal_statistics'] = stats_filepath
            except ImportError:
                print("[Warning] evaluation_utils not available, skipping statistics save")

        return saved_files

    def load_processing_pipeline_config(self, filename):
        """Load processing pipeline configuration."""
        config_filepath = os.path.join(self.base_dir, 'metadata', filename)
        with open(config_filepath, 'r') as f:
            config = json.load(f)

        print(f"Pipeline configuration loaded from: {config_filepath}")
        return config
