#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Training Launcher.

This module provides a unified entry point for training different models.
Supports multiple model architectures with a simple command-line interface.

Usage:
    python train_model.py --model causal_ft_transformer
    python train_model.py --model causal_transformer
    python train_model.py --model ft_transformer
    python train_model.py --model transformer
    python train_model.py --model lstm

Models:
    - causal_ft_transformer: Causal Feature Tokenizer Transformer
    - causal_transformer: Causal Standard Transformer
    - ft_transformer: FT-Transformer Baseline
    - transformer: Base Tabular Transformer
    - lstm: LSTM Baseline
"""

import argparse
import sys
import os

# Add project root to Python path for proper module imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.data_manager import DataManager


def get_available_models():
    """Return dictionary of available models."""
    return {
        'causal_ft_transformer': {
            'name': 'Causal-FT-Transformer',
            'module': 'train.train_causal_ft_transformer',
            'class': 'TabularTransformerPredictor',
            'description': 'Causal Feature Tokenizer Transformer with LPCMCI priors',
            'has_main': False
        },
        'causal_transformer': {
            'name': 'Causal-Transformer',
            'module': 'train.train_causal_transformer',
            'class': 'TabularTransformerPredictor',
            'description': 'Standard Transformer with causal priors',
            'has_main': False
        },
        'ft_transformer': {
            'name': 'FT-Transformer',
            'module': 'train.train_ft_transformer',
            'class': 'TabularTransformerPredictor',
            'description': 'FT-Transformer Baseline with location tokens',
            'has_main': False
        },
        'transformer': {
            'name': 'Base Transformer',
            'module': 'train.train_transformer',
            'class': 'TabularTransformerPredictor',
            'description': 'Baseline Tabular Transformer',
            'has_main': False
        },
        'lstm': {
            'name': 'LSTM',
            'module': 'train.train_lstm',
            'class': 'LSTMBaselinePredictor',
            'description': 'LSTM Baseline for time-series prediction',
            'has_main': False
        }
    }


def list_models():
    """List all available models."""
    models = get_available_models()
    print("\n" + "=" * 60)
    print("Available Models:")
    print("=" * 60)
    for key, info in models.items():
        print(f"  {key:<25} - {info['name']}")
        print(f"                              {info['description']}")
    print("=" * 60)


def get_default_config(model_name):
    """Return default configuration for a model."""
    configs = {
        'causal_ft_transformer': {
            'target': 'Groundwater_',
            'window_size': 4,
            'use_location_token': True,
            'location_embed_dim': 32,
            'model': {
                'd_model': 64,
                'n_heads': 4,
                'num_layers': 2,
                'dim_feedforward': 128,
                'dropout': 0.1
            },
            'training': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'weight_decay': 1e-5,
                'use_callbacks': True,
                'callback_params': None,
                'verbose': 1
            }
        },
        'causal_transformer': {
            'target': 'Groundwater_',
            'window_size': 4,
            'model': {
                'd_model': 64,
                'n_heads': 4,
                'num_layers': 2,
                'dim_feedforward': 128,
                'dropout': 0.1
            },
            'training': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'weight_decay': 1e-5,
                'use_callbacks': True,
                'callback_params': None,
                'verbose': 1
            }
        },
        'ft_transformer': {
            'target': 'Groundwater_',
            'window_size': 4,
            'use_location_token': True,
            'location_embed_dim': 32,
            'model': {
                'd_model': 64,
                'n_heads': 4,
                'num_layers': 2,
                'dim_feedforward': 128,
                'dropout': 0.1
            },
            'training': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'weight_decay': 1e-5,
                'use_callbacks': True,
                'callback_params': None,
                'verbose': 1
            }
        },
        'lstm': {
            'window_size': 4,
            'hidden_size': 64,
            'dropout': 0.05,
            'learning_rate': 0.001,
            'batch_size': 256,
            'epochs': 50
        },
        'transformer': {
            'target': 'Groundwater_',
            'window_size': 4,
            'model': {
                'd_model': 64,
                'n_heads': 4,
                'num_layers': 2,
                'dim_feedforward': 128,
                'dropout': 0.1
            },
            'training': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'weight_decay': 1e-5,
                'use_callbacks': True,
                'callback_params': None,
                'verbose': 1
            }
        }
    }
    return configs.get(model_name, {})


def train_model(model_name):
    """
    Train a specific model.

    Args:
        model_name: Name of the model to train
    """
    models = get_available_models()

    if model_name not in models:
        print(f"[Error] Unknown model: {model_name}")
        print("Available models:")
        for key in models.keys():
            print(f"  - {key}")
        sys.exit(1)

    model_info = models[model_name]
    print(f"\n[Info] Training model: {model_info['name']}")
    print(f"[Info] Description: {model_info['description']}")

    try:
        module = __import__(model_info['module'], fromlist=[model_info['class']])
        predictor_class = getattr(module, model_info['class'])
        main_func = getattr(module, 'main', None)

        if main_func is not None:
            # Module has main() function, execute it
            main_func()
        else:
            # Module doesn't have main(), use default config and run training
            print(f"[Info] Using default configuration for {model_name}")
            config = get_default_config(model_name)

            if model_name == 'causal_ft_transformer':
                predictor = predictor_class(
                    window_size=config['window_size'],
                    use_location_token=config['use_location_token']
                )
                dm = DataManager()
                avail = dm.list_available_data('preprocessed')
                if not avail['preprocessed']:
                    print("[Error] No preprocessed data found.")
                    return
                data_path = avail['preprocessed'][0]

                predictor.load_preprocessed_data(data_path)
                predictor.create_sequences()
                predictor.build_model(
                    d_model=config['model']['d_model'],
                    n_heads=config['model']['n_heads'],
                    num_layers=config['model']['num_layers'],
                    dim_feedforward=config['model']['dim_feedforward'],
                    dropout=config['model']['dropout'],
                    location_embed_dim=config['location_embed_dim']
                )
                predictor.train_model(
                    epochs=config['training']['epochs'],
                    batch_size=config['training']['batch_size'],
                    learning_rate=config['training']['learning_rate'],
                    weight_decay=config['training']['weight_decay'],
                    use_callbacks=config['training']['use_callbacks'],
                    callback_params=config['training']['callback_params'],
                    verbose=config['training']['verbose'],
                    causal_consistency_lambda=config['training']['causal_consistency_lambda']
                )
                evaluation_results = predictor.evaluate_model()

                model_data = {
                    'model_state_dict': predictor.model.state_dict(),
                    'model': predictor.model,
                    'window_size': config['window_size'],
                    'n_features': predictor.X_train.shape[2] if hasattr(predictor, 'X_train') and predictor.X_train is not None else None,
                    'target_feature': config['target'],
                    'use_location_token': config['use_location_token'],
                    'location_embed_dim': config['location_embed_dim'],
                    'd_model': config['model']['d_model'],
                    'n_heads': config['model']['n_heads'],
                    'num_layers': config['model']['num_layers'],
                    'dim_feedforward': config['model']['dim_feedforward'],
                    'dropout': config['model']['dropout']
                }
                training_config = {
                    'd_model': config['model']['d_model'],
                    'n_heads': config['model']['n_heads'],
                    'num_layers': config['model']['num_layers'],
                    'dim_feedforward': config['model']['dim_feedforward'],
                    'dropout': config['model']['dropout'],
                    'learning_rate': config['training']['learning_rate'],
                    'batch_size': config['training']['batch_size'],
                    'epochs': config['training']['epochs'],
                    'weight_decay': config['training']['weight_decay'],
                    'causal_consistency_lambda': config['training']['causal_consistency_lambda'],
                    'window_size': config['window_size'],
                    'target_feature': config['target'],
                    'use_location_token': config['use_location_token'],
                    'location_embed_dim': config['location_embed_dim']
                }
                dm.save_causal_ft_transformer_model_results(
                    model_data,
                    evaluation_results,
                    training_config,
                    y_true=evaluation_results.get('test', {}).get('true_real'),
                    y_pred=evaluation_results.get('test', {}).get('pred_real')
                )

            elif model_name == 'causal_transformer':
                predictor = predictor_class(
                    window_size=config['window_size']
                )
                dm = DataManager()
                avail = dm.list_available_data('preprocessed')
                if not avail['preprocessed']:
                    print("[Error] No preprocessed data found.")
                    return
                data_path = avail['preprocessed'][0]

                predictor.load_preprocessed_data(data_path)
                predictor.create_sequences()
                predictor.build_model(
                    d_model=config['model']['d_model'],
                    n_heads=config['model']['n_heads'],
                    num_layers=config['model']['num_layers'],
                    dim_feedforward=config['model']['dim_feedforward'],
                    dropout=config['model']['dropout']
                )
                predictor.train_model(
                    epochs=config['training']['epochs'],
                    batch_size=config['training']['batch_size'],
                    learning_rate=config['training']['learning_rate'],
                    weight_decay=config['training']['weight_decay'],
                    use_callbacks=config['training']['use_callbacks'],
                    callback_params=config['training']['callback_params'],
                    verbose=config['training']['verbose']
                )
                evaluation_results = predictor.evaluate_model()

                model_data = {
                    'model_state_dict': predictor.model.state_dict(),
                    'model': predictor.model,
                    'window_size': config['window_size'],
                    'n_features': predictor.X_train.shape[2] if hasattr(predictor, 'X_train') and predictor.X_train is not None else None,
                    'target_feature': config['target'],
                    'd_model': config['model']['d_model'],
                    'n_heads': config['model']['n_heads'],
                    'num_layers': config['model']['num_layers'],
                    'dim_feedforward': config['model']['dim_feedforward'],
                    'dropout': config['model']['dropout']
                }
                training_config = {
                    'd_model': config['model']['d_model'],
                    'n_heads': config['model']['n_heads'],
                    'num_layers': config['model']['num_layers'],
                    'dim_feedforward': config['model']['dim_feedforward'],
                    'dropout': config['model']['dropout'],
                    'learning_rate': config['training']['learning_rate'],
                    'batch_size': config['training']['batch_size'],
                    'epochs': config['training']['epochs'],
                    'weight_decay': config['training']['weight_decay'],
                    'window_size': config['window_size'],
                    'target_feature': config['target']
                }
                dm.save_causal_transformer_model_results(
                    model_data,
                    evaluation_results,
                    training_config,
                    y_true=evaluation_results.get('test', {}).get('true_real'),
                    y_pred=evaluation_results.get('test', {}).get('pred_real')
                )

            elif model_name == 'ft_transformer':
                predictor = predictor_class(
                    window_size=config['window_size'],
                    use_location_token=config['use_location_token']
                )
                dm = DataManager()
                avail = dm.list_available_data('preprocessed')
                if not avail['preprocessed']:
                    print("[Error] No preprocessed data found.")
                    return
                data_path = avail['preprocessed'][0]

                predictor.load_preprocessed_data(data_path)
                predictor.create_sequences()
                predictor.build_model(
                    d_model=config['model']['d_model'],
                    n_heads=config['model']['n_heads'],
                    num_layers=config['model']['num_layers'],
                    dim_feedforward=config['model']['dim_feedforward'],
                    dropout=config['model']['dropout'],
                    location_embed_dim=config['location_embed_dim']
                )
                predictor.train_model(
                    epochs=config['training']['epochs'],
                    batch_size=config['training']['batch_size'],
                    learning_rate=config['training']['learning_rate'],
                    weight_decay=config['training']['weight_decay'],
                    use_callbacks=config['training']['use_callbacks'],
                    callback_params=config['training']['callback_params'],
                    verbose=config['training']['verbose']
                )
                evaluation_results = predictor.evaluate_model()

                model_data = {
                    'model_state_dict': predictor.model.state_dict(),
                    'model': predictor.model,
                    'window_size': config['window_size'],
                    'n_features': predictor.X_train.shape[2] if hasattr(predictor, 'X_train') and predictor.X_train is not None else None,
                    'target_feature': config['target'],
                    'use_location_token': config['use_location_token'],
                    'location_embed_dim': config['location_embed_dim'],
                    'd_model': config['model']['d_model'],
                    'n_heads': config['model']['n_heads'],
                    'num_layers': config['model']['num_layers'],
                    'dim_feedforward': config['model']['dim_feedforward'],
                    'dropout': config['model']['dropout']
                }
                training_config = {
                    'd_model': config['model']['d_model'],
                    'n_heads': config['model']['n_heads'],
                    'num_layers': config['model']['num_layers'],
                    'dim_feedforward': config['model']['dim_feedforward'],
                    'dropout': config['model']['dropout'],
                    'learning_rate': config['training']['learning_rate'],
                    'batch_size': config['training']['batch_size'],
                    'epochs': config['training']['epochs'],
                    'weight_decay': config['training']['weight_decay'],
                    'window_size': config['window_size'],
                    'target_feature': config['target'],
                    'use_location_token': config['use_location_token'],
                    'location_embed_dim': config['location_embed_dim']
                }
                dm.save_ft_transformer_model_results(
                    model_data,
                    evaluation_results,
                    training_config,
                    y_true=evaluation_results.get('test', {}).get('true_real'),
                    y_pred=evaluation_results.get('test', {}).get('pred_real')
                )

            elif model_name == 'lstm':
                predictor = predictor_class(
                    window_size=config['window_size'],
                    hidden_size=config['hidden_size'],
                    dropout=config['dropout']
                )
                predictor.load_data()
                predictor.create_sequences()
                predictor.build_model()
                predictor.train(
                    epochs=config['epochs'],
                    batch_size=config['batch_size'],
                    lr=config['learning_rate']
                )
                evaluation_results = predictor.evaluate()

                dm = DataManager()
                model_data = {
                    'model_state_dict': predictor.model.state_dict(),
                    'window_size': predictor.window_size,
                    'n_features': predictor.n_features,
                    'target_feature': predictor.target_feature
                }
                training_config = {
                    'learning_rate': config['learning_rate'],
                    'batch_size': config['batch_size'],
                    'epochs': config['epochs'],
                    'window_size': config['window_size'],
                    'hidden_size': config['hidden_size'],
                    'optimizer': 'Adam'
                }
                dm.save_lstm_model_results(
                    model_data=model_data,
                    evaluation_results=evaluation_results,
                    training_config=training_config,
                    y_true=evaluation_results.get('test', {}).get('true_real'),
                    y_pred=evaluation_results.get('test', {}).get('pred_real')
                )

            elif model_name == 'transformer':
                predictor = predictor_class(
                    window_size=config['window_size']
                )
                dm = DataManager()
                avail = dm.list_available_data('preprocessed')
                if not avail['preprocessed']:
                    print("[Error] No preprocessed data found.")
                    return
                data_path = avail['preprocessed'][0]

                predictor.load_preprocessed_data(data_path)
                predictor.create_sequences()
                predictor.build_model(
                    d_model=config['model']['d_model'],
                    n_heads=config['model']['n_heads'],
                    num_layers=config['model']['num_layers'],
                    dim_feedforward=config['model']['dim_feedforward'],
                    dropout=config['model']['dropout']
                )
                predictor.train_model(
                    epochs=config['training']['epochs'],
                    batch_size=config['training']['batch_size'],
                    learning_rate=config['training']['learning_rate'],
                    weight_decay=config['training']['weight_decay'],
                    use_callbacks=config['training']['use_callbacks'],
                    callback_params=config['training']['callback_params'],
                    verbose=config['training']['verbose']
                )
                evaluation_results = predictor.evaluate_model()

                model_data = {
                    'model_state_dict': predictor.model.state_dict(),
                    'model': predictor.model,
                    'window_size': config['window_size'],
                    'n_features': predictor.X_train.shape[2] if hasattr(predictor, 'X_train') and predictor.X_train is not None else None,
                    'target_feature': config['target'],
                    'd_model': config['model']['d_model'],
                    'n_heads': config['model']['n_heads'],
                    'num_layers': config['model']['num_layers'],
                    'dim_feedforward': config['model']['dim_feedforward'],
                    'dropout': config['model']['dropout']
                }
                training_config = {
                    'd_model': config['model']['d_model'],
                    'n_heads': config['model']['n_heads'],
                    'num_layers': config['model']['num_layers'],
                    'dim_feedforward': config['model']['dim_feedforward'],
                    'dropout': config['model']['dropout'],
                    'learning_rate': config['training']['learning_rate'],
                    'batch_size': config['training']['batch_size'],
                    'epochs': config['training']['epochs'],
                    'weight_decay': config['training']['weight_decay'],
                    'window_size': config['window_size'],
                    'target_feature': config['target']
                }
                dm.save_transformer_model_results(
                    model_data,
                    evaluation_results,
                    training_config,
                    y_true=evaluation_results.get('test', {}).get('true_real'),
                    y_pred=evaluation_results.get('test', {}).get('pred_real')
                )

        print(f"\n[Info] {model_info['name']} training completed!")

    except ImportError as e:
        print(f"[Error] Failed to import {model_info['module']}: {e}")
        print("Please ensure the training module exists.")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Model Training Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_model.py --model causal_ft_transformer
    python train_model.py --model causal_transformer
    python train_model.py --model ft_transformer
    python train_model.py --model transformer
    python train_model.py --model lstm

List available models:
    python train_model.py --list
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Name of the model to train'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available models'
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if not args.model:
        parser.print_help()
        print("\n[Error] Please specify a model using --model or --list to see available models.")
        sys.exit(1)

    train_model(args.model)


if __name__ == "__main__":
    main()
