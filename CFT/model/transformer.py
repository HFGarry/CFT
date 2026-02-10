#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Transformer Model Architecture.

This module implements a baseline Tabular Transformer for groundwater level prediction
without site embedding or location tokens.

Key Components:
    - PositionalEncoding: Sinusoidal positional encoding layer
    - BaseTabularTransformer: Main transformer architecture
    - TabularDataset: Dataset wrapper for training and evaluation

Reference:
    Vaswani, A., et al. (2017). Attention is all you need. NeurIPS 2017.
"""

import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset


class PositionalEncoding(nn.Module):
    """
    Positional Encoding Layer.

    Implements sinusoidal positional encoding to inject position information
    into the input sequence, allowing the model to leverage the sequential
    nature of the data.
    """

    def __init__(self, d_model, max_len=500, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BaseTabularTransformer(nn.Module):
    """
    Baseline Tabular Transformer without Site Embedding.

    A standard transformer encoder architecture for tabular data prediction.
    Uses feature projection, positional encoding, and transformer encoder layers.
    """

    def __init__(self, window_size, n_features, d_model=64, n_heads=4,
                 num_layers=2, dim_feedforward=128, dropout=0.1):
        super(BaseTabularTransformer, self).__init__()

        self.window_size = window_size
        self.d_model = d_model

        # 1. Feature Projection
        self.input_proj = nn.Linear(n_features, d_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=window_size, dropout=dropout)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Output Head
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training stability."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, src):
        """
        Forward pass.

        Args:
            src: (batch_size, window_size, n_features) - input features
        """
        # Feature projection
        x = self.input_proj(src)  # (batch, window, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Last-step output
        x = x[:, -1, :]

        # Output
        x = self.dropout(x)
        output = self.output_proj(x)

        return output


class TabularDataset(Dataset):
    """Dataset wrapper for transformer training."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
