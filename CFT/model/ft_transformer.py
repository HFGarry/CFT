#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FT-Transformer Model Architecture.

This module implements a simple FT-Transformer baseline for tabular data.
Uses Feature Tokenizer + Transformer Encoder + CLS Token with optional
Location Token Encoding.

Key Components:
    - SimpleFTTransformer: Main transformer architecture
    - TabularDataset: Dataset wrapper for training and evaluation


"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SimpleFTTransformer(nn.Module):
    """
    FT-Transformer with Location Token Encoding.

    Uses Feature Tokenizer + Transformer Encoder + CLS Token.
    Location (Lat, Lon) is processed through a small MLP to create a "Location Token".
    """

    def __init__(self, window_size, n_features, d_model=64, n_heads=4,
                 num_layers=2, dim_feedforward=128, dropout=0.1,
                 use_location_token=True, location_embed_dim=32):
        super(SimpleFTTransformer, self).__init__()

        self.window_size = window_size
        self.n_features = n_features
        self.n_num_features = window_size * n_features
        self.d_model = d_model
        self.use_location_token = use_location_token

        # 1. Numerical Feature Tokens (Feature Tokenizer)
        self.num_weight = nn.Parameter(torch.Tensor(self.n_num_features, d_model))
        self.num_bias = nn.Parameter(torch.Tensor(self.n_num_features, d_model))

        # 2. Location Token Encoder (MLP for Lat/Lon)
        if use_location_token:
            # Input: 2 (Lat, Lon), Output: d_model dimension
            self.location_mlp = nn.Sequential(
                nn.Linear(2, location_embed_dim),
                nn.GELU(),
                nn.Linear(location_embed_dim, d_model)
            )
            self.has_location_token = True
        else:
            self.has_location_token = False

        # 3. [CLS] token
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, d_model))

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Output Head
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.num_weight)
        nn.init.zeros_(self.num_bias)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Initialize location MLP
        if self.has_location_token:
            for module in self.location_mlp:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, src, location_coords=None):
        """
        Args:
            src: (batch_size, window_size, n_features) - input features
            location_coords: (batch_size, 2) - Latitude, Longitude for each sample
        """
        batch_size = src.shape[0]

        # 1. Feature Tokenization
        x_num = src.view(batch_size, self.n_num_features).unsqueeze(-1)  # (B, N, 1)
        num_tokens = x_num * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)  # (B, N, d_model)

        # 2. Generate Location Token from Lat/Lon
        tokens_list = []

        # Add [CLS] token
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (1, 1, d_model)
        tokens_list.append(cls_token)

        # Add Location Token if coordinates provided
        if self.has_location_token and location_coords is not None:
            location_token = self.location_mlp(location_coords)  # (B, d_model)
            location_token = location_token.unsqueeze(1)  # (B, 1, d_model)
            tokens_list.append(location_token)

        # Add feature tokens
        tokens_list.append(num_tokens)

        # Concatenate all tokens: [CLS] + [Location] + Feature Tokens
        tokens = torch.cat(tokens_list, dim=1)  # (B, N+1+loc, d_model)

        # 3. Transformer Interaction
        x = self.transformer_encoder(tokens)

        # 4. Output [CLS] representation
        x_cls = x[:, 0, :]
        return self.output_proj(self.dropout(x_cls))


class TabularDataset(Dataset):
    """Dataset wrapper containing site_ids and location coordinates for evaluation."""

    def __init__(self, X, site_ids, y, location_coords=None):
        self.X = torch.FloatTensor(X)
        self.site_ids = torch.LongTensor(site_ids)
        self.y = torch.FloatTensor(y).unsqueeze(1)
        # Location coordinates: (batch_size, 2) - [Latitude, Longitude]
        if location_coords is not None:
            self.location_coords = torch.FloatTensor(location_coords)
        else:
            self.location_coords = None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.location_coords is not None:
            return self.X[idx], self.site_ids[idx], self.y[idx], self.location_coords[idx]
        return self.X[idx], self.site_ids[idx], self.y[idx]
