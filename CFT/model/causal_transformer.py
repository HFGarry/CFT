#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal-Standard-Transformer Model Architecture.

This module implements a Standard Transformer (Step-wise) enhanced with 
LPCMCI-based causal priors for groundwater level prediction.

Key Components:
    - PaperStyleCausalUnit: Feature weighting unit based on causal priors
    - StandardCausalTransformer: Main transformer architecture with causal guidance
    - TabularDataset: Dataset wrapper for training and evaluation

Reference:
    Tigramite's LPCMCI algorithm for causal feature weighting.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class PaperStyleCausalUnit(nn.Module):
    """
    Paper-style Causal Weighting Unit.

    This module implements a learnable causal weight mechanism that integrates
    prior causal knowledge (from LPCMCI analysis) with data-driven adaptation.
    The weights are computed as a parameterized function of the causal prior matrix.
    """

    def __init__(self, window_size, n_features, causal_prior):
        super(PaperStyleCausalUnit, self).__init__()
        # causal_prior: Absolute path coefficient matrix from LPCMCI (shape: w, n_features)
        self.register_buffer('causal_prior', torch.FloatTensor(causal_prior))

        # Initialization: learnable_w=1.0, bias=0.5
        # This ensures ~62% information flow even for features with zero prior,
        # facilitating gradient propagation during early training stages
        self.learnable_w = nn.Parameter(torch.ones(window_size, n_features))
        self.bias = nn.Parameter(torch.ones(window_size, n_features) * 0.5)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Core computation: W_causal = Sigmoid(W_learnable * M_prior + Bias)

        Args:
            x: Input tensor of shape (batch, window, features)
        Returns:
            weighted_x: Input tensor multiplied by causal weights
            causal_weights: The computed attention weights
        """
        causal_weights = self.sigmoid(self.learnable_w * self.causal_prior + self.bias)
        return x * causal_weights.unsqueeze(0), causal_weights


# Backward compatibility alias
CausalWeightUnit = PaperStyleCausalUnit


class StandardCausalTransformer(nn.Module):
    """
    Causal Standard Transformer.

    A standard Transformer (Step-wise) enhanced with LPCMCI-based causal priors.
    Uses feature projection followed by causal weighting and transformer encoding.
    """

    def __init__(self, window_size, n_features, causal_strengths, d_model=64, n_heads=4,
                 num_layers=2, dim_feedforward=128, dropout=0.1):
        super(StandardCausalTransformer, self).__init__()

        self.window_size = window_size
        self.n_features = n_features
        self.d_model = d_model

        # Causal unit
        self.causal_unit = PaperStyleCausalUnit(window_size, n_features, causal_strengths)

        # Feature projection
        self.feature_projection = nn.Linear(n_features, d_model)

        # CLS Token
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.feature_projection.weight)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, src):
        batch_size = src.shape[0]

        # Step 1: Causal Weighting
        src_weighted, _ = self.causal_unit(src)

        # Step 2: Projection
        tokens = self.feature_projection(src_weighted)

        # Step 3: CLS Token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)

        # Step 4: Transformer Encoding
        x = self.transformer_encoder(self.dropout(tokens))

        # Step 5: Output (CLS token)
        return self.output_proj(x[:, 0, :])


class TabularDataset(Dataset):
    """Dataset wrapper for Causal-Transformer."""

    def __init__(self, X, site_ids, y):
        self.X = torch.FloatTensor(X)
        self.site_ids = torch.LongTensor(site_ids)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.site_ids[idx], self.y[idx]
