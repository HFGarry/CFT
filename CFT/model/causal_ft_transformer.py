#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal-FT-Transformer Model Architecture.

This module implements the causal feature tokenizer transformer architecture
with integrated causal discovery mechanisms (LPCMCI) for groundwater level
prediction tasks.

Key Components:
    - PaperStyleCausalUnit: Feature token weighting unit based on causal priors
    - CausalFTTransformer: Main transformer architecture with causal guidance
    - TabularDataset: Dataset wrapper for training and evaluation

Reference:
    LPCMCI-based causal discovery for constructing prior causal matrices.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class PaperStyleCausalUnit(nn.Module):
    """
    Paper-style Causal Weighting Unit: Token-wise Feature Weighting.

    This module applies learned causal weights to feature tokens based on
    prior causal knowledge extracted from LPCMCI analysis. The weights are
    computed as a parameterized function of the causal prior matrix.
    """

    def __init__(self, n_num_features, causal_prior_flattened):
        super(PaperStyleCausalUnit, self).__init__()
        # causal_prior_flattened shape: (window_size * n_features)
        self.register_buffer('causal_prior', torch.FloatTensor(causal_prior_flattened))

        # Learnable parameters: w * prior + bias
        self.learnable_w = nn.Parameter(torch.ones(n_num_features))
        self.bias = nn.Parameter(torch.ones(n_num_features) * 0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tokens):
        """
        Args:
            tokens: (batch, n_num_features, d_model)
        Returns:
            weighted_tokens: Tokens after applying causal weights
            weights: The computed causal weights
        """
        # Compute weights for each feature token
        causal_weights = self.sigmoid(self.learnable_w * self.causal_prior + self.bias)
        # Apply weights to each element in the d_model dimension
        # (batch, n_num_features, d_model) * (1, n_num_features, 1)
        return tokens * causal_weights.view(1, -1, 1), causal_weights


class CausalFTTransformer(nn.Module):
    """
    Causal Feature Tokenizer Transformer.

    An enhanced FT-Transformer architecture that integrates prior causal
    knowledge from causal discovery (LPCMCI) into the feature attention
    mechanism. The model learns to modulate feature importance based on
    both data-driven patterns and domain knowledge.
    """

    def __init__(self, window_size, n_features, causal_prior, d_model=64, n_heads=4,
                 num_layers=2, dim_feedforward=128, dropout=0.1,
                 use_location_token=True, location_embed_dim=32):
        super(CausalFTTransformer, self).__init__()

        self.window_size = window_size
        self.n_features = n_features
        self.n_num_features = window_size * n_features
        self.d_model = d_model
        self.use_location_token = use_location_token

        # 1. Feature Tokenizer (numerical feature projection)
        self.num_weight = nn.Parameter(torch.Tensor(self.n_num_features, d_model))
        self.num_bias = nn.Parameter(torch.Tensor(self.n_num_features, d_model))

        # 2. Causal Weighting Unit (causal guidance layer)
        # Flatten (window, n_features) prior to match token sequence
        prior_flat = causal_prior.flatten()
        self.causal_unit = PaperStyleCausalUnit(self.n_num_features, prior_flat)

        # 3. Location Token Encoder (geographic coordinate encoding)
        if use_location_token:
            self.location_mlp = nn.Sequential(
                nn.Linear(2, location_embed_dim),
                nn.GELU(),
                nn.Linear(location_embed_dim, d_model)
            )
            self.has_location_token = True
        else:
            self.has_location_token = False

        # 4. [CLS] token
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, d_model))

        # 5. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 6. Output Head
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.num_weight)
        nn.init.zeros_(self.num_bias)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        if self.has_location_token:
            for m in self.location_mlp:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, src, location_coords=None):
        batch_size = src.shape[0]

        # Step 1: Feature Tokenization -> (B, N_num, d_model)
        x_num = src.view(batch_size, self.n_num_features).unsqueeze(-1)
        num_tokens = x_num * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)

        # Step 2: Causal Weighting (token-wise weighting)
        num_tokens, _ = self.causal_unit(num_tokens)

        # Step 3: Token Concatenation
        tokens_list = []
        # Add [CLS] token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens_list.append(cls_token)
        # Add Location token
        if self.has_location_token and location_coords is not None:
            loc_token = self.location_mlp(location_coords).unsqueeze(1)
            tokens_list.append(loc_token)
        # Add Feature tokens
        tokens_list.append(num_tokens)

        combined_tokens = torch.cat(tokens_list, dim=1)

        # Step 4: Transformer Interaction
        x = self.transformer_encoder(combined_tokens)

        # Step 5: Output Head (using CLS token)
        return self.output_proj(self.dropout(x[:, 0, :]))


class TabularDataset(Dataset):
    """Dataset wrapper for Causal-FT-Transformer."""

    def __init__(self, X, site_ids, y, location_coords=None):
        self.X = torch.FloatTensor(X)
        self.site_ids = torch.LongTensor(site_ids)
        self.y = torch.FloatTensor(y).unsqueeze(1)
        self.location_coords = torch.FloatTensor(location_coords) if location_coords is not None else None

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        if self.location_coords is not None:
            return self.X[idx], self.site_ids[idx], self.y[idx], self.location_coords[idx]
        return self.X[idx], self.site_ids[idx], self.y[idx]
