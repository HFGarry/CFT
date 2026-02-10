#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Model Architecture.

This module implements a standard Long Short-Term Memory network for 
time-series groundwater level prediction. Implements a sequence-to-point 
regression architecture.

Key Components:
    - LSTMModel: Main LSTM regression model

Reference:
    Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. 
    Neural computation, 9(8), 1735-1780.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM-based Regression Model.

    Architecture:
        - LSTM Layer: Captures temporal dependencies in sequential input
        - Dropout Layer: Regularization to prevent overfitting
        - Dense Layer (32 units, ReLU): Non-linear transformation
        - Output Layer (1 unit): Final regression prediction
    """

    def __init__(self, input_size, hidden_size=64, dropout=0.2):
        super(LSTMModel, self).__init__()
        # Input tensor shape: (batch_size, sequence_length, input_features)
        # batch_first=True configures LSTM to accept (batch, seq, features) format
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        """
        Forward pass computation.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            Prediction tensor of shape (batch_size, 1)
        """
        # LSTM returns tuple of (output_sequence, (hidden_state, cell_state))
        # output_sequence shape: (batch, seq_len, hidden_size)
        out, _ = self.lstm(x)

        # Extract hidden state from the final time step for sequence-to-point prediction
        out = out[:, -1, :]

        # Apply regularization and non-linear transformation
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        return self.fc2(out)
