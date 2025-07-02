"""
Battery ML Model Module

This module provides implementations of machine learning models for predicting
Battery State of Health (SOH) and State of Charge (SOC).
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import torch
import torch.nn as nn
from typing import Dict

class BatteryLSTMModel(tf.keras.Model):
    """
    LSTM-based model for predicting SOH and SOC.
    
    Args:
        sequence_length: Length of input sequences
        num_features: Number of input features
    """
    def __init__(self, sequence_length: int, num_features: int):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_features = num_features
        
        # Shared LSTM layers
        self.lstm1 = layers.LSTM(128, return_sequences=True)
        self.lstm2 = layers.LSTM(64, return_sequences=False)
        self.dropout = layers.Dropout(0.2)
        
        # SOH prediction branch
        self.soh_dense1 = layers.Dense(32, activation='relu')
        self.soh_output = layers.Dense(1, activation='sigmoid', name='soh_output')
        
        # SOC prediction branch
        self.soc_dense1 = layers.Dense(32, activation='relu')
        self.soc_output = layers.Dense(1, activation='sigmoid', name='soc_output')
        
    def call(self, inputs, training=None):
        """
        Forward pass of the model.
        
        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, num_features)
            training: Boolean indicating whether in training mode
            
        Returns:
            Dictionary containing SOH and SOC predictions
        """
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.dropout(x, training=training)
        
        # SOH branch
        soh_x = self.soh_dense1(x)
        soh_pred = self.soh_output(soh_x)
        
        # SOC branch
        soc_x = self.soc_dense1(x)
        soc_pred = self.soc_output(soc_x)
        
        return {'soh_output': soh_pred, 'soc_output': soc_pred}

class BatteryTransformerModel(nn.Module):
    """
    Transformer-based model for predicting SOH and SOC.
    
    Args:
        num_features: Number of input features
        d_model: Dimension of model
    """
    def __init__(self, num_features: int, d_model: int = 128):
        super().__init__()
        self.embedding = nn.Linear(num_features, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8), num_layers=6
        )
        self.soh_head = nn.Linear(d_model, 1)
        self.soc_head = nn.Linear(d_model, 1)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features)
            
        Returns:
            Dictionary containing SOH and SOC predictions
        """
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        
        soh = torch.sigmoid(self.soh_head(x))
        soc = torch.sigmoid(self.soc_head(x))
        
        return {'soh': soh, 'soc': soc}
