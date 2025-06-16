"""
Model Evaluation Module

This module provides functionality for evaluating model performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class BatteryModelEvaluator:
    """
    Class for evaluating battery prediction models.
    
    Args:
        model: Trained prediction model
        X_test: Test input features
        y_true: Dictionary of true values for SOH and SOC
    """
    def __init__(self, model, X_test: np.ndarray, y_true: Dict[str, np.ndarray]):
        """Initialize the evaluator."""
        self.model = model
        self.X_test = X_test
        self.y_true = y_true
        self.predictions = None
        
    def evaluate_model(self) -> Dict:
        """
        Evaluate model performance.
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        self.predictions = self.model.predict(self.X_test)
        
        # Calculate metrics
        metrics = {
            'soh_metrics': self._calculate_metrics(
                self.y_true['soh'], self.predictions['soh_output'].flatten()
            ),
            'soc_metrics': self._calculate_metrics(
                self.y_true['soc'], self.predictions['soc_output'].flatten()
            )
        }
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def plot_predictions(self) -> plt.Figure:
        """
        Generate prediction plots.
        
        Returns:
            Matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # SOH predictions
        ax1.scatter(
            self.y_true['soh'],
            self.predictions['soh_output'].flatten(),
            alpha=0.6,
            label='Predictions'
        )
        ax1.plot([0, 1], [0, 1], 'r--', lw=2, label='Ideal')
        ax1.set_xlabel('True SOH')
        ax1.set_ylabel('Predicted SOH')
        ax1.set_title('SOH Predictions')
        ax1.legend()
        
        # SOC predictions
        ax2.scatter(
            self.y_true['soc'],
            self.predictions['soc_output'].flatten(),
            alpha=0.6,
            label='Predictions'
        )
        ax2.plot([0, 1], [0, 1], 'r--', lw=2, label='Ideal')
        ax2.set_xlabel('True SOC')
        ax2.set_ylabel('Predicted SOC')
        ax2.set_title('SOC Predictions')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_error_distribution(self) -> plt.Figure:
        """
        Plot error distribution for SOH and SOC predictions.
        
        Returns:
            Matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # SOH error distribution
        soh_errors = self.predictions['soh_output'].flatten() - self.y_true['soh']
        sns.histplot(soh_errors, kde=True, ax=ax1)
        ax1.set_title('SOH Prediction Error Distribution')
        ax1.set_xlabel('Error')
        ax1.set_ylabel('Frequency')
        
        # SOC error distribution
        soc_errors = self.predictions['soc_output'].flatten() - self.y_true['soc']
        sns.histplot(soc_errors, kde=True, ax=ax2)
        ax2.set_title('SOC Prediction Error Distribution')
        ax2.set_xlabel('Error')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig
