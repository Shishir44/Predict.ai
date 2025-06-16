import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import tensorflow as tf
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelExplainer:
    def __init__(self, 
                model_dir: str = "models/ensemble",
                explanation_dir: str = "explanations"):
        self.model_dir = Path(model_dir)
        self.explanation_dir = Path(explanation_dir)
        self.explanation_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model(self, model_type: str = "lstm") -> Any:
        """Load model for explanation."""
        if model_type == "lstm":
            model_path = self.model_dir / 'lstm_model.h5'
            if not model_path.exists():
                raise FileNotFoundError(f"LSTM model not found at {model_path}")
                
            return tf.keras.models.load_model(str(model_path))
            
        elif model_type == "transformer":
            model_path = self.model_dir / 'transformer_model.pth'
            if not model_path.exists():
                raise FileNotFoundError(f"Transformer model not found at {model_path}")
                
            model = TransformerModel(input_dim=10)  # Adjust input_dim as needed
            model.load_state_dict(torch.load(model_path))
            model.eval()
            return model
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def create_lime_explainer(self, X_train: np.ndarray) -> LimeTabularExplainer:
        """Create LIME explainer."""
        return LimeTabularExplainer(
            X_train,
            feature_names=['capacity', 'voltage', 'current', 'temperature'],
            class_names=['SOH', 'SOC'],
            mode='regression'
        )
        
    def explain_instance_lime(self, 
                            model: Any,
                            X: np.ndarray,
                            explainer: LimeTabularExplainer,
                            instance_idx: int) -> Dict[str, Any]:
        """Explain a single instance using LIME."""
        instance = X[instance_idx]
        
        # Create explanation
        explanation = explainer.explain_instance(
            instance,
            model.predict,
            num_features=4
        )
        
        # Extract feature importance
        soh_importance = {
            feature: importance
            for feature, importance in explanation.as_list(label=0)
        }
        
        soc_importance = {
            feature: importance
            for feature, importance in explanation.as_list(label=1)
        }
        
        return {
            'instance': instance_idx,
            'soh_importance': soh_importance,
            'soc_importance': soc_importance
        }
        
    def create_shap_explainer(self, model: Any, X_train: np.ndarray) -> shap.Explainer:
        """Create SHAP explainer."""
        if isinstance(model, tf.keras.Model):
            return shap.KernelExplainer(model.predict, X_train)
        else:
            return shap.KernelExplainer(model.predict, X_train)
            
    def explain_instance_shap(self, 
                            explainer: shap.Explainer,
                            instance: np.ndarray) -> Dict[str, Any]:
        """Explain a single instance using SHAP."""
        shap_values = explainer.shap_values(instance)
        
        return {
            'soh_shap': shap_values[0].tolist(),
            'soc_shap': shap_values[1].tolist()
        }
        
    def plot_feature_importance(self, 
                              importance: Dict[str, float],
                              output_path: str,
                              title: str) -> None:
        """Plot feature importance."""
        plt.figure(figsize=(10, 6))
        features = list(importance.keys())
        values = list(importance.values())
        
        sns.barplot(x=values, y=features)
        plt.title(title)
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def generate_global_explanation(self, 
                                  model: Any,
                                  X: np.ndarray,
                                  y_true: np.ndarray) -> Dict[str, Any]:
        """Generate global model explanation."""
        # Create SHAP explainer
        explainer = self.create_shap_explainer(model, X)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        # Calculate mean absolute SHAP values
        soh_importance = {
            'capacity': np.abs(shap_values[0][:, 0]).mean(),
            'voltage': np.abs(shap_values[0][:, 1]).mean(),
            'current': np.abs(shap_values[0][:, 2]).mean(),
            'temperature': np.abs(shap_values[0][:, 3]).mean()
        }
        
        soc_importance = {
            'capacity': np.abs(shap_values[1][:, 0]).mean(),
            'voltage': np.abs(shap_values[1][:, 1]).mean(),
            'current': np.abs(shap_values[1][:, 2]).mean(),
            'temperature': np.abs(shap_values[1][:, 3]).mean()
        }
        
        # Plot feature importance
        self.plot_feature_importance(
            soh_importance,
            self.explanation_dir / 'soh_global_importance.png',
            'Global SOH Feature Importance'
        )
        
        self.plot_feature_importance(
            soc_importance,
            self.explanation_dir / 'soc_global_importance.png',
            'Global SOC Feature Importance'
        )
        
        # Create SHAP summary plots
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values[0], X, show=False)
        plt.savefig(self.explanation_dir / 'soh_summary_plot.png')
        plt.close()
        
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values[1], X, show=False)
        plt.savefig(self.explanation_dir / 'soc_summary_plot.png')
        plt.close()
        
        return {
            'soh_importance': soh_importance,
            'soc_importance': soc_importance,
            'shap_values': shap_values
        }
        
    def generate_local_explanation(self, 
                                 model: Any,
                                 X: np.ndarray,
                                 instance_idx: int) -> Dict[str, Any]:
        """Generate local explanation for a specific instance."""
        # Create LIME explainer
        explainer = self.create_lime_explainer(X)
        
        # Explain instance
        lime_explanation = self.explain_instance_lime(
            model,
            X,
            explainer,
            instance_idx
        )
        
        # Create SHAP explainer
        shap_explainer = self.create_shap_explainer(model, X)
        
        # Explain instance with SHAP
        shap_explanation = self.explain_instance_shap(
            shap_explainer,
            X[instance_idx]
        )
        
        return {
            'lime': lime_explanation,
            'shap': shap_explanation
        }
        
    def save_explanation(self, explanation: Dict[str, Any], filename: str) -> None:
        """Save explanation to file."""
        with open(self.explanation_dir / filename, 'w') as f:
            json.dump(explanation, f, indent=4)
            
    def create_explanation_report(self, 
                                global_explanation: Dict[str, Any],
                                local_explanation: Dict[str, Any],
                                instance_idx: int) -> None:
        """Create explanation report."""
        # Create report content
        report = {
            'timestamp': datetime.now().isoformat(),
            'instance': instance_idx,
            'global_explanation': global_explanation,
            'local_explanation': local_explanation
        }
        
        # Save report
        self.save_explanation(report, 'explanation_report.json')
        
    def explain_model(self, instance_idx: int = None) -> None:
        """Explain model predictions."""
        try:
            # Load model
            model = self.load_model()
            
            # Load data
            preprocessor = BatteryDataPreprocessor()
            X, y_soh, y_soc = preprocessor.prepare_data_for_training()[0:3]
            
            # Generate global explanation
            global_explanation = self.generate_global_explanation(model, X, y_soh)
            
            # Generate local explanation if instance_idx is provided
            local_explanation = None
            if instance_idx is not None:
                local_explanation = self.generate_local_explanation(model, X, instance_idx)
                
            # Create report
            self.create_explanation_report(global_explanation, local_explanation, instance_idx)
            
            logger.info("Model explanation completed successfully")
            
        except Exception as e:
            logger.error(f"Error explaining model: {str(e)}")
            raise

def main():
    """Main function to run model explanation."""
    explainer = ModelExplainer()
    
    # Explain model for a specific instance
    instance_idx = 42  # Example instance
    explainer.explain_model(instance_idx)

if __name__ == "__main__":
    main()
