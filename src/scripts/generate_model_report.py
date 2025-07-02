import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelReporter:
    def __init__(self, 
                model_dir: str = "models/ensemble",
                report_dir: str = "reports",
                template_dir: str = "templates"):
        self.model_dir = Path(model_dir)
        self.report_dir = Path(report_dir)
        self.template_dir = Path(template_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model_info(self) -> Dict:
        """Load model information."""
        # Load version info
        version_file = self.model_dir / 'versions.json'
        if version_file.exists():
            with open(version_file, 'r') as f:
                versions = json.load(f)
                current_version = versions.get('current_version', None)
                version_info = versions.get('versions', {}).get(current_version, {})
                return version_info
        return {}
        
    def generate_performance_metrics(self) -> Dict[str, float]:
        """Generate performance metrics."""
        validator = ModelValidator()
        validation_results = validator.validate_model()
        return validation_results['metrics']
        
    def plot_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Plot performance metrics."""
        # Create bar plot
        plt.figure(figsize=(12, 6))
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        sns.barplot(data=metrics_df, x='Metric', y='Value')
        plt.title('Model Performance Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.report_dir / 'performance_metrics.png')
        plt.close()
        
    def generate_feature_importance(self) -> Dict[str, float]:
        """Generate feature importance analysis."""
        # Load validation data
        preprocessor = BatteryDataPreprocessor()
        X, _, _ = preprocessor.prepare_data_for_training()[0:3]
        
        # Calculate feature importance using permutation importance
        from sklearn.inspection import permutation_importance
        model = tf.keras.models.load_model(self.model_dir / 'lstm_model.h5')
        
        # Get feature names
        feature_names = ['capacity', 'voltage', 'current', 'temperature']
        
        # Calculate importance
        importance = permutation_importance(
            model,
            X,
            y_soh,  # Using SOH as target
            n_repeats=10,
            random_state=42
        )
        
        # Create importance dictionary
        return {
            feature: importance.importances_mean[i]
            for i, feature in enumerate(feature_names)
        }
        
    def plot_feature_importance(self, importance: Dict[str, float]) -> None:
        """Plot feature importance."""
        # Create bar plot
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
        sns.barplot(data=importance_df, x='Importance', y='Feature')
        plt.title('Feature Importance Analysis')
        plt.tight_layout()
        plt.savefig(self.report_dir / 'feature_importance.png')
        plt.close()
        
    def generate_model_architecture(self) -> Dict[str, Any]:
        """Generate model architecture information."""
        model = tf.keras.models.load_model(self.model_dir / 'lstm_model.h5')
        
        # Get layer information
        layers = []
        for layer in model.layers:
            layers.append({
                'name': layer.name,
                'type': type(layer).__name__,
                'output_shape': layer.output_shape,
                'parameters': layer.count_params()
            })
            
        return {
            'total_parameters': model.count_params(),
            'layers': layers
        }
        
    def generate_model_report(self) -> None:
        """Generate comprehensive model report."""
        # Load model info
        model_info = self.load_model_info()
        
        # Generate performance metrics
        metrics = self.generate_performance_metrics()
        self.plot_performance_metrics(metrics)
        
        # Generate feature importance
        importance = self.generate_feature_importance()
        self.plot_feature_importance(importance)
        
        # Generate model architecture
        architecture = self.generate_model_architecture()
        
        # Create report content
        report_content = {
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_info': model_info,
            'performance_metrics': metrics,
            'feature_importance': importance,
            'architecture': architecture
        }
        
        # Save report content
        with open(self.report_dir / 'model_report.json', 'w') as f:
            json.dump(report_content, f, indent=4)
            
        # Generate HTML report using template
        env = Environment(loader=FileSystemLoader(self.template_dir))
        template = env.get_template('model_report_template.html')
        
        # Render template
        html_content = template.render(
            report_content=report_content,
            performance_plot=str(self.report_dir / 'performance_metrics.png'),
            importance_plot=str(self.report_dir / 'feature_importance.png')
        )
        
        # Save HTML report
        with open(self.report_dir / 'model_report.html', 'w') as f:
            f.write(html_content)
            
        logger.info("Model report generated successfully")
        
    def create_report_template(self) -> None:
        """Create HTML template for model report."""
        template_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Battery Prediction Model Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .section { margin-bottom: 40px; }
        .header { color: #333; font-size: 24px; margin-bottom: 20px; }
        .subheader { color: #666; font-size: 18px; margin-bottom: 15px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .metric-card { padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .metric-label { color: #666; font-size: 14px; }
        .metric-value { font-size: 18px; font-weight: bold; }
        .image-container { margin-top: 20px; }
    </style>
</head>
<body>
    <div class="section">
        <div class="header">Model Report</div>
        <div class="subheader">Generated: {{ report_content.date }}</div>
    </div>
    
    <div class="section">
        <div class="header">Model Information</div>
        <div class="subheader">Current Version: {{ report_content.model_info.version }}</div>
        <div class="subheader">Timestamp: {{ report_content.model_info.timestamp }}</div>
        <div class="subheader">Model Type: {{ report_content.model_info.model_type }}</div>
    </div>
    
    <div class="section">
        <div class="header">Performance Metrics</div>
        <div class="metrics">
            {% for metric, value in report_content.performance_metrics.items() %}
            <div class="metric-card">
                <div class="metric-label">{{ metric }}</div>
                <div class="metric-value">{{ "%.4f"|format(value) }}</div>
            </div>
            {% endfor %}
        </div>
        <div class="image-container">
            <img src="{{ performance_plot }}" alt="Performance Metrics">
        </div>
    </div>
    
    <div class="section">
        <div class="header">Feature Importance</div>
        <div class="image-container">
            <img src="{{ importance_plot }}" alt="Feature Importance">
        </div>
    </div>
    
    <div class="section">
        <div class="header">Model Architecture</div>
        <div class="subheader">Total Parameters: {{ report_content.architecture.total_parameters }}</div>
        <div class="subheader">Layers:</div>
        <ul>
            {% for layer in report_content.architecture.layers %}
            <li>
                <strong>{{ layer.name }} ({{ layer.type }})</strong><br>
                Output Shape: {{ layer.output_shape }}<br>
                Parameters: {{ layer.parameters }}
            </li>
            {% endfor %}
        </ul>
    </div>
</body>
</html>
"""
        
        # Create templates directory if it doesn't exist
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Save template
        with open(self.template_dir / 'model_report_template.html', 'w') as f:
            f.write(template_content)
            
        logger.info("Created model report template")

def main():
    """Main function to generate model report."""
    reporter = ModelReporter()
    
    # Create report template if it doesn't exist
    if not (reporter.template_dir / 'model_report_template.html').exists():
        reporter.create_report_template()
        
    # Generate report
    reporter.generate_model_report()
    logger.info("Model report generation completed")

if __name__ == "__main__":
    main()
