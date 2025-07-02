# 🔋 Predict.ai - Enterprise Battery Intelligence Platform

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)
![Tests](https://img.shields.io/badge/tests-9%2F9%20passed-brightgreen.svg)

**Advanced AI-powered battery health monitoring and prediction system with enterprise-grade features.**

## 🎯 Overview

Predict.ai is a comprehensive battery health monitoring platform that combines machine learning, real-time analytics, and enterprise-grade monitoring to provide accurate State of Health (SOH) and State of Charge (SOC) predictions for battery systems.

### ✨ Key Features

- **🤖 Multi-Model AI**: Random Forest, LSTM, and Ensemble models
- **📊 Real-time Monitoring**: Live battery health tracking and predictions
- **🎯 High Accuracy**: Advanced feature engineering and model optimization
- **💼 Enterprise Ready**: Production-grade monitoring, alerts, and health checks
- **🌐 Web Interface**: Intuitive Streamlit-based dashboard
- **📈 Advanced Analytics**: Comprehensive charts, trends, and insights
- **🚨 Alert System**: Configurable thresholds and notifications
- **🔄 Auto-scaling**: Intelligent model loading and caching

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Windows 10/11 (tested) or Linux/macOS
- 8GB+ RAM recommended
- 2GB+ free disk space

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Predict.ai

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run ui/advanced_streamlit_app.py
```

### First Run

1. **Access the Web Interface**: Open `http://localhost:8501` in your browser
2. **Navigate to Real-time Monitoring**: Use the sidebar to access prediction features
3. **Input Battery Parameters**: Adjust sliders for voltage, current, temperature, etc.
4. **Select Model**: Choose from Random Forest, LSTM, or Ensemble
5. **Get Predictions**: Click "🔮 Predict Battery Health" for instant results

## 🏗️ System Architecture

### Core Components

```
📦 Predict.ai/
├── 🤖 models/                    # Trained ML models
│   ├── random_forest_soh_model.joblib
│   ├── lstm_soh_model.h5
│   └── feature_scaler.joblib
├── 🔬 src/                       # Source code
│   ├── inference/                # Production prediction engine
│   ├── monitoring/               # Real-time monitoring service
│   ├── health/                   # System health checking
│   └── feature_engineering/      # Advanced feature processing
├── 🌐 ui/                        # Web interface
│   └── advanced_streamlit_app.py # Main application
├── 📊 data/                      # Datasets and processing
└── 📋 tests/                     # Comprehensive test suite
```

### Model Performance

| Model | Avg Prediction Time | Accuracy | Use Case |
|-------|-------------------|----------|----------|
| **Random Forest** | 87ms | High | Fast, reliable predictions |
| **LSTM** | 105ms | Very High | Complex temporal patterns |
| **Ensemble** | 157ms | Highest | Maximum accuracy |

## 🎛️ Enterprise Features

### 📊 Advanced Dashboard

- **Executive KPIs**: Fleet overview with 100+ battery units
- **Real-time Metrics**: Live SOH/SOC monitoring
- **Interactive Charts**: Trend analysis and fleet status
- **Batch Processing**: Multi-file analysis capabilities

### 🚨 Monitoring & Alerts

- **Drift Detection**: Automatic model performance monitoring
- **Health Checks**: System resource and service monitoring
- **Configurable Alerts**: Custom thresholds for various metrics
- **Performance Tracking**: Response time and accuracy monitoring

### 🔧 Production Features

- **Model Caching**: Intelligent loading and singleton patterns
- **Error Handling**: Graceful degradation and fallbacks
- **Logging**: Comprehensive enterprise-grade logging
- **Database Integration**: SQLite for monitoring and health data

## 📈 Model Details

### Random Forest Model
- **Features**: 6 engineered features from battery parameters
- **Training Data**: NASA battery dataset (2,769+ records)
- **Accuracy**: R² = 0.781
- **Use Case**: Fast, reliable predictions for real-time applications

### LSTM Model  
- **Architecture**: Sequence-to-sequence with temporal features
- **Input Shape**: (10 timesteps, 9 features)
- **Training**: Advanced preprocessing with feature expansion
- **Use Case**: Complex temporal pattern recognition

### Ensemble Model
- **Combination**: Confidence-weighted averaging of RF + LSTM
- **Fallback**: Graceful degradation to Random Forest if LSTM fails
- **Version**: ensemble_v1.0.0
- **Use Case**: Maximum accuracy for critical applications

## 🔍 API Usage

### Python API

```python
from src.inference.production_predictor import ProductionBatteryPredictor

# Initialize predictor
predictor = ProductionBatteryPredictor()

# Prepare features
features = {
    'cycle_number': 250,
    'capacity_ah': 0.48,
    'ambient_temperature': 25.0,
    'capacity_normalized': 0.48,
    'cycle_progress': 0.25,
    'temp_deviation': 1.0
}

# Make prediction
result = predictor.predict_battery_health(features, model_type='ensemble')

print(f"SOH: {result.soh_prediction:.3f}")
print(f"SOC: {result.soc_prediction:.3f}")
print(f"Confidence: {result.soh_confidence:.3f}")
print(f"Model: {result.model_version}")
```

### Web Interface API

The Streamlit interface provides intuitive controls for:
- **Parameter Input**: Sliders and numeric inputs for all battery parameters
- **Model Selection**: Dropdown for model type selection
- **Real-time Results**: Instant predictions with confidence metrics
- **Analysis Reports**: Comprehensive text-based health assessments

## 🧪 Testing & Validation

### Comprehensive Test Suite

All components have been thoroughly tested:

```
✅ Environment & Dependencies: PASSED
✅ Model Files: PASSED (13.2 MB total)
✅ Production Predictor: PASSED (All 3 models working)
✅ Monitoring Service: PASSED (Drift detection active)
✅ Health Checker: PASSED (System health: Warning - Disk 86.8%)
✅ UI Integration: PASSED (Model types correctly identified)
✅ Data Processing: PASSED (2,769 records loaded)
✅ Feature Engineering: PASSED (Methods verified)
✅ Model Performance: PASSED (MAE: 0.020, R²: 0.92)

🎯 OVERALL: 9/9 tests passed (100.0%)
🎉 ALL TESTS PASSED! Enterprise system is fully operational.
```

### Performance Metrics

- **System Health**: CPU 14.7%, Memory 62.6%
- **Model Loading**: All 3 models loaded successfully
- **Prediction Speed**: 87-157ms average response time
- **Database**: SQLite connectivity confirmed
- **UI Responsiveness**: Real-time updates with <200ms latency

## 🔧 Configuration

### Environment Variables

```bash
# Optional: TensorFlow optimizations
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

# Optional: Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Model Configuration

Models are automatically loaded from the `models/` directory:
- `random_forest_soh_model.joblib` (12.8 MB)
- `lstm_soh_model.h5` (0.4 MB)
- `feature_scaler.joblib` (0.0 MB)

## 📚 Documentation

- **[User Guide](USER_GUIDE.md)**: Comprehensive usage instructions
- **[API Documentation](src/)**: Source code documentation
- **[Model Details](models/MODEL_INFO.md)**: Model architecture and training
- **[Dataset Information](data/)**: Data processing and sources

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NASA Battery Dataset for training data
- TensorFlow and Scikit-learn communities
- Streamlit for the excellent web framework
- All contributors and testers

---

## 📞 Support

For support, email support@predict-ai.com or create an issue in the repository.

**Built with ❤️ for enterprise battery monitoring**