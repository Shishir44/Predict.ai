# Battery SOH/SOC Prediction System

A machine learning system for predicting Battery State of Health (SOH) and State of Charge (SOC) using time-series data.

## Project Overview

This project implements a dual-output machine learning model that can predict both SOH and SOC of lithium-ion batteries using historical cycling data. The system includes:
- Data processing and feature engineering
- LSTM and Transformer-based models
- Real-time prediction system
- Web-based monitoring interface

## Features

- Dual-output prediction for SOH and SOC
- Time-series feature engineering
- Multiple model architectures (LSTM, Transformer)
- Real-time monitoring dashboard
- Automated hyperparameter optimization
- Comprehensive evaluation metrics

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
battery_ml_project/
├── data/
│   ├── raw/          # Raw battery data
│   ├── processed/    # Processed features
│   └── synthetic/    # Synthetic data
├── models/
│   ├── soh_model/    # SOH-specific models
│   ├── soc_model/    # SOC-specific models
│   └── saved_models/ # Trained model checkpoints
├── src/
│   ├── data_processing/  # Data loading and preprocessing
│   ├── feature_engineering/  # Feature extraction
│   ├── model_training/  # Model architecture and training
│   ├── evaluation/     # Model evaluation
│   └── inference/      # Prediction system
├── notebooks/         # Jupyter notebooks for analysis
├── ui/               # Streamlit dashboard
├── config/           # Configuration files
├── tests/           # Test files
├── requirements.txt
├── setup.py
└── README.md
```

## Usage

1. Start the Streamlit dashboard:
```bash
streamlit run ui/streamlit_app.py
```

2. Access the dashboard at `http://localhost:8501`

## Development

The project follows Python best practices:
- PEP 8 style guide
- Type hints
- Comprehensive documentation
- Unit tests
- CI/CD pipeline

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments

- TensorFlow and PyTorch for deep learning implementations
- Streamlit for the web interface
- Optuna for hyperparameter optimization
- MLflow for model tracking