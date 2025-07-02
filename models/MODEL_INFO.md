# Battery Health Prediction Models

## Trained Models

### Random Forest Model
- **File:** random_forest_soh_model.joblib
- **RMSE:** 0.0005
- **MAE:** 0.0001
- **R²:** 0.9997

### LSTM Model
- **File:** lstm_soh_model.h5
- **RMSE:** 0.0577
- **MAE:** 0.0566
- **R²:** -3.0213

## Additional Files

- **feature_scaler.joblib** - StandardScaler for preprocessing

## Usage

1. Load the model files using joblib (RF) or tf.keras (LSTM)
2. Scale input features using the saved scaler
3. Make predictions using model.predict()
4. Models predict SOH (State of Health) values

## Input Features

- cycle_number: Battery cycle count
- capacity_ah: Discharge capacity (Ah)
- voltage: Average voltage (V)
- current: Average current (A)
- temperature: Temperature (°C)
- internal_resistance: Internal resistance (Ohm)
- energy_wh: Energy (Wh)
- efficiency: Efficiency (%)
- days_elapsed: Days since start
