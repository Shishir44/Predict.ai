# Synthetic Battery Dataset

## Dataset Statistics

- Total Batteries: 6
- Total Cycles: 2400
- SOH Range: 0.874 - 1.000
- Capacity Range: 1.746 - 2.000 Ah

## Features

- capacity_ah: Discharge capacity (Ah)
- voltage: Average voltage (V)
- current: Average current (A)
- temperature: Average temperature (C)
- soh: State of Health (target)
- cycle_number: Cycle count

## Usage

1. Load battery_dataset.csv
2. Use 'soh' as target variable
3. Train LSTM, Random Forest, or XGBoost
4. Split by battery_id for proper evaluation
