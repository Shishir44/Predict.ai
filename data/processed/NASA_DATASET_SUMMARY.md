# NASA Comprehensive Battery Dataset Summary

## Dataset Overview

- **Total Records:** 2769
- **Unique Batteries:** 34
- **Temperature Conditions:** [4, 22, 24, 43, 44]
- **Capacity Range:** 0.000 - 2.640 Ah
- **SOH Range:** 0.000 - 27.550

## Battery Statistics

| Battery | Cycles | Max Cycle | Min Capacity | Max Capacity | Min SOH | Max SOH | Temp |
|---------|--------|-----------|--------------|--------------|---------|---------|------|
| B0005 | 168 | 614 | 1.287 | 1.856 | 0.693 | 1.000 | 24.0°C |
| B0006 | 168 | 614 | 1.154 | 2.035 | 0.567 | 1.000 | 24.0°C |
| B0007 | 168 | 614 | 1.400 | 1.891 | 0.741 | 1.000 | 24.0°C |
| B0018 | 132 | 319 | 1.341 | 1.855 | 0.723 | 1.000 | 24.0°C |
| B0025 | 28 | 78 | 1.768 | 1.849 | 0.957 | 1.001 | 24.0°C |
| B0026 | 28 | 78 | 1.386 | 1.817 | 0.765 | 1.002 | 24.0°C |
| B0027 | 28 | 78 | 1.770 | 1.823 | 0.971 | 1.000 | 24.0°C |
| B0028 | 28 | 78 | 1.717 | 1.805 | 0.952 | 1.000 | 24.0°C |
| B0029 | 40 | 94 | 1.612 | 1.845 | 0.950 | 1.087 | 43.0°C |
| B0030 | 40 | 94 | 1.563 | 1.782 | 0.944 | 1.076 | 43.0°C |
| B0031 | 40 | 94 | 1.667 | 1.833 | 1.000 | 1.100 | 43.0°C |
| B0032 | 40 | 94 | 1.636 | 1.894 | 0.959 | 1.111 | 43.0°C |
| B0033 | 197 | 483 | 0.068 | 1.885 | 1.000 | 27.550 | 24.0°C |
| B0034 | 197 | 483 | 0.746 | 1.820 | 1.000 | 2.440 | 24.0°C |
| B0036 | 197 | 483 | 1.002 | 2.444 | 1.000 | 2.439 | 24.0°C |
| B0038 | 47 | 121 | 0.898 | 1.786 | 1.000 | 1.989 | 38.9°C |
| B0039 | 47 | 121 | 0.119 | 1.772 | 1.000 | 14.882 | 38.9°C |
| B0040 | 47 | 121 | 0.426 | 1.734 | 0.632 | 2.575 | 38.9°C |
| B0041 | 67 | 162 | 0.044 | 1.216 | 0.793 | 21.856 | 4.0°C |
| B0042 | 112 | 274 | 0.000 | 1.731 | 0.000 | 1.002 | 10.6°C |
| B0043 | 112 | 274 | 0.000 | 1.714 | 0.000 | 1.000 | 10.6°C |
| B0044 | 112 | 274 | 0.000 | 1.687 | 0.000 | 1.000 | 10.6°C |
| B0045 | 72 | 181 | 0.000 | 1.082 | 0.000 | 1.000 | 4.0°C |
| B0046 | 72 | 181 | 0.000 | 1.728 | 0.000 | 1.000 | 4.0°C |
| B0047 | 72 | 181 | 0.000 | 1.674 | 0.000 | 1.000 | 4.0°C |
| B0048 | 72 | 181 | 0.000 | 1.658 | 0.000 | 1.000 | 4.0°C |
| B0049 | 25 | 59 | 0.000 | 2.379 | 0.000 | 2.771 | 4.0°C |
| B0050 | 21 | 51 | 0.000 | 2.640 | 0.000 | 3.059 | 4.0°C |
| B0051 | 25 | 59 | 0.000 | 2.331 | 0.000 | 3.622 | 4.0°C |
| B0052 | 4 | 9 | 0.861 | 1.418 | 1.000 | 1.648 | 4.0°C |
| B0053 | 56 | 137 | 0.000 | 1.154 | 0.000 | 1.079 | 4.0°C |
| B0054 | 103 | 253 | 0.000 | 1.167 | 0.000 | 1.577 | 4.0°C |
| B0055 | 102 | 251 | 0.799 | 1.320 | 1.000 | 1.652 | 4.0°C |
| B0056 | 102 | 251 | 0.785 | 1.344 | 1.000 | 1.712 | 4.0°C |

## Features Available

- **cycle_number**: int64
- **ambient_temperature**: int64
- **capacity_ah**: float64
- **soh**: float64
- **voltage_max**: float64
- **voltage_min**: float64
- **voltage_mean**: float64
- **voltage_std**: float64
- **voltage_range**: float64
- **current_max**: float64
- **current_mean**: float64
- **current_std**: float64
- **temperature_max**: float64
- **temperature_min**: float64
- **temperature_mean**: float64
- **temperature_std**: float64
- **temperature_rise**: float64
- **discharge_time**: float64
- **energy_wh**: float64
- **efficiency**: float64
- **internal_resistance_est**: float64
- **data_points**: int64
- **data_duration**: float64
- **soh_relative**: float64

## Recommended Usage

1. **Target Variable**: `soh_relative` (Relative State of Health)
2. **Train/Test Split**: Split by battery_id to ensure proper evaluation
3. **Features**: All columns except battery_id and soh_relative
4. **Models**: LSTM, Random Forest, XGBoost, GRU
5. **Cross-validation**: Group by battery_id

## Temperature Conditions

- **4°C**: 16 batteries
- **22°C**: 3 batteries
- **24°C**: 14 batteries
- **43°C**: 4 batteries
- **44°C**: 3 batteries
