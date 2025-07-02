# Battery Prediction Datasets Overview

## Recommended Datasets for SOH/SOC Prediction

### CRITICAL Priority

#### BatteryLife Dataset (KDD 2025)
- **Chemistry:** Li-ion, Zn-ion, Na-ion
- **Cells:** 2.5x larger than previous datasets
- **Description:** Most comprehensive dataset with 16 integrated datasets

### HIGH Priority

#### NASA Prognostics Center Battery Data
- **Chemistry:** Li-ion 18650
- **Cells:** 4
- **Description:** Li-ion battery aging data from NASA Ames
- **URL:** https://c3.nasa.gov/dashlink/resources/133/

#### Oxford Battery Degradation Dataset
- **Chemistry:** Li-ion commercial
- **Cells:** 8
- **Description:** Commercial Li-ion battery degradation
- **URL:** https://ora.ox.ac.uk/objects/uuid:03ba4734-3d73-4bdc-8f3e-ffb50b70f6ba

## Usage Instructions

1. **Start with BatteryLife Dataset (KDD 2025)** - Most comprehensive
2. **Add NASA data** - Most validated and benchmarked
3. **Include Oxford data** - For path-dependent studies

## Directory Structure

```
data/raw/
├── batterylife/     # KDD 2025 - Most comprehensive
├── nasa/           # NASA Prognostics Center
├── oxford/         # Oxford University
└── processed/      # Preprocessed data
```
