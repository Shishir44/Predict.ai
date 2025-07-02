#!/usr/bin/env python3
"""
Generate Synthetic Battery Dataset
=================================

Creates realistic synthetic battery aging data for immediate model training.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticBatteryGenerator:
    """Generate realistic synthetic battery aging data"""

    def __init__(self, data_dir: str = "data/raw/synthetic"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Battery parameters based on typical Li-ion 18650 cells
        self.nominal_capacity = 2.0  # Ah
        self.nominal_voltage = 3.7   # V

    def generate_battery_cycles(self, battery_id: str, num_cycles: int = 500) -> pd.DataFrame:
        """Generate synthetic cycling data for one battery"""

        logger.info(f"Generating {num_cycles} cycles for battery {battery_id}")

        cycles_data = []

        # Initial battery state
        degradation_rate = random.uniform(0.0001, 0.0003)

        for cycle in range(1, num_cycles + 1):
            # Battery degradation over time
            capacity_fade = 1 - (cycle * degradation_rate * random.uniform(0.8, 1.2))

            # Current cycle capacity
            cycle_capacity = self.nominal_capacity * capacity_fade

            # Generate realistic measurements with noise
            voltage_avg = 3.6 + random.uniform(-0.1, 0.1) - (cycle * 0.0001)
            current_avg = 2.0 + random.uniform(-0.2, 0.2)
            temperature_avg = 25 + random.uniform(-3, 8) + (cycle * 0.005)
            resistance = 0.05 + (cycle * 0.00001) + random.uniform(-0.005, 0.005)

            cycle_metrics = {
                'battery_id': battery_id,
                'cycle_number': cycle,
                'capacity_ah': cycle_capacity,
                'voltage': voltage_avg,
                'current': current_avg,
                'temperature': temperature_avg,
                'internal_resistance': resistance,
                'soh': cycle_capacity / self.nominal_capacity,  # State of Health
                'energy_wh': cycle_capacity * self.nominal_voltage,
                'efficiency': (cycle_capacity / self.nominal_capacity) * 100,
                'days_elapsed': cycle * random.uniform(0.5, 2.0),
            }

            cycles_data.append(cycle_metrics)

        return pd.DataFrame(cycles_data)

    def generate_dataset(self, num_batteries: int = 4, cycles_per_battery: int = 400) -> pd.DataFrame:
        """Generate complete multi-battery aging dataset"""

        logger.info(f"Generating dataset: {num_batteries} batteries, {cycles_per_battery} cycles each")

        all_data = []

        for i in range(num_batteries):
            battery_id = f"SYNTH_B{i+1:03d}"
            battery_data = self.generate_battery_cycles(battery_id, cycles_per_battery)
            all_data.append(battery_data)

        # Combine all battery data
        dataset = pd.concat(all_data, ignore_index=True)

        # Add measurement noise
        noise_cols = ['capacity_ah', 'voltage', 'current', 'temperature']
        for col in noise_cols:
            noise = np.random.normal(0, dataset[col].std() * 0.01, len(dataset))
            dataset[col] = dataset[col] + noise

        # Save dataset
        dataset_file = self.data_dir / 'battery_dataset.csv'
        dataset.to_csv(dataset_file, index=False)

        logger.info(f"Generated dataset with {len(dataset)} data points")
        logger.info(f"Saved to: {dataset_file}")

        return dataset

    def create_info_file(self, dataset: pd.DataFrame):
        """Create dataset information file"""

        info_file = self.data_dir / 'DATASET_INFO.md'
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("# Synthetic Battery Dataset\n\n")
            f.write("## Dataset Statistics\n\n")
            f.write(f"- Total Batteries: {dataset['battery_id'].nunique()}\n")
            f.write(f"- Total Cycles: {len(dataset)}\n")
            f.write(f"- SOH Range: {dataset['soh'].min():.3f} - {dataset['soh'].max():.3f}\n")
            f.write(f"- Capacity Range: {dataset['capacity_ah'].min():.3f} - {dataset['capacity_ah'].max():.3f} Ah\n\n")

            f.write("## Features\n\n")
            f.write("- capacity_ah: Discharge capacity (Ah)\n")
            f.write("- voltage: Average voltage (V)\n")
            f.write("- current: Average current (A)\n")
            f.write("- temperature: Average temperature (C)\n")
            f.write("- soh: State of Health (target)\n")
            f.write("- cycle_number: Cycle count\n\n")

            f.write("## Usage\n\n")
            f.write("1. Load battery_dataset.csv\n")
            f.write("2. Use 'soh' as target variable\n")
            f.write("3. Train LSTM, Random Forest, or XGBoost\n")
            f.write("4. Split by battery_id for proper evaluation\n")


def main():
    """Generate synthetic battery dataset"""
    print("Synthetic Battery Dataset Generator")
    print("=" * 40)

    generator = SyntheticBatteryGenerator()
    dataset = generator.generate_dataset(num_batteries=6, cycles_per_battery=400)
    generator.create_info_file(dataset)

    print(f"\nDataset Generated!")
    print(f"Location: {generator.data_dir}")
    print(f"Records: {len(dataset)} cycles from {dataset['battery_id'].nunique()} batteries")
    print(f"SOH range: {dataset['soh'].min():.3f} - {dataset['soh'].max():.3f}")
    print("Ready for training!")


if __name__ == "__main__":
    main()
