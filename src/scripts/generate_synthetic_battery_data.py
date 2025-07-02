#!/usr/bin/env python3
"""
Generate Synthetic Battery Dataset
=================================

Creates realistic synthetic battery aging data that matches the NASA dataset format.
This allows immediate model training and testing while waiting for real datasets.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
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
        self.nominal_voltage = 3.7  # V
        self.max_voltage = 4.2      # V
        self.min_voltage = 2.5      # V
        self.nominal_capacity = 2.0  # Ah
        self.nominal_energy = 7.4   # Wh

    def generate_battery_cycles(self, battery_id: str, num_cycles: int = 500,
                               degradation_rate: float = 0.0002) -> pd.DataFrame:
        """Generate synthetic cycling data for one battery"""

        logger.info(f"🔋 Generating {num_cycles} cycles for battery {battery_id}")

        cycles_data = []

        # Initial battery state
        current_capacity = self.nominal_capacity
        current_resistance = 0.05  # Initial internal resistance (Ohm)

        for cycle in range(1, num_cycles + 1):
            # Battery degradation over time
            capacity_fade = 1 - (cycle * degradation_rate * random.uniform(0.8, 1.2))
            resistance_growth = 1 + (cycle * degradation_rate * 0.5 * random.uniform(0.9, 1.1))

            # Current cycle capacity and resistance
            cycle_capacity = self.nominal_capacity * capacity_fade
            cycle_resistance = current_resistance * resistance_growth

            # Generate discharge curve
            discharge_data = self._generate_discharge_curve(
                cycle, cycle_capacity, cycle_resistance
            )

            # Generate charge curve
            charge_data = self._generate_charge_curve(
                cycle, cycle_capacity, cycle_resistance
            )

            # Calculate cycle metrics
            cycle_metrics = {
                'battery_id': battery_id,
                'cycle_number': cycle,
                'capacity_ah': cycle_capacity,
                'energy_wh': cycle_capacity * self.nominal_voltage,
                'soh': cycle_capacity / self.nominal_capacity,  # State of Health
                'internal_resistance': cycle_resistance,

                # Discharge metrics
                'discharge_voltage_max': discharge_data['voltage'].max(),
                'discharge_voltage_min': discharge_data['voltage'].min(),
                'discharge_voltage_avg': discharge_data['voltage'].mean(),
                'discharge_current_avg': abs(discharge_data['current'].mean()),
                'discharge_time': discharge_data['time'].max(),
                'discharge_temperature_avg': discharge_data['temperature'].mean(),
                'discharge_temperature_max': discharge_data['temperature'].max(),

                # Charge metrics
                'charge_voltage_max': charge_data['voltage'].max(),
                'charge_current_avg': charge_data['current'].mean(),
                'charge_time': charge_data['time'].max(),
                'charge_temperature_avg': charge_data['temperature'].mean(),

                # Derived features for ML
                'voltage_range': discharge_data['voltage'].max() - discharge_data['voltage'].min(),
                'temperature_rise': discharge_data['temperature'].max() - discharge_data['temperature'].min(),
                'efficiency': (cycle_capacity / self.nominal_capacity) * 100,

                # Time-based features
                'days_elapsed': cycle * random.uniform(0.5, 2.0),  # Irregular cycling
                'ambient_temperature': 25 + random.uniform(-5, 10),  # Environmental variation
            }

            cycles_data.append(cycle_metrics)

        return pd.DataFrame(cycles_data)

    def _generate_discharge_curve(self, cycle: int, capacity: float, resistance: float) -> pd.DataFrame:
        """Generate realistic discharge voltage/current curve"""

        # Discharge current (typically constant current discharge)
        discharge_current = -2.0  # Negative for discharge (A)

        # Time points for discharge
        discharge_time = capacity / abs(discharge_current)  # Hours
        time_points = np.linspace(0, discharge_time * 3600, 100)  # Convert to seconds

        # Generate voltage curve (simplified Li-ion discharge curve)
        soc_points = np.linspace(1.0, 0.1, len(time_points))  # State of charge

        voltages = []
        for soc in soc_points:
            # Simplified voltage model: V = OCV - I*R
            if soc > 0.8:
                ocv = 4.1 - 0.1 * (1 - soc)  # High SOC plateau
            elif soc > 0.2:
                ocv = 3.8 - 0.3 * (0.8 - soc) / 0.6  # Linear region
            else:
                ocv = 3.5 - 1.0 * (0.2 - soc) / 0.2  # Low SOC drop

            # Add aging effects
            ocv -= (cycle * 0.0001)  # Voltage degradation

            # Voltage drop due to resistance
            voltage = ocv - abs(discharge_current) * resistance
            voltages.append(max(voltage, self.min_voltage))

        # Temperature rise during discharge
        base_temp = 25 + random.uniform(-2, 5)
        temp_rise = abs(discharge_current) * resistance * 10  # Heating due to I²R
        temperatures = [base_temp + temp_rise * (t / max(time_points)) for t in time_points]

        return pd.DataFrame({
            'time': time_points,
            'voltage': voltages,
            'current': [discharge_current] * len(time_points),
            'temperature': temperatures
        })

    def _generate_charge_curve(self, cycle: int, capacity: float, resistance: float) -> pd.DataFrame:
        """Generate realistic charge voltage/current curve"""

        # Charge time (typically longer than discharge)
        charge_time_hours = capacity / 1.0 * 1.2  # Slower charge rate
        time_points = np.linspace(0, charge_time_hours * 3600, 80)

        # CC-CV charging profile
        voltages = []
        currents = []

        for i, t in enumerate(time_points):
            progress = i / len(time_points)

            if progress < 0.7:  # Constant Current phase
                current = 1.0  # 1A charge current
                voltage = 3.5 + progress * 0.7 + current * resistance
            else:  # Constant Voltage phase
                voltage = 4.2 - (cycle * 0.0001)  # Slight degradation
                current = 1.0 * (1 - (progress - 0.7) / 0.3)  # Tapering current

            voltages.append(min(voltage, 4.25))
            currents.append(max(current, 0.05))

        # Temperature during charge
        base_temp = 25 + random.uniform(-2, 3)
        temperatures = [base_temp + np.mean(currents) * resistance * 8 + random.uniform(-1, 1)
                       for _ in time_points]

        return pd.DataFrame({
            'time': time_points,
            'voltage': voltages,
            'current': currents,
            'temperature': temperatures
        })

    def generate_multi_battery_dataset(self, num_batteries: int = 4,
                                     cycles_per_battery: int = 500) -> pd.DataFrame:
        """Generate complete multi-battery aging dataset"""

        logger.info(f"🚀 Generating synthetic dataset: {num_batteries} batteries, {cycles_per_battery} cycles each")

        all_battery_data = []

        for i in range(num_batteries):
            battery_id = f"SYNTH_B{i+1:03d}"

            # Vary degradation rates between batteries
            degradation_rate = random.uniform(0.0001, 0.0003)

            # Generate cycles for this battery
            battery_data = self.generate_battery_cycles(
                battery_id, cycles_per_battery, degradation_rate
            )

            all_battery_data.append(battery_data)

        # Combine all battery data
        complete_dataset = pd.concat(all_battery_data, ignore_index=True)

        # Add some realistic noise and variations
        complete_dataset = self._add_realistic_variations(complete_dataset)

        # Save dataset
        dataset_file = self.data_dir / 'synthetic_battery_dataset.csv'
        complete_dataset.to_csv(dataset_file, index=False)

        logger.info(f"✅ Generated dataset with {len(complete_dataset)} data points")
        logger.info(f"📁 Saved to: {dataset_file}")

        return complete_dataset

    def _add_realistic_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic measurement noise and variations"""

        # Add measurement noise
        noise_cols = ['capacity_ah', 'discharge_voltage_avg', 'discharge_current_avg',
                     'discharge_temperature_avg', 'internal_resistance']

        for col in noise_cols:
            if col in df.columns:
                noise_factor = 0.02  # 2% noise
                noise = np.random.normal(0, df[col].std() * noise_factor, len(df))
                df[col] = df[col] + noise

                # Ensure physical constraints
                if 'voltage' in col:
                    df[col] = np.clip(df[col], 2.0, 4.5)
                elif 'capacity' in col:
                    df[col] = np.clip(df[col], 0.1, 3.0)
                elif 'temperature' in col:
                    df[col] = np.clip(df[col], 15, 60)
                elif 'resistance' in col:
                    df[col] = np.clip(df[col], 0.01, 1.0)

        # Add some missing data (realistic)
        missing_rate = 0.001  # 0.1% missing data
        for col in ['discharge_temperature_max', 'charge_temperature_avg']:
            if col in df.columns:
                missing_mask = np.random.random(len(df)) < missing_rate
                df.loc[missing_mask, col] = np.nan

        return df

    def create_dataset_info(self, dataset_df: pd.DataFrame):
        """Create comprehensive dataset information"""

        info_file = self.data_dir / 'SYNTHETIC_DATASET_INFO.md'

        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("# Synthetic Battery Dataset - Ready for Training\n\n")
            f.write("## Dataset Overview\n\n")
            f.write("This synthetic dataset mimics real Li-ion battery aging behavior\n")
            f.write("based on established electrochemical models and experimental observations.\n\n")

            f.write("## Dataset Statistics\n\n")
            f.write(f"- **Total Batteries:** {dataset_df['battery_id'].nunique()}\n")
            f.write(f"- **Total Cycles:** {len(dataset_df)}\n")
            f.write(f"- **Features:** {len(dataset_df.columns)}\n")
            f.write(f"- **SOH Range:** {dataset_df['soh'].min():.3f} - {dataset_df['soh'].max():.3f}\n")
            f.write(f"- **Capacity Range:** {dataset_df['capacity_ah'].min():.3f} - {dataset_df['capacity_ah'].max():.3f} Ah\n\n")

            f.write("## Key Features for ML\n\n")
            features = [
                'capacity_ah - Discharge capacity (Ah)',
                'soh - State of Health (target variable)',
                'internal_resistance - Internal resistance (Ohm)',
                'discharge_voltage_avg - Average discharge voltage (V)',
                'discharge_current_avg - Average discharge current (A)',
                'discharge_temperature_avg - Average temperature (°C)',
                'cycle_number - Cycle count (aging indicator)',
                'efficiency - Capacity retention (%)',
                'voltage_range - Voltage swing during discharge',
                'temperature_rise - Temperature increase during cycling'
            ]

            for feature in features:
                f.write(f"- **{feature}**\n")

            f.write("\n## Battery Performance Summary\n\n")
            for battery_id in dataset_df['battery_id'].unique():
                battery_data = dataset_df[dataset_df['battery_id'] == battery_id]
                f.write(f"### {battery_id}\n")
                f.write(f"- Cycles: {len(battery_data)}\n")
                f.write(f"- Initial SOH: {battery_data['soh'].iloc[0]:.3f}\n")
                f.write(f"- Final SOH: {battery_data['soh'].iloc[-1]:.3f}\n")
                f.write(f"- SOH Degradation: {(1-battery_data['soh'].iloc[-1])*100:.1f}%\n\n")

            f.write("## Model Training Recommendations\n\n")
            f.write("1. **Target Variable:** `soh` (State of Health)\n")
            f.write("2. **Input Features:** All columns except `soh` and `battery_id`\n")
            f.write("3. **Train/Test Split:** Use different batteries for train/test\n")
            f.write("4. **Models to Try:** LSTM, Random Forest, XGBoost, GRU\n")
            f.write("5. **Evaluation Metrics:** RMSE, MAE, R²\n\n")

            f.write("## Files Available\n\n")
            f.write("- `synthetic_battery_dataset.csv` - Main training dataset\n")
            f.write("- This info file\n\n")

            f.write("## Next Steps\n\n")
            f.write("1. Load the CSV file into your training pipeline\n")
            f.write("2. Perform train/test split by battery ID\n")
            f.write("3. Feature engineering (scaling, sequences for LSTM)\n")
            f.write("4. Train models and evaluate performance\n")
            f.write("5. Use trained models in the Streamlit app\n")


def main():
    """Generate synthetic battery dataset"""
    print("🔋 Synthetic Battery Dataset Generator")
    print("=" * 50)

    # Initialize generator
    generator = SyntheticBatteryGenerator()

    # Generate multi-battery dataset
    dataset = generator.generate_multi_battery_dataset(
        num_batteries=6,     # Multiple batteries for robust training
        cycles_per_battery=400  # Sufficient cycles to see degradation
    )

    # Create comprehensive info
    generator.create_dataset_info(dataset)

    print(f"\n✅ Synthetic Dataset Generated!")
    print(f"📁 Location: {generator.data_dir}")
    print(f"📊 Dataset: {len(dataset)} cycles from {dataset['battery_id'].nunique()} batteries")
    print(f"🎯 SOH range: {dataset['soh'].min():.3f} - {dataset['soh'].max():.3f}")
    print(f"🚀 Ready for immediate model training!")

    # Display sample statistics
    print(f"\n📈 Dataset Preview:")
    print(dataset.groupby('battery_id')['soh'].agg(['min', 'max', 'count']).round(3))


if __name__ == "__main__":
    main()
