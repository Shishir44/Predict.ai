#!/usr/bin/env python3
"""
Process NASA Battery Datasets
============================

Process all NASA battery .mat files from the comprehensive dataset
and create a unified training dataset for SOH/SOC prediction.
"""

import pandas as pd
import numpy as np
import scipy.io
import logging
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NASADatasetProcessor:
    """Process comprehensive NASA battery dataset"""

    def __init__(self, data_dir: str = "data/raw/5"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get all .mat files
        self.mat_files = list(self.data_dir.glob("*.mat"))
        logger.info(f"Found {len(self.mat_files)} battery dataset files")

    def extract_battery_data(self, mat_file: Path) -> Optional[pd.DataFrame]:
        """Extract data from a single .mat file"""
        try:
            logger.info(f"Processing {mat_file.name}...")

            # Load MATLAB file
            mat_data = scipy.io.loadmat(mat_file, struct_as_record=False, squeeze_me=True)

            # Find the main data structure
            data_key = None
            for key in mat_data.keys():
                if not key.startswith('_') and hasattr(mat_data[key], 'cycle'):
                    data_key = key
                    break

            if data_key is None:
                logger.warning(f"No battery data found in {mat_file.name}")
                return None

            battery_data = mat_data[data_key]

            # Extract battery ID from filename
            battery_id = mat_file.stem  # e.g., "B0025"

            # Process cycles
            cycles_data = []

            if hasattr(battery_data, 'cycle'):
                cycles = battery_data.cycle

                # Handle both single cycle and array of cycles
                if not isinstance(cycles, (list, np.ndarray)):
                    cycles = [cycles]

                for cycle_idx, cycle in enumerate(cycles):
                    if not hasattr(cycle, 'type') or not hasattr(cycle, 'data'):
                        continue

                    cycle_type = str(cycle.type)
                    ambient_temp = getattr(cycle, 'ambient_temperature', 25.0)

                    # Process discharge cycles (most important for SOH)
                    if cycle_type.lower() == 'discharge' and hasattr(cycle.data, 'Voltage_measured'):
                        cycle_info = self._process_discharge_cycle(
                            cycle, battery_id, cycle_idx, ambient_temp
                        )
                        if cycle_info:
                            cycles_data.append(cycle_info)

            if cycles_data:
                df = pd.DataFrame(cycles_data)
                logger.info(f"Extracted {len(df)} discharge cycles from {battery_id}")
                return df
            else:
                logger.warning(f"No valid discharge cycles found in {battery_id}")
                return None

        except Exception as e:
            logger.error(f"Error processing {mat_file.name}: {e}")
            return None

    def _process_discharge_cycle(self, cycle, battery_id: str, cycle_idx: int, ambient_temp: float) -> Optional[Dict]:
        """Process a single discharge cycle"""
        try:
            data = cycle.data

            # Extract measurements
            voltage = np.asarray(data.Voltage_measured, dtype=float)
            current = np.asarray(data.Current_measured, dtype=float)
            temperature = np.asarray(data.Temperature_measured, dtype=float)
            time = np.asarray(data.Time, dtype=float)

            # Get capacity if available
            capacity = getattr(data, 'Capacity', None)
            if capacity is not None:
                capacity = float(capacity)
            else:
                # Estimate capacity from current and time
                if len(current) > 1 and len(time) > 1:
                    time_diff = np.diff(time)
                    current_vals = current[:-1]
                    capacity = np.sum(np.abs(current_vals) * time_diff) / 3600  # Convert to Ah
                else:
                    capacity = 0.0

            # Calculate cycle metrics
            cycle_metrics = {
                'battery_id': battery_id,
                'cycle_number': cycle_idx + 1,
                'ambient_temperature': ambient_temp,

                # Capacity and SOH
                'capacity_ah': capacity,
                'soh': capacity / 2.0 if capacity > 0 else 0.5,  # Assuming 2Ah nominal capacity

                # Voltage metrics
                'voltage_max': float(np.max(voltage)),
                'voltage_min': float(np.min(voltage)),
                'voltage_mean': float(np.mean(voltage)),
                'voltage_std': float(np.std(voltage)),
                'voltage_range': float(np.max(voltage) - np.min(voltage)),

                # Current metrics
                'current_max': float(np.max(np.abs(current))),
                'current_mean': float(np.mean(np.abs(current))),
                'current_std': float(np.std(current)),

                # Temperature metrics
                'temperature_max': float(np.max(temperature)),
                'temperature_min': float(np.min(temperature)),
                'temperature_mean': float(np.mean(temperature)),
                'temperature_std': float(np.std(temperature)),
                'temperature_rise': float(np.max(temperature) - np.min(temperature)),

                # Time and energy metrics
                'discharge_time': float(time[-1] - time[0]) if len(time) > 1 else 0.0,
                'energy_wh': capacity * np.mean(voltage) if capacity > 0 else 0.0,

                # Derived features
                'efficiency': (capacity / 2.0) * 100 if capacity > 0 else 50.0,  # Percentage
                'internal_resistance_est': (np.max(voltage) - np.min(voltage)) / np.mean(np.abs(current)) if np.mean(np.abs(current)) > 0 else 0.1,

                # Data quality metrics
                'data_points': len(voltage),
                'data_duration': float(time[-1] - time[0]) if len(time) > 1 else 0.0
            }

            return cycle_metrics

        except Exception as e:
            logger.error(f"Error processing cycle {cycle_idx} of {battery_id}: {e}")
            return None

    def process_all_batteries(self) -> pd.DataFrame:
        """Process all battery files and create unified dataset"""
        logger.info("Processing all NASA battery datasets...")

        all_data = []

        for mat_file in self.mat_files:
            battery_df = self.extract_battery_data(mat_file)
            if battery_df is not None and len(battery_df) > 0:
                all_data.append(battery_df)

        if not all_data:
            logger.error("No data extracted from any files!")
            return pd.DataFrame()

        # Combine all battery data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Clean and validate data
        combined_df = self._clean_dataset(combined_df)

        # Calculate relative SOH (normalize by first cycle of each battery)
        combined_df = self._calculate_relative_soh(combined_df)

        # Save dataset
        output_file = self.output_dir / 'nasa_comprehensive_dataset.csv'
        combined_df.to_csv(output_file, index=False)

        logger.info(f"Created comprehensive dataset with {len(combined_df)} records")
        logger.info(f"Dataset saved to: {output_file}")

        return combined_df

    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset"""
        logger.info("Cleaning dataset...")

        initial_count = len(df)

        # Remove rows with invalid capacity (negative or too high)
        df = df[df['capacity_ah'] >= 0]
        df = df[df['capacity_ah'] <= 5.0]  # Reasonable upper limit

        # Remove rows with invalid voltage
        df = df[df['voltage_mean'] >= 2.0]
        df = df[df['voltage_mean'] <= 4.5]

        # Remove rows with invalid temperature
        df = df[df['temperature_mean'] >= -10]
        df = df[df['temperature_mean'] <= 80]

        # Remove duplicates
        df = df.drop_duplicates()

        # Fill missing values
        df = df.ffill().bfill()

        logger.info(f"Cleaned dataset: {initial_count} -> {len(df)} records")

        return df

    def _calculate_relative_soh(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate relative SOH based on first cycle of each battery"""
        logger.info("Calculating relative SOH...")

        for battery_id in df['battery_id'].unique():
            battery_mask = df['battery_id'] == battery_id
            battery_data = df[battery_mask].sort_values('cycle_number')

            if len(battery_data) > 0:
                initial_capacity = battery_data['capacity_ah'].iloc[0]
                if initial_capacity > 0:
                    df.loc[battery_mask, 'soh_relative'] = df.loc[battery_mask, 'capacity_ah'] / initial_capacity
                else:
                    df.loc[battery_mask, 'soh_relative'] = 1.0

        return df

    def create_dataset_summary(self, df: pd.DataFrame):
        """Create comprehensive dataset summary"""
        summary_file = self.output_dir / 'NASA_DATASET_SUMMARY.md'

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# NASA Comprehensive Battery Dataset Summary\n\n")

            f.write("## Dataset Overview\n\n")
            f.write(f"- **Total Records:** {len(df)}\n")
            f.write(f"- **Unique Batteries:** {df['battery_id'].nunique()}\n")
            f.write(f"- **Temperature Conditions:** {sorted(df['ambient_temperature'].unique())}\n")
            f.write(f"- **Capacity Range:** {df['capacity_ah'].min():.3f} - {df['capacity_ah'].max():.3f} Ah\n")
            f.write(f"- **SOH Range:** {df['soh_relative'].min():.3f} - {df['soh_relative'].max():.3f}\n\n")

            f.write("## Battery Statistics\n\n")
            battery_stats = df.groupby('battery_id').agg({
                'cycle_number': ['count', 'max'],
                'capacity_ah': ['min', 'max'],
                'soh_relative': ['min', 'max'],
                'ambient_temperature': 'mean'
            }).round(3)

            f.write("| Battery | Cycles | Max Cycle | Min Capacity | Max Capacity | Min SOH | Max SOH | Temp |\n")
            f.write("|---------|--------|-----------|--------------|--------------|---------|---------|------|\n")

            for battery_id in sorted(df['battery_id'].unique()):
                battery_data = df[df['battery_id'] == battery_id]
                cycles = len(battery_data)
                max_cycle = battery_data['cycle_number'].max()
                min_cap = battery_data['capacity_ah'].min()
                max_cap = battery_data['capacity_ah'].max()
                min_soh = battery_data['soh_relative'].min()
                max_soh = battery_data['soh_relative'].max()
                temp = battery_data['ambient_temperature'].mean()

                f.write(f"| {battery_id} | {cycles} | {max_cycle} | {min_cap:.3f} | {max_cap:.3f} | {min_soh:.3f} | {max_soh:.3f} | {temp:.1f}°C |\n")

            f.write("\n## Features Available\n\n")
            features = [col for col in df.columns if col not in ['battery_id']]
            for feature in features:
                f.write(f"- **{feature}**: {df[feature].dtype}\n")

            f.write("\n## Recommended Usage\n\n")
            f.write("1. **Target Variable**: `soh_relative` (Relative State of Health)\n")
            f.write("2. **Train/Test Split**: Split by battery_id to ensure proper evaluation\n")
            f.write("3. **Features**: All columns except battery_id and soh_relative\n")
            f.write("4. **Models**: LSTM, Random Forest, XGBoost, GRU\n")
            f.write("5. **Cross-validation**: Group by battery_id\n\n")

            f.write("## Temperature Conditions\n\n")
            temp_groups = df.groupby('ambient_temperature')['battery_id'].nunique()
            for temp, count in temp_groups.items():
                f.write(f"- **{temp}°C**: {count} batteries\n")


def main():
    """Process all NASA battery datasets"""
    print("🔋 NASA Comprehensive Battery Dataset Processor")
    print("=" * 55)

    # Initialize processor
    processor = NASADatasetProcessor()

    # Process all datasets
    df = processor.process_all_batteries()

    if len(df) > 0:
        # Create summary
        processor.create_dataset_summary(df)

        print(f"\n✅ Dataset Processing Complete!")
        print(f"📊 Total Records: {len(df)}")
        print(f"🔋 Unique Batteries: {df['battery_id'].nunique()}")
        print(f"📁 Output: data/processed/nasa_comprehensive_dataset.csv")
        print(f"🌡️ Temperature Conditions: {sorted(df['ambient_temperature'].unique())}")
        print(f"🎯 SOH Range: {df['soh_relative'].min():.3f} - {df['soh_relative'].max():.3f}")
        print(f"🚀 Ready for advanced model training!")

        # Display sample
        print(f"\n📋 Sample Data:")
        print(df[['battery_id', 'cycle_number', 'capacity_ah', 'soh_relative', 'ambient_temperature']].head(10))

    else:
        print("❌ No data could be processed!")


if __name__ == "__main__":
    main()
 