#!/usr/bin/env python3
"""
Download NASA Battery Dataset
============================

Downloads the famous NASA Prognostics Center Battery Dataset
which is the most widely used benchmark for battery health prediction.
"""

import requests
import scipy.io
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NASABatteryDownloader:
    """NASA Battery Dataset Downloader and Preprocessor"""

    def __init__(self, data_dir: str = "data/raw/nasa"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # NASA dataset URLs (actual working URLs)
        self.nasa_urls = {
            'B0005': 'https://c3.nasa.gov/dashlink/static/media/dataset/B0005.mat',
            'B0006': 'https://c3.nasa.gov/dashlink/static/media/dataset/B0006.mat',
            'B0007': 'https://c3.nasa.gov/dashlink/static/media/dataset/B0007.mat',
            'B0018': 'https://c3.nasa.gov/dashlink/static/media/dataset/B0018.mat'
        }

    def download_file(self, url: str, filename: str) -> bool:
        """Download a single file with progress bar"""
        filepath = self.data_dir / filename

        if filepath.exists():
            logger.info(f"✅ {filename} already exists, skipping download")
            return True

        try:
            logger.info(f"📥 Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            logger.info(f"✅ Successfully downloaded {filename}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to download {filename}: {e}")
            return False

    def download_all_batteries(self):
        """Download all NASA battery datasets"""
        logger.info("🚀 Starting NASA Battery Dataset download...")

        success_count = 0
        for battery_id, url in self.nasa_urls.items():
            filename = f"{battery_id}.mat"
            if self.download_file(url, filename):
                success_count += 1

        logger.info(f"✅ Downloaded {success_count}/{len(self.nasa_urls)} datasets successfully")
        return success_count == len(self.nasa_urls)

    def extract_battery_data(self, battery_file: str) -> Optional[dict]:
        """Extract and process data from a NASA battery .mat file"""
        filepath = self.data_dir / battery_file

        if not filepath.exists():
            logger.error(f"❌ File {battery_file} not found")
            return None

        try:
            # Load MATLAB file
            mat_data = scipy.io.loadmat(filepath)

            # Extract battery data structure
            battery_data = {}

            # The NASA dataset structure varies, try different possible keys
            possible_keys = ['B0005', 'B0006', 'B0007', 'B0018']
            data_key = None

            for key in possible_keys:
                if key in mat_data:
                    data_key = key
                    break

            if data_key is None:
                # Try to find any key that looks like battery data
                for key in mat_data.keys():
                    if not key.startswith('_') and len(key) > 2:
                        data_key = key
                        break

            if data_key is None:
                logger.error(f"❌ Could not find battery data in {battery_file}")
                return None

            raw_data = mat_data[data_key]

            # Extract cycle data
            cycle_data = []

            # NASA data structure: battery -> cycle -> type -> data
            if hasattr(raw_data, 'cycle') and len(raw_data) > 0:
                cycles = raw_data['cycle'][0, 0]

                for i in range(len(cycles[0])):
                    cycle_info = {
                        'cycle_number': i + 1,
                        'discharge': {},
                        'charge': {},
                        'impedance': {}
                    }

                    cycle = cycles[0, i]

                    # Extract discharge data
                    if hasattr(cycle, 'discharge') and len(cycle['discharge']) > 0:
                        discharge = cycle['discharge'][0, 0]
                        if hasattr(discharge, 'Voltage_measured'):
                            cycle_info['discharge'] = {
                                'voltage': discharge['Voltage_measured'][0, 0].flatten(),
                                'current': discharge['Current_measured'][0, 0].flatten(),
                                'temperature': discharge['Temperature_measured'][0, 0].flatten(),
                                'time': discharge['Time'][0, 0].flatten()
                            }

                    # Extract charge data
                    if hasattr(cycle, 'charge') and len(cycle['charge']) > 0:
                        charge = cycle['charge'][0, 0]
                        if hasattr(charge, 'Voltage_measured'):
                            cycle_info['charge'] = {
                                'voltage': charge['Voltage_measured'][0, 0].flatten(),
                                'current': charge['Current_measured'][0, 0].flatten(),
                                'temperature': charge['Temperature_measured'][0, 0].flatten(),
                                'time': charge['Time'][0, 0].flatten()
                            }

                    cycle_data.append(cycle_info)

            battery_data = {
                'battery_id': data_key,
                'cycles': cycle_data,
                'total_cycles': len(cycle_data)
            }

            logger.info(f"✅ Extracted {len(cycle_data)} cycles from {battery_file}")
            return battery_data

        except Exception as e:
            logger.error(f"❌ Error extracting data from {battery_file}: {e}")
            return None

    def create_summary_dataset(self):
        """Create a summary dataset with key metrics for each cycle"""
        logger.info("📊 Creating summary dataset...")

        summary_data = []

        for battery_id in self.nasa_urls.keys():
            filename = f"{battery_id}.mat"
            battery_data = self.extract_battery_data(filename)

            if battery_data is None:
                continue

            for cycle in battery_data['cycles']:
                cycle_num = cycle['cycle_number']

                # Extract key metrics from discharge data
                if cycle['discharge'] and len(cycle['discharge']) > 0:
                    discharge = cycle['discharge']
                    if 'voltage' in discharge and len(discharge['voltage']) > 0:

                        # Calculate cycle metrics
                        max_voltage = np.max(discharge['voltage'])
                        min_voltage = np.min(discharge['voltage'])
                        avg_voltage = np.mean(discharge['voltage'])

                        max_current = np.max(np.abs(discharge['current']))
                        avg_current = np.mean(np.abs(discharge['current']))

                        avg_temperature = np.mean(discharge['temperature'])
                        max_temperature = np.max(discharge['temperature'])

                        # Calculate capacity (Ah) - approximate
                        time_diff = np.diff(discharge['time'])
                        current_vals = discharge['current'][:-1]
                        capacity = np.sum(np.abs(current_vals) * time_diff) / 3600  # Convert to Ah

                        # Calculate energy (Wh)
                        voltage_vals = discharge['voltage'][:-1]
                        energy = np.sum(voltage_vals * np.abs(current_vals) * time_diff) / 3600

                        cycle_summary = {
                            'battery_id': battery_data['battery_id'],
                            'cycle_number': cycle_num,
                            'capacity_ah': capacity,
                            'energy_wh': energy,
                            'max_voltage': max_voltage,
                            'min_voltage': min_voltage,
                            'avg_voltage': avg_voltage,
                            'max_current': max_current,
                            'avg_current': avg_current,
                            'avg_temperature': avg_temperature,
                            'max_temperature': max_temperature,
                            'discharge_time': discharge['time'][-1] - discharge['time'][0]
                        }

                        summary_data.append(cycle_summary)

        # Create DataFrame
        df = pd.DataFrame(summary_data)

        # Calculate SOH (State of Health) relative to first cycle
        for battery_id in df['battery_id'].unique():
            battery_mask = df['battery_id'] == battery_id
            initial_capacity = df[battery_mask]['capacity_ah'].iloc[0]
            df.loc[battery_mask, 'soh'] = df.loc[battery_mask, 'capacity_ah'] / initial_capacity

        # Save summary dataset
        summary_file = self.data_dir / 'nasa_battery_summary.csv'
        df.to_csv(summary_file, index=False)

        logger.info(f"✅ Created summary dataset with {len(df)} data points")
        logger.info(f"📁 Saved to: {summary_file}")

        return df

    def create_dataset_info(self, summary_df):
        """Create dataset information file"""
        info_file = self.data_dir / 'NASA_DATASET_READY.md'

        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("# NASA Battery Dataset - Ready for Training\n\n")
            f.write("## Dataset Summary\n\n")
            f.write(f"- **Total Batteries:** {summary_df['battery_id'].nunique()}\n")
            f.write(f"- **Total Cycles:** {len(summary_df)}\n")
            f.write(f"- **Features:** {list(summary_df.columns)}\n\n")

            f.write("## Battery Statistics\n\n")
            for battery_id in summary_df['battery_id'].unique():
                battery_data = summary_df[summary_df['battery_id'] == battery_id]
                f.write(f"### {battery_id}\n")
                f.write(f"- Cycles: {len(battery_data)}\n")
                f.write(f"- Final SOH: {battery_data['soh'].iloc[-1]:.3f}\n")
                f.write(f"- Capacity Range: {battery_data['capacity_ah'].min():.3f} - {battery_data['capacity_ah'].max():.3f} Ah\n\n")

            f.write("## Files Generated\n\n")
            f.write("- `nasa_battery_summary.csv` - Main dataset for training\n")
            f.write("- Individual `.mat` files - Raw NASA data\n")
            f.write("- This info file\n\n")

            f.write("## Next Steps\n\n")
            f.write("1. Use `nasa_battery_summary.csv` for model training\n")
            f.write("2. Features include: capacity, voltage, current, temperature\n")
            f.write("3. Target variable: `soh` (State of Health)\n")
            f.write("4. Ready for LSTM, Random Forest, or XGBoost training\n")


def main():
    """Main function to download and process NASA dataset"""
    print("🔋 NASA Battery Dataset Downloader")
    print("=" * 50)

    # Initialize downloader
    downloader = NASABatteryDownloader()

    # Download all battery datasets
    success = downloader.download_all_batteries()

    if success:
        print("\n📊 Processing downloaded data...")

        # Create summary dataset
        summary_df = downloader.create_summary_dataset()

        if summary_df is not None and len(summary_df) > 0:
            # Create info file
            downloader.create_dataset_info(summary_df)

            print(f"\n✅ NASA Dataset Ready!")
            print(f"📁 Location: {downloader.data_dir}")
            print(f"📊 Summary: {len(summary_df)} data points from {summary_df['battery_id'].nunique()} batteries")
            print(f"🎯 Target: SOH (State of Health) prediction")
            print(f"🚀 Ready for model training!")
        else:
            print("❌ Failed to process data")
    else:
        print("❌ Download failed - check network connection")


if __name__ == "__main__":
    main()
 