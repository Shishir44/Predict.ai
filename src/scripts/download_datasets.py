#!/usr/bin/env python3
"""
Download Best Battery Datasets for SOH/SOC Prediction
=====================================================

This script downloads and organizes the most comprehensive battery datasets
available for State of Health (SOH) and State of Charge (SOC) prediction.
"""

import os
import requests
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatteryDatasetDownloader:
    """
    Comprehensive battery dataset downloader and organizer
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def setup_batterylife_dataset(self):
        """Setup BatteryLife dataset (KDD 2025) - Most comprehensive"""
        logger.info("🚀 Setting up BatteryLife Dataset (KDD 2025)...")
        batterylife_dir = self.data_dir / 'batterylife'
        batterylife_dir.mkdir(exist_ok=True)

        # Create information file
        info_file = batterylife_dir / 'BATTERYLIFE_INFO.md'
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("# BatteryLife Dataset (KDD 2025)\n\n")
            f.write("## Most Comprehensive Battery Prediction Dataset\n\n")
            f.write("**Description:** BatteryLife: Most comprehensive battery prediction dataset\n\n")
            f.write("**Paper:** BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction (KDD 2025)\n\n")
            f.write("## Key Features:\n\n")
            features = [
                '16 integrated datasets',
                '2.5x larger than previous largest dataset',
                '8 battery formats',
                '59 chemical systems',
                '9 operating temperatures',
                '421 charge/discharge protocols',
                'First Zinc-ion and Sodium-ion battery datasets',
                'Industrial-tested large-capacity Li-ion batteries'
            ]
            for feature in features:
                f.write(f"- {feature}\n")
            f.write(f"\n## Dataset Access:\n\n")
            f.write(f"- **GitHub:** https://github.com/BatteryLife-Dataset/BatteryLife\n")
            f.write(f"- **Paper:** https://arxiv.org/abs/2502.18807\n\n")

        logger.info("✅ BatteryLife dataset information created")
        return batterylife_dir

    def download_nasa_data(self):
        """Download NASA Prognostics Center battery data"""
        logger.info("Setting up NASA Battery Data...")
        nasa_dir = self.data_dir / 'nasa'
        nasa_dir.mkdir(exist_ok=True)

        # Create NASA dataset info
        info_file = nasa_dir / 'NASA_DATASET_INFO.md'
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("# NASA Prognostics Center Battery Data\n\n")
            f.write("## NASA Ames Research Center\n\n")
            f.write("**Description:** Li-ion battery aging data from NASA Ames\n\n")
            f.write("**URL:** https://c3.nasa.gov/dashlink/resources/133/\n\n")
            f.write("## Available Data:\n\n")
            f.write("- Battery B0005 (Room temperature)\n")
            f.write("- Battery B0006 (Room temperature)\n")
            f.write("- Battery B0007 (Room temperature)\n")
            f.write("- Battery B0018 (Room temperature)\n\n")
            f.write("## How to Download:\n\n")
            f.write("1. Visit https://c3.nasa.gov/dashlink/resources/133/\n")
            f.write("2. Download the battery data files (.mat format)\n")
            f.write("3. Place them in this directory\n")

        return nasa_dir

    def create_dataset_overview(self):
        """Create comprehensive dataset overview"""
        logger.info("Creating dataset overview...")

        overview_file = self.data_dir / 'DATASET_OVERVIEW.md'
        with open(overview_file, 'w', encoding='utf-8') as f:
            f.write("# Battery Prediction Datasets Overview\n\n")
            f.write("## Recommended Datasets for SOH/SOC Prediction\n\n")

            f.write("### CRITICAL Priority\n\n")
            f.write("#### BatteryLife Dataset (KDD 2025)\n")
            f.write("- **Chemistry:** Li-ion, Zn-ion, Na-ion\n")
            f.write("- **Cells:** 2.5x larger than previous datasets\n")
            f.write("- **Description:** Most comprehensive dataset with 16 integrated datasets\n\n")

            f.write("### HIGH Priority\n\n")
            f.write("#### NASA Prognostics Center Battery Data\n")
            f.write("- **Chemistry:** Li-ion 18650\n")
            f.write("- **Cells:** 4\n")
            f.write("- **Description:** Li-ion battery aging data from NASA Ames\n")
            f.write("- **URL:** https://c3.nasa.gov/dashlink/resources/133/\n\n")

            f.write("#### Oxford Battery Degradation Dataset\n")
            f.write("- **Chemistry:** Li-ion commercial\n")
            f.write("- **Cells:** 8\n")
            f.write("- **Description:** Commercial Li-ion battery degradation\n")
            f.write("- **URL:** https://ora.ox.ac.uk/objects/uuid:03ba4734-3d73-4bdc-8f3e-ffb50b70f6ba\n\n")

            f.write("## Usage Instructions\n\n")
            f.write("1. **Start with BatteryLife Dataset (KDD 2025)** - Most comprehensive\n")
            f.write("2. **Add NASA data** - Most validated and benchmarked\n")
            f.write("3. **Include Oxford data** - For path-dependent studies\n\n")

            f.write("## Directory Structure\n\n")
            f.write("```\n")
            f.write("data/raw/\n")
            f.write("├── batterylife/     # KDD 2025 - Most comprehensive\n")
            f.write("├── nasa/           # NASA Prognostics Center\n")
            f.write("├── oxford/         # Oxford University\n")
            f.write("└── processed/      # Preprocessed data\n")
            f.write("```\n")

    def download_all(self):
        """Download all available datasets"""
        logger.info("🚀 Starting comprehensive dataset download...")

        downloaded_datasets = {}

        try:
            # Setup datasets in priority order
            downloaded_datasets['batterylife'] = self.setup_batterylife_dataset()
            downloaded_datasets['nasa'] = self.download_nasa_data()

            # Create overview
            self.create_dataset_overview()

            logger.info("✅ Dataset setup process completed!")
            logger.info("📊 Check DATASET_OVERVIEW.md for detailed information")

            return downloaded_datasets

        except Exception as e:
            logger.error(f"❌ Error during dataset setup: {e}")
            raise


def main():
    """Main function to download datasets"""
    print("🔋 Battery Dataset Downloader")
    print("=" * 50)

    # Initialize downloader
    downloader = BatteryDatasetDownloader(data_dir="data/raw")

    # Download all datasets
    datasets = downloader.download_all()

    print("\n✅ Setup process completed!")
    print("\n📁 Prepared datasets:")
    for name, path in datasets.items():
        print(f"  - {name}: {path}")

    print(f"\n📊 Check {downloader.data_dir}/DATASET_OVERVIEW.md for details")
    print("🚀 Ready to start downloading and training!")


if __name__ == "__main__":
    main()
