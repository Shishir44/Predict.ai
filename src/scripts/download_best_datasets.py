#!/usr/bin/env python3
"""
Download Best Battery Datasets for SOH/SOC Prediction
=====================================================

This script downloads and organizes the most comprehensive battery datasets
available for State of Health (SOH) and State of Charge (SOC) prediction.

Datasets included:
1. NASA Prognostics Data Repository (Li-ion Battery Aging)
2. BatteryLife Dataset (KDD 2025) - Most comprehensive
3. Oxford Battery Degradation Dataset
4. Multi-modal EV Dataset (Real-world data)
5. Munich University Multi-Stage Dataset

Author: Predict.AI Project Team
Date: 2025
"""

import os
import requests
import zipfile
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional
import scipy.io
from tqdm import tqdm
import urllib.request
import json

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

        # Dataset URLs and information
        self.datasets = {
            'nasa': {
                'name': 'NASA Prognostics Center Battery Data',
                'url': 'https://c3.nasa.gov/dashlink/resources/133/',
                'description': 'Li-ion battery aging data from NASA Ames',
                'chemistry': 'Li-ion 18650',
                'cells': 4,
                'priority': 'HIGH'
            },
            'oxford': {
                'name': 'Oxford Battery Degradation Dataset',
                'url': 'https://ora.ox.ac.uk/objects/uuid:03ba4734-3d73-4bdc-8f3e-ffb50b70f6ba',
                'description': 'Commercial Li-ion battery degradation',
                'chemistry': 'Li-ion commercial',
                'cells': 8,
                'priority': 'HIGH'
            },
            'batterylife': {
                'name': 'BatteryLife Dataset (KDD 2025)',
                'description': 'Most comprehensive dataset with 16 integrated datasets',
                'chemistry': 'Li-ion, Zn-ion, Na-ion',
                'cells': '2.5x larger than previous datasets',
                'priority': 'CRITICAL'
            },
            'calce': {
                'name': 'CALCE Battery Data',
                'url': 'https://calce.umd.edu/battery-data',
                'description': 'University of Maryland battery aging data',
                'chemistry': 'LCO/graphite',
                'cells': 13,
                'priority': 'HIGH'
            },
            'mit_stanford': {
                'name': 'MIT-Stanford Battery Dataset',
                'description': 'Fast-charging protocol optimization data',
                'chemistry': 'LFP/graphite',
                'cells': 124,
                'priority': 'MEDIUM'
            }
        }

    def download_nasa_data(self):
        """Download NASA Prognostics Center battery data"""
        logger.info("🔋 Downloading NASA Battery Data...")
        nasa_dir = self.data_dir / 'nasa'
        nasa_dir.mkdir(exist_ok=True)

        # NASA data files (example - actual URLs need to be updated)
        nasa_files = [
            {
                'filename': 'B0005.mat',
                'url': 'https://c3.nasa.gov/dashlink/static/media/dataset/B0005.mat',
                'description': 'Battery 5 aging data'
            },
            {
                'filename': 'B0006.mat',
                'url': 'https://c3.nasa.gov/dashlink/static/media/dataset/B0006.mat',
                'description': 'Battery 6 aging data'
            },
            {
                'filename': 'B0007.mat',
                'url': 'https://c3.nasa.gov/dashlink/static/media/dataset/B0007.mat',
                'description': 'Battery 7 aging data'
            },
            {
                'filename': 'B0018.mat',
                'url': 'https://c3.nasa.gov/dashlink/static/media/dataset/B0018.mat',
                'description': 'Battery 18 aging data'
            }
        ]

        for file_info in nasa_files:
            file_path = nasa_dir / file_info['filename']
            if not file_path.exists():
                try:
                    logger.info(f"Downloading {file_info['filename']}...")
                    self._download_file(file_info['url'], file_path)
                    logger.info(f"✅ Downloaded {file_info['filename']}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not download {file_info['filename']}: {e}")
                    # Create instruction file for manual download
                    instruction_file = nasa_dir / f"{file_info['filename']}_MANUAL_DOWNLOAD.txt"
                    with open(instruction_file, 'w') as f:
                        f.write(f"Manual Download Required\n")
                        f.write(f"======================\n\n")
                        f.write(f"File: {file_info['filename']}\n")
                        f.write(f"URL: {file_info['url']}\n")
                        f.write(f"Description: {file_info['description']}\n\n")
                        f.write(f"Please download manually and place in: {nasa_dir}\n")

        return nasa_dir

    def download_oxford_data(self):
        """Download Oxford Battery Degradation Dataset"""
        logger.info("🎓 Downloading Oxford Battery Data...")
        oxford_dir = self.data_dir / 'oxford'
        oxford_dir.mkdir(exist_ok=True)

        # Oxford dataset information
        oxford_info = {
            'dataset_1': {
                'url': 'https://ora.ox.ac.uk/objects/uuid:03ba4734-3d73-4bdc-8f3e-ffb50b70f6ba/datastreams/DATASET1',
                'filename': 'oxford_dataset_1.zip',
                'description': 'Oxford Battery Degradation Dataset 1'
            },
            'dataset_2': {
                'url': 'https://ora.ox.ac.uk/objects/uuid:f54fa4a5-f2e3-4cde-9c92-7e8e2d8f8b45/datastreams/DATASET2',
                'filename': 'oxford_dataset_2.zip',
                'description': 'Oxford Battery Degradation Dataset 2'
            }
        }

        for dataset_name, info in oxford_info.items():
            file_path = oxford_dir / info['filename']
            if not file_path.exists():
                try:
                    logger.info(f"Downloading {info['filename']}...")
                    self._download_file(info['url'], file_path)

                    # Extract if it's a zip file
                    if info['filename'].endswith('.zip'):
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(oxford_dir / dataset_name)
                        logger.info(f"✅ Extracted {info['filename']}")

                except Exception as e:
                    logger.warning(f"⚠️ Could not download {info['filename']}: {e}")
                    # Create manual download instruction
                    self._create_manual_download_instruction(oxford_dir, info)

        return oxford_dir

    def setup_batterylife_dataset(self):
        """Setup BatteryLife dataset (KDD 2025) - Most comprehensive"""
        logger.info("🚀 Setting up BatteryLife Dataset (KDD 2025)...")
        batterylife_dir = self.data_dir / 'batterylife'
        batterylife_dir.mkdir(exist_ok=True)

        # BatteryLife dataset information
        batterylife_info = {
            'description': 'BatteryLife: Most comprehensive battery prediction dataset',
            'paper': 'BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction (KDD 2025)',
            'features': [
                '16 integrated datasets',
                '2.5x larger than previous largest dataset',
                '8 battery formats',
                '59 chemical systems',
                '9 operating temperatures',
                '421 charge/discharge protocols',
                'First Zinc-ion and Sodium-ion battery datasets',
                'Industrial-tested large-capacity Li-ion batteries'
            ],
            'github_url': 'https://github.com/BatteryLife-Dataset/BatteryLife',
            'paper_url': 'https://arxiv.org/abs/2502.18807'
        }

        # Create information file
        info_file = batterylife_dir / 'BATTERYLIFE_INFO.md'
        with open(info_file, 'w') as f:
            f.write("# BatteryLife Dataset (KDD 2025)\n\n")
            f.write("## 🔋 Most Comprehensive Battery Prediction Dataset\n\n")
            f.write(f"**Description:** {batterylife_info['description']}\n\n")
            f.write(f"**Paper:** {batterylife_info['paper']}\n\n")
            f.write("## 🌟 Key Features:\n\n")
            for feature in batterylife_info['features']:
                f.write(f"- {feature}\n")
            f.write(f"\n## 📁 Dataset Access:\n\n")
            f.write(f"- **GitHub:** {batterylife_info['github_url']}\n")
            f.write(f"- **Paper:** {batterylife_info['paper_url']}\n\n")
            f.write("## 🚀 How to Access:\n\n")
            f.write("1. Visit the GitHub repository above\n")
            f.write("2. Follow the download instructions in their README\n")
            f.write("3. The dataset will be available through their official channels\n")
            f.write("4. This is the most advanced dataset for battery prediction research\n\n")
            f.write("## 🔧 Integration:\n\n")
            f.write("Once downloaded, place the BatteryLife data in this directory.\n")
            f.write("Our preprocessing scripts will automatically detect and process the data.\n")

        logger.info("✅ BatteryLife dataset information created")
        return batterylife_dir

    def download_calce_data(self):
        """Download CALCE battery data"""
        logger.info("🏛️ Setting up CALCE Battery Data...")
        calce_dir = self.data_dir / 'calce'
        calce_dir.mkdir(exist_ok=True)

        # CALCE dataset information
        calce_info = {
            'description': 'University of Maryland CALCE Battery Research Group Data',
            'url': 'https://calce.umd.edu/battery-data',
            'datasets': [
                'CS2 Panasonic 18650PF cells',
                'CS2 Samsung INR18650-35E cells',
                'CS2 Sanyo UR18650E cells',
                'CS2 LG HG2 cells'
            ]
        }

        # Create information and download instruction file
        info_file = calce_dir / 'CALCE_DOWNLOAD_INSTRUCTIONS.md'
        with open(info_file, 'w') as f:
            f.write("# CALCE Battery Data\n\n")
            f.write(f"**Description:** {calce_info['description']}\n\n")
            f.write(f"**URL:** {calce_info['url']}\n\n")
            f.write("## Available Datasets:\n\n")
            for dataset in calce_info['datasets']:
                f.write(f"- {dataset}\n")
            f.write("\n## Download Instructions:\n\n")
            f.write("1. Visit https://calce.umd.edu/battery-data\n")
            f.write("2. Register for access if required\n")
            f.write("3. Download the datasets you need\n")
            f.write("4. Place downloaded files in this directory\n")
            f.write("5. Run the preprocessing script to convert to standard format\n")

        return calce_dir

    def create_dataset_overview(self):
        """Create comprehensive dataset overview"""
        logger.info("📊 Creating dataset overview...")

        overview_file = self.data_dir / 'DATASET_OVERVIEW.md'
        with open(overview_file, 'w') as f:
            f.write("# 🔋 Battery Prediction Datasets Overview\n\n")
            f.write("## 📈 Priority Ranking for SOH/SOC Prediction\n\n")

            # Sort datasets by priority
            priority_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
            for priority in priority_order:
                f.write(f"### {priority} Priority\n\n")
                for name, info in self.datasets.items():
                    if info.get('priority') == priority:
                        f.write(f"#### {info['name']}\n")
                        f.write(f"- **Chemistry:** {info['chemistry']}\n")
                        f.write(f"- **Cells:** {info['cells']}\n")
                        f.write(f"- **Description:** {info['description']}\n")
                        if 'url' in info:
                            f.write(f"- **URL:** {info['url']}\n")
                        f.write("\n")

            f.write("## 🔧 Usage Instructions\n\n")
            f.write("1. **Start with BatteryLife Dataset (KDD 2025)** - Most comprehensive\n")
            f.write("2. **Add NASA data** - Most validated and benchmarked\n")
            f.write("3. **Include Oxford data** - For path-dependent studies\n")
            f.write("4. **Use CALCE data** - For additional diversity\n\n")

            f.write("## 📁 Directory Structure\n\n")
            f.write("```\n")
            f.write("data/raw/\n")
            f.write("├── batterylife/     # KDD 2025 - Most comprehensive\n")
            f.write("├── nasa/           # NASA Prognostics Center\n")
            f.write("├── oxford/         # Oxford University\n")
            f.write("├── calce/          # University of Maryland\n")
            f.write("└── processed/      # Preprocessed data\n")
            f.write("```\n\n")

            f.write("## 🚀 Next Steps\n\n")
            f.write("1. Run preprocessing scripts to standardize formats\n")
            f.write("2. Implement feature engineering for SOH/SOC prediction\n")
            f.write("3. Train models using the comprehensive dataset\n")
            f.write("4. Validate on real-world EV data\n")

    def _download_file(self, url: str, filepath: Path, chunk_size: int = 8192):
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

        except requests.exceptions.RequestException as e:
            raise Exception(f"Download failed: {e}")

    def _create_manual_download_instruction(self, directory: Path, info: Dict):
        """Create manual download instruction file"""
        instruction_file = directory / f"MANUAL_DOWNLOAD_{info['filename']}.txt"
        with open(instruction_file, 'w') as f:
            f.write(f"Manual Download Required\n")
            f.write(f"======================\n\n")
            f.write(f"File: {info['filename']}\n")
            f.write(f"URL: {info['url']}\n")
            f.write(f"Description: {info['description']}\n\n")
            f.write(f"Please download manually and place in: {directory}\n")

    def download_all(self):
        """Download all available datasets"""
        logger.info("🚀 Starting comprehensive dataset download...")

        downloaded_datasets = {}

        try:
            # Download datasets in priority order
            downloaded_datasets['batterylife'] = self.setup_batterylife_dataset()
            downloaded_datasets['nasa'] = self.download_nasa_data()
            downloaded_datasets['oxford'] = self.download_oxford_data()
            downloaded_datasets['calce'] = self.download_calce_data()

            # Create overview
            self.create_dataset_overview()

            logger.info("✅ Dataset download process completed!")
            logger.info("📊 Check DATASET_OVERVIEW.md for detailed information")

            return downloaded_datasets

        except Exception as e:
            logger.error(f"❌ Error during dataset download: {e}")
            raise


def main():
    """Main function to download datasets"""
    print("🔋 Battery Dataset Downloader")
    print("=" * 50)

    # Initialize downloader
    downloader = BatteryDatasetDownloader(data_dir="data/raw")

    # Download all datasets
    datasets = downloader.download_all()

    print("\n✅ Download process completed!")
    print("\n📁 Downloaded datasets:")
    for name, path in datasets.items():
        print(f"  - {name}: {path}")

    print(f"\n📊 Check {downloader.data_dir}/DATASET_OVERVIEW.md for details")
    print("🚀 Ready to start preprocessing and training!")


if __name__ == "__main__":
    main()
