import os
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import zipfile
import shutil
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class BatteryDataCollector:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_nasa_battery_data(self) -> Dict[str, pd.DataFrame]:
        """Collect battery data from NASA's battery dataset."""
        base_url = "https://data.nasa.gov/api/views/92fc-b976/rows.csv"
        datasets = {
            "B0005": f"{base_url}?accessType=DOWNLOAD",
            "B0006": f"{base_url}?accessType=DOWNLOAD",
            "B0007": f"{base_url}?accessType=DOWNLOAD",
            "B0018": f"{base_url}?accessType=DOWNLOAD"
        }
        
        battery_data = {}
        
        for battery_id, url in datasets.items():
            try:
                logger.info(f"Downloading NASA battery data for {battery_id}")
                response = requests.get(url)
                df = pd.read_csv(pd.compat.StringIO(response.text))
                df.to_csv(self.data_dir / f"nasa_{battery_id}.csv", index=False)
                battery_data[battery_id] = df
                logger.info(f"Successfully saved {battery_id} data")
            except Exception as e:
                logger.error(f"Error downloading {battery_id}: {str(e)}")
                
        return battery_data

    def collect_kaggle_data(self, dataset_name: str, api_token: str = None) -> None:
        """Collect battery data from Kaggle."""
        try:
            import kaggle
            
            if api_token:
                kaggle.api.authenticate()
            
            logger.info(f"Downloading Kaggle dataset: {dataset_name}")
            kaggle.api.dataset_download_files(
                dataset_name,
                path=str(self.data_dir / "kaggle"),
                unzip=True
            )
            logger.info("Kaggle dataset downloaded successfully")
            
        except ImportError:
            logger.error("Kaggle API not installed. Install it using: pip install kaggle")
        except Exception as e:
            logger.error(f"Error downloading Kaggle dataset: {str(e)}")

    def collect_synthetic_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic battery data for testing."""
        np.random.seed(42)
        
        # Generate synthetic features
        cycle = np.arange(1, num_samples + 1)
        capacity = np.random.normal(100, 5, num_samples).clip(0, 100)
        voltage = np.random.normal(3.7, 0.1, num_samples).clip(3.0, 4.2)
        current = np.random.normal(1.0, 0.1, num_samples)
        temperature = np.random.normal(25, 5, num_samples)
        
        # Generate SOH and SOC values
        soh = (capacity / 100) * 100
        soc = np.random.normal(80, 10, num_samples).clip(0, 100)
        
        # Create DataFrame
        df = pd.DataFrame({
            'cycle': cycle,
            'capacity': capacity,
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'SOH': soh,
            'SOC': soc
        })
        
        df.to_csv(self.data_dir / "synthetic_battery_data.csv", index=False)
        logger.info("Synthetic battery data generated and saved")
        return df

    def clean_nasa_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess NASA battery data."""
        # Handle missing values
        df = df.fillna(method='ffill')
        
        # Convert data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # Normalize features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
            
        return df

    def combine_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple battery datasets into a single DataFrame."""
        combined = pd.concat(list(datasets.values()), keys=list(datasets.keys()))
        combined.reset_index(level=0, inplace=True)
        combined.rename(columns={'level_0': 'battery_id'}, inplace=True)
        return combined

    def save_dataset(self, df: pd.DataFrame, filename: str) -> None:
        """Save processed dataset to disk."""
        df.to_csv(self.data_dir / filename, index=False)
        logger.info(f"Dataset saved to {filename}")

    def load_dataset(self, filename: str) -> pd.DataFrame:
        """Load a saved dataset from disk."""
        return pd.read_csv(self.data_dir / filename)
