import os
import requests
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatteryDataDownloader:
    def __init__(self, 
                data_dir: str = "data/raw",
                processed_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets = {
            'nasa': {
                'url': 'https://ti.arc.nasa.gov/c/6/',
                'description': 'NASA Battery Data Set',
                'files': ['B0005.mat', 'B0006.mat', 'B0007.mat', 'B0018.mat']
            },
            'kaggle': {
                'url': 'https://www.kaggle.com/datasets/parisrohan/battery-discharge-data',
                'description': 'Kaggle Battery Discharge Data',
                'files': ['battery_discharge_data.csv']
            }
        }
        
    def download_nasa_data(self) -> None:
        """Download NASA battery data."""
        nasa_dir = self.data_dir / 'nasa'
        nasa_dir.mkdir(exist_ok=True)
        
        for file in self.datasets['nasa']['files']:
            url = f"{self.datasets['nasa']['url']}{file}"
            file_path = nasa_dir / file
            
            if file_path.exists():
                logger.info(f"{file} already exists, skipping download")
                continue
                
            try:
                logger.info(f"Downloading {file}...")
                response = requests.get(url)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                    
                logger.info(f"Successfully downloaded {file}")
                
            except requests.RequestException as e:
                logger.error(f"Error downloading {file}: {str(e)}")
                
    def download_kaggle_data(self) -> None:
        """Download Kaggle battery data."""
        kaggle_dir = self.data_dir / 'kaggle'
        kaggle_dir.mkdir(exist_ok=True)
        
        for file in self.datasets['kaggle']['files']:
            file_path = kaggle_dir / file
            
            if file_path.exists():
                logger.info(f"{file} already exists, skipping download")
                continue
                
            try:
                logger.info(f"Downloading {file}...")
                # Note: Kaggle data requires authentication
                # This is a placeholder for actual Kaggle API usage
                # You'll need to set up Kaggle API credentials
                # and use kaggle.api.dataset_download_files()
                logger.warning(f"Kaggle download requires authentication. Please download manually from: {self.datasets['kaggle']['url']}")
                
            except Exception as e:
                logger.error(f"Error downloading {file}: {str(e)}")
                
    def process_nasa_data(self) -> None:
        """Process NASA battery data."""
        import scipy.io
        
        nasa_dir = self.data_dir / 'nasa'
        processed_path = self.processed_dir / 'nasa_processed.csv'
        
        all_data = []
        
        for file in self.datasets['nasa']['files']:
            file_path = nasa_dir / file
            
            if not file_path.exists():
                logger.warning(f"{file} not found, skipping processing")
                continue
                
            try:
                logger.info(f"Processing {file}...")
                data = scipy.io.loadmat(file_path)
                
                # Extract relevant features
                df = pd.DataFrame({
                    'cycle': data['cycle'][:, 0],
                    'capacity': data['capacity'][:, 0],
                    'voltage': data['voltage'][:, 0],
                    'current': data['current'][:, 0],
                    'temperature': data['temperature'][:, 0],
                    'SOH': data['SOH'][:, 0],
                    'SOC': data['SOC'][:, 0]
                })
                
                df['battery_id'] = file.split('.')[0]
                df['timestamp'] = pd.to_datetime('now')  # Add timestamp
                
                all_data.append(df)
                
            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")
                
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df.to_csv(processed_path, index=False)
            logger.info(f"Processed NASA data saved to {processed_path}")
            
    def process_kaggle_data(self) -> None:
        """Process Kaggle battery data."""
        kaggle_dir = self.data_dir / 'kaggle'
        processed_path = self.processed_dir / 'kaggle_processed.csv'
        
        if not (kaggle_dir / self.datasets['kaggle']['files'][0]).exists():
            logger.warning("Kaggle data not found. Please download manually from Kaggle")
            return
            
        try:
            logger.info("Processing Kaggle data...")
            df = pd.read_csv(kaggle_dir / self.datasets['kaggle']['files'][0])
            
            # Clean and transform data
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['battery_id'] = df['battery_id'].astype(str)
            
            # Calculate SOH and SOC
            df['SOH'] = df.groupby('battery_id')['capacity'].apply(
                lambda x: x / x.iloc[0]
            )
            
            df['SOC'] = df['voltage'] / df['voltage'].max()
            
            df.to_csv(processed_path, index=False)
            logger.info(f"Processed Kaggle data saved to {processed_path}")
            
        except Exception as e:
            logger.error(f"Error processing Kaggle data: {str(e)}")
            
    def generate_synthetic_data(self, num_samples: int = 10000) -> None:
        """Generate synthetic battery data."""
        np.random.seed(42)
        
        # Generate synthetic features
        timestamps = pd.date_range(start='2020-01-01', periods=num_samples, freq='H')
        battery_ids = np.random.choice(['B0001', 'B0002', 'B0003', 'B0004'], size=num_samples)
        
        # Generate realistic battery data
        capacity = np.random.uniform(1.0, 2.0, size=num_samples)
        voltage = np.random.uniform(3.0, 4.5, size=num_samples)
        current = np.random.uniform(-5.0, 5.0, size=num_samples)
        temperature = np.random.uniform(20.0, 40.0, size=num_samples)
        
        # Calculate SOH and SOC
        soh = np.random.uniform(0.7, 1.0, size=num_samples)
        soc = np.random.uniform(0.0, 1.0, size=num_samples)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'battery_id': battery_ids,
            'capacity': capacity,
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'SOH': soh,
            'SOC': soc
        })
        
        # Save synthetic data
        synthetic_path = self.processed_dir / 'synthetic_data.csv'
        df.to_csv(synthetic_path, index=False)
        logger.info(f"Generated synthetic data saved to {synthetic_path}")
        
    def combine_data(self) -> None:
        """Combine all processed data into a single file."""
        processed_files = list(self.processed_dir.glob('*.csv'))
        
        if not processed_files:
            logger.warning("No processed data files found")
            return
            
        try:
            logger.info("Combining processed data...")
            dfs = []
            
            for file in processed_files:
                df = pd.read_csv(file)
                df['source'] = file.stem.split('_')[0]
                dfs.append(df)
                
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_path = self.processed_dir / 'battery_data.csv'
            combined_df.to_csv(combined_path, index=False)
            logger.info(f"Combined data saved to {combined_path}")
            
        except Exception as e:
            logger.error(f"Error combining data: {str(e)}")
            
    def download_and_process(self) -> None:
        """Download and process all battery data."""
        try:
            # Download data
            logger.info("Downloading NASA data...")
            self.download_nasa_data()
            
            logger.info("Downloading Kaggle data...")
            self.download_kaggle_data()
            
            # Process data
            logger.info("Processing NASA data...")
            self.process_nasa_data()
            
            logger.info("Processing Kaggle data...")
            self.process_kaggle_data()
            
            # Generate synthetic data
            logger.info("Generating synthetic data...")
            self.generate_synthetic_data()
            
            # Combine all data
            logger.info("Combining all data sources...")
            self.combine_data()
            
        except Exception as e:
            logger.error(f"Error in data download and processing: {str(e)}")
            raise

def main():
    """Main function to download and process battery data."""
    downloader = BatteryDataDownloader()
    downloader.download_and_process()

if __name__ == "__main__":
    main()
