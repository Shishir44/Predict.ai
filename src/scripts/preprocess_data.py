import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatteryDataPreprocessor:
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.scalers = {}
        self.sequence_length = 50  # Number of previous time steps to use
        
    def load_data(self) -> pd.DataFrame:
        """Load processed battery data."""
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            logger.error("No CSV files found in data directory")
            return None
            
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)
        
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences of data for time series prediction."""
        features = df.drop(['battery_id', 'SOH', 'SOC'], axis=1)
        soh = df['SOH'].values
        soc = df['SOC'].values
        
        X = []
        y_soh = []
        y_soc = []
        
        for i in range(len(features) - self.sequence_length):
            X.append(features.iloc[i:i + self.sequence_length].values)
            y_soh.append(soh[i + self.sequence_length])
            y_soc.append(soc[i + self.sequence_length])
        
        return (
            np.array(X),
            np.array(y_soh),
            np.array(y_soc)
        )
        
    def normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features using StandardScaler."""
        if 'features' not in self.scalers:
            self.scalers['features'] = StandardScaler()
            self.scalers['features'].fit(X.reshape(-1, X.shape[-1]))
        
        X_normalized = self.scalers['features'].transform(X.reshape(-1, X.shape[-1]))
        return X_normalized.reshape(X.shape)
        
    def normalize_labels(self, y_soh: np.ndarray, y_soc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize labels using MinMaxScaler."""
        if 'soh' not in self.scalers:
            self.scalers['soh'] = MinMaxScaler(feature_range=(0, 1))
            self.scalers['soh'].fit(y_soh.reshape(-1, 1))
            
        if 'soc' not in self.scalers:
            self.scalers['soc'] = MinMaxScaler(feature_range=(0, 1))
            self.scalers['soc'].fit(y_soc.reshape(-1, 1))
        
        y_soh_normalized = self.scalers['soh'].transform(y_soh.reshape(-1, 1)).flatten()
        y_soc_normalized = self.scalers['soc'].transform(y_soc.reshape(-1, 1)).flatten()
        
        return y_soh_normalized, y_soc_normalized
        
    def save_scalers(self, output_dir: str = "data/scalers") -> None:
        """Save scalers for later use."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, output_dir / f'{name}_scaler.pkl')
        
    def load_scalers(self, input_dir: str = "data/scalers") -> None:
        """Load previously saved scalers."""
        input_dir = Path(input_dir)
        
        for scaler_file in input_dir.glob("*.pkl"):
            name = scaler_file.stem.replace('_scaler', '')
            self.scalers[name] = joblib.load(scaler_file)
        
    def split_data(self, X: np.ndarray, y_soh: np.ndarray, y_soc: np.ndarray, 
                  test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        """Split data into train, validation, and test sets."""
        # First split into train and test
        X_train, X_test, y_soh_train, y_soh_test, y_soc_train, y_soc_test = train_test_split(
            X, y_soh, y_soc, test_size=test_size, random_state=42
        )
        
        # Split train into train and validation
        X_train, X_val, y_soh_train, y_soh_val, y_soc_train, y_soc_val = train_test_split(
            X_train, y_soh_train, y_soc_train, test_size=val_size/(1-test_size), random_state=42
        )
        
        return (
            X_train, X_val, X_test,
            y_soh_train, y_soh_val, y_soh_test,
            y_soc_train, y_soc_val, y_soc_test
        )
        
    def prepare_data_for_training(self, test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        """Prepare data for model training."""
        logger.info("Starting data preprocessing")
        
        try:
            # Load data
            df = self.load_data()
            if df is None:
                logger.error("Failed to load data")
                return None
                
            # Create sequences
            logger.info("Creating sequences")
            X, y_soh, y_soc = self.create_sequences(df)
            
            # Normalize features
            logger.info("Normalizing features")
            X_normalized = self.normalize_features(X)
            
            # Normalize labels
            logger.info("Normalizing labels")
            y_soh_normalized, y_soc_normalized = self.normalize_labels(y_soh, y_soc)
            
            # Split data
            logger.info("Splitting data into train/validation/test sets")
            X_train, X_val, X_test, y_soh_train, y_soh_val, y_soh_test, \
            y_soc_train, y_soc_val, y_soc_test = self.split_data(
                X_normalized, y_soh_normalized, y_soc_normalized,
                test_size=test_size, val_size=val_size
            )
            
            # Save scalers
            logger.info("Saving scalers")
            self.save_scalers()
            
            logger.info("Data preprocessing completed successfully")
            return (
                (X_train, X_val, X_test),
                (y_soh_train, y_soh_val, y_soh_test),
                (y_soc_train, y_soc_val, y_soc_test)
            )
            
        except Exception as e:
            logger.error(f"Error during data preprocessing: {str(e)}")
            raise

def main():
    """Main function to preprocess battery data."""
    preprocessor = BatteryDataPreprocessor()
    preprocessor.prepare_data_for_training()

if __name__ == "__main__":
    main()
