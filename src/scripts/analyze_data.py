import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatteryDataAnalyzer:
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.df = None
        
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
        
        self.df = pd.concat(dfs, ignore_index=True)
        return self.df
        
    def get_basic_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate basic statistics for battery measurements."""
        if self.df is None:
            self.load_data()
            
        stats = {}
        numeric_cols = ['capacity', 'voltage', 'current', 'temperature', 'SOH', 'SOC']
        
        for battery_id in self.df['battery_id'].unique():
            battery_df = self.df[self.df['battery_id'] == battery_id]
            battery_stats = {
                'mean': battery_df[numeric_cols].mean().to_dict(),
                'std': battery_df[numeric_cols].std().to_dict(),
                'min': battery_df[numeric_cols].min().to_dict(),
                'max': battery_df[numeric_cols].max().to_dict(),
                'median': battery_df[numeric_cols].median().to_dict()
            }
            stats[battery_id] = battery_stats
            
        return stats
        
    def calculate_degradation_rate(self) -> Dict[str, float]:
        """Calculate capacity degradation rate for each battery."""
        if self.df is None:
            self.load_data()
            
        degradation_rates = {}
        
        for battery_id in self.df['battery_id'].unique():
            battery_df = self.df[self.df['battery_id'] == battery_id]
            battery_df = battery_df.sort_values('cycle')
            
            # Calculate degradation rate as slope of capacity vs cycle
            x = battery_df['cycle'].values
            y = battery_df['capacity'].values
            
            if len(x) < 2:
                degradation_rates[battery_id] = 0
                continue
                
            slope, _ = np.polyfit(x, y, 1)
            degradation_rates[battery_id] = slope
            
        return degradation_rates
        
    def detect_anomalies(self, threshold: float = 3.0) -> Dict[str, pd.DataFrame]:
        """Detect anomalies in battery measurements."""
        if self.df is None:
            self.load_data()
            
        anomalies = {}
        numeric_cols = ['capacity', 'voltage', 'current', 'temperature', 'SOH', 'SOC']
        
        for battery_id in self.df['battery_id'].unique():
            battery_df = self.df[self.df['battery_id'] == battery_id]
            battery_anomalies = pd.DataFrame()
            
            for col in numeric_cols:
                # Calculate z-scores
                z_scores = (battery_df[col] - battery_df[col].mean()) / battery_df[col].std()
                
                # Find anomalies
                anomalous = battery_df[abs(z_scores) > threshold]
                if not anomalous.empty:
                    anomalous['feature'] = col
                    anomalous['z_score'] = z_scores[abs(z_scores) > threshold]
                    battery_anomalies = pd.concat([battery_anomalies, anomalous])
            
            if not battery_anomalies.empty:
                anomalies[battery_id] = battery_anomalies
                
        return anomalies
        
    def calculate_feature_correlations(self) -> Dict[str, pd.DataFrame]:
        """Calculate correlations between battery features."""
        if self.df is None:
            self.load_data()
            
        correlations = {}
        numeric_cols = ['capacity', 'voltage', 'current', 'temperature', 'SOH', 'SOC']
        
        for battery_id in self.df['battery_id'].unique():
            battery_df = self.df[self.df['battery_id'] == battery_id]
            corr_matrix = battery_df[numeric_cols].corr()
            correlations[battery_id] = corr_matrix
            
        return correlations
        
    def save_analysis_results(self, output_dir: str = "data/analysis") -> None:
        """Save analysis results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save basic statistics
        stats = self.get_basic_statistics()
        pd.DataFrame(stats).to_csv(output_dir / 'basic_statistics.csv')
        
        # Save degradation rates
        degradation_rates = self.calculate_degradation_rate()
        pd.DataFrame.from_dict(degradation_rates, orient='index').to_csv(output_dir / 'degradation_rates.csv')
        
        # Save anomalies
        anomalies = self.detect_anomalies()
        for battery_id, anomaly_df in anomalies.items():
            anomaly_df.to_csv(output_dir / f'anomalies_{battery_id}.csv')
        
        # Save correlations
        correlations = self.calculate_feature_correlations()
        for battery_id, corr_df in correlations.items():
            corr_df.to_csv(output_dir / f'correlations_{battery_id}.csv')

def main():
    """Main function to analyze battery data."""
    logger.info("Starting data analysis")
    
    try:
        analyzer = BatteryDataAnalyzer()
        
        # Load data
        df = analyzer.load_data()
        if df is None:
            logger.error("Failed to load data")
            return
            
        # Perform analysis
        logger.info("Calculating basic statistics")
        stats = analyzer.get_basic_statistics()
        
        logger.info("Calculating degradation rates")
        degradation_rates = analyzer.calculate_degradation_rate()
        
        logger.info("Detecting anomalies")
        anomalies = analyzer.detect_anomalies()
        
        logger.info("Calculating feature correlations")
        correlations = analyzer.calculate_feature_correlations()
        
        # Save results
        logger.info("Saving analysis results")
        analyzer.save_analysis_results()
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error during data analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
