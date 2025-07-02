import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_dir: str = "data/processed") -> pd.DataFrame:
    """Load processed battery data."""
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        logger.error("No CSV files found in data directory")
        return None
    
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def plot_capacity_degradation(df: pd.DataFrame, save_path: str = None) -> None:
    """Plot battery capacity degradation over cycles."""
    plt.figure(figsize=(12, 6))
    
    for battery_id in df['battery_id'].unique():
        battery_df = df[df['battery_id'] == battery_id]
        plt.plot(battery_df['cycle'], battery_df['capacity'], label=f'Battery {battery_id}')
    
    plt.title('Battery Capacity Degradation Over Cycles')
    plt.xlabel('Cycle Number')
    plt.ylabel('Capacity (%)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path / 'capacity_degradation.png')
    plt.show()

def plot_voltage_current(df: pd.DataFrame, save_path: str = None) -> None:
    """Plot voltage vs current relationship."""
    plt.figure(figsize=(12, 6))
    
    for battery_id in df['battery_id'].unique():
        battery_df = df[df['battery_id'] == battery_id]
        plt.scatter(battery_df['current'], battery_df['voltage'], 
                   label=f'Battery {battery_id}', alpha=0.5)
    
    plt.title('Voltage vs Current Relationship')
    plt.xlabel('Current (A)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path / 'voltage_current.png')
    plt.show()

def plot_temperature_distribution(df: pd.DataFrame, save_path: str = None) -> None:
    """Plot temperature distribution."""
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=df, x='temperature', hue='battery_id', fill=True, alpha=0.3)
    plt.title('Temperature Distribution')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Density')
    
    if save_path:
        plt.savefig(save_path / 'temperature_distribution.png')
    plt.show()

def plot_soh_soc_relationship(df: pd.DataFrame, save_path: str = None) -> None:
    """Plot SOH vs SOC relationship."""
    plt.figure(figsize=(12, 6))
    
    for battery_id in df['battery_id'].unique():
        battery_df = df[df['battery_id'] == battery_id]
        plt.scatter(battery_df['SOH'], battery_df['SOC'], 
                   label=f'Battery {battery_id}', alpha=0.5)
    
    plt.title('SOH vs SOC Relationship')
    plt.xlabel('State of Health (%)')
    plt.ylabel('State of Charge (%)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path / 'soh_soc_relationship.png')
    plt.show()

def main():
    """Main function to visualize battery data."""
    logger.info("Starting data visualization")
    
    try:
        # Load data
        df = load_data()
        if df is None:
            logger.error("Failed to load data")
            return
        
        # Create visualization directory
        viz_dir = Path("data/visualization")
        viz_dir.mkdir(exist_ok=True)
        
        # Generate visualizations
        logger.info("Generating capacity degradation plot")
        plot_capacity_degradation(df, viz_dir)
        
        logger.info("Generating voltage-current relationship plot")
        plot_voltage_current(df, viz_dir)
        
        logger.info("Generating temperature distribution plot")
        plot_temperature_distribution(df, viz_dir)
        
        logger.info("Generating SOH-SOC relationship plot")
        plot_soh_soc_relationship(df, viz_dir)
        
        logger.info("Visualization completed successfully")
        
    except Exception as e:
        logger.error(f"Error during data visualization: {str(e)}")
        raise

if __name__ == "__main__":
    main()
