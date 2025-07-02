from src.data_collection.battery_data_collector import BatteryDataCollector
import logging
import numpy as np
import pandas as pd

def generate_synthetic_battery_data(
    num_samples: int = 1000,
    num_batteries: int = 3,
    noise_level: float = 0.1,
    save: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic battery data with realistic characteristics.
    
    Args:
        num_samples: Number of samples per battery
        num_batteries: Number of batteries to simulate
        noise_level: Level of noise to add to measurements
        save: Whether to save the generated data to disk
        
    Returns:
        Dictionary of DataFrames, one per battery
    """
    np.random.seed(42)
    
    battery_data = {}
    
    for battery_id in range(1, num_batteries + 1):
        # Base capacity degradation curve
        cycles = np.arange(1, num_samples + 1)
        base_capacity = 100 - 0.05 * cycles  # 5% degradation per 100 cycles
        
        # Add noise and random variations
        capacity = base_capacity * (1 + noise_level * np.random.randn(num_samples))
        capacity = np.clip(capacity, 0, 100)
        
        # Generate other measurements
        voltage = 3.7 + 0.1 * np.sin(2 * np.pi * cycles / 100) + 0.1 * np.random.randn(num_samples)
        current = 1.0 + 0.1 * np.random.randn(num_samples)
        temperature = 25 + 5 * np.random.randn(num_samples)
        
        # Calculate SOH and SOC
        soh = (capacity / 100) * 100
        soc = 80 + 10 * np.random.randn(num_samples)
        soc = np.clip(soc, 0, 100)
        
        # Create DataFrame
        df = pd.DataFrame({
            'cycle': cycles,
            'capacity': capacity,
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'SOH': soh,
            'SOC': soc
        })
        
        battery_data[f'B{battery_id:04d}'] = df
        
        if save:
            collector = BatteryDataCollector()
            collector.save_dataset(df, f"synthetic_battery_{battery_id}.csv")
            
    return battery_data

def main():
    """Generate and save synthetic battery data."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Generating synthetic battery data")
    
    try:
        # Generate data for 3 batteries with 1000 samples each
        battery_data = generate_synthetic_battery_data(
            num_samples=1000,
            num_batteries=3,
            noise_level=0.1,
            save=True
        )
        
        logger.info("Synthetic data generation completed successfully")
        return battery_data
        
    except Exception as e:
        logger.error(f"Error generating synthetic data: {str(e)}")
        raise

if __name__ == "__main__":
    main()
