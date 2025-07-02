from src.data_collection.battery_data_collector import BatteryDataCollector
import logging
import os

def setup_kaggle_api():
    """Set up Kaggle API credentials."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    # Create kaggle.json with API credentials
    kaggle_json = kaggle_dir / "kaggle.json"
    if not kaggle_json.exists():
        logger = logging.getLogger(__name__)
        logger.warning("Kaggle API credentials not found. Please set them up at https://www.kaggle.com/account")
        return False
    return True

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Kaggle battery data collection")
    
    # Initialize data collector
    collector = BatteryDataCollector()
    
    try:
        # Set up Kaggle API
        if not setup_kaggle_api():
            logger.error("Kaggle API setup failed. Please configure your credentials.")
            return
        
        # Download battery datasets
        datasets = [
            "usdot-fmcsa/battery-data-for-vehicle-fleet-management",
            "usdot-fmcsa/battery-data-for-electric-vehicle-fleet-management",
            "usdot-fmcsa/battery-data-for-electric-vehicle-fleet-management-2"
        ]
        
        for dataset in datasets:
            logger.info(f"Downloading dataset: {dataset}")
            collector.collect_kaggle_data(dataset)
            
        logger.info("Kaggle data collection completed successfully")
        
    except Exception as e:
        logger.error(f"Error during Kaggle data collection: {str(e)}")

if __name__ == "__main__":
    main()
