from src.data_collection.battery_data_collector import BatteryDataCollector
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    # Initialize data collector
    collector = BatteryDataCollector()
    
    # Download NASA battery data
    logger = logging.getLogger(__name__)
    logger.info("Starting NASA battery data collection")
    
    try:
        # Collect NASA data
        nasa_data = collector.collect_nasa_battery_data()
        
        # Clean and combine datasets
        cleaned_data = {}
        for battery_id, df in nasa_data.items():
            cleaned_df = collector.clean_nasa_data(df)
            cleaned_data[battery_id] = cleaned_df
            
        combined_df = collector.combine_datasets(cleaned_data)
        
        # Save processed data
        collector.save_dataset(combined_df, "processed_nasa_battery_data.csv")
        logger.info("Data collection and processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error during data collection: {str(e)}")
