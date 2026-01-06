import sys
import os

# Add project root to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils import load_config, setup_logger

def explore_data():
    config = load_config()
    logger = setup_logger("EXPLORATION", os.path.join(config['outputs']['logs'], "exploration.log"))
    
    logger.info("Starting Data Exploration...")
    logger.info(f"Loading train data from {config['data']['train_path']}")
    logger.info(f"Loading test data from {config['data']['test_path']}")
    
    # Placeholder for exploration logic (plotting, stats, etc.)
    logger.info("Generating visualizations (Placeholder)...")
    logger.info(f"Saving plots to {config['outputs']['visualizations']}")
    
    logger.info("Data Exploration completed.")

if __name__ == "__main__":
    explore_data()
