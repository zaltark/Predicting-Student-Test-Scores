import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.exploration.explore import explore_data
from src.processing.preprocess import preprocess_data
from src.features.engineer_features import engineer_features
from src.training.train_model import train_model
from src.utils import load_config, setup_logger

def run_pipeline():
    config = load_config()
    logger = setup_logger("PIPELINE", os.path.join(config['outputs']['logs'], "pipeline.log"))
    
    logger.info("=========================================")
    logger.info("   STARTING END-TO-END PIPELINE")
    logger.info("=========================================")
    
    try:
        explore_data()
        preprocess_data()
        engineer_features()
        train_model()
        logger.info("=========================================")
        logger.info("   PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=========================================")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise e

if __name__ == "__main__":
    run_pipeline()
