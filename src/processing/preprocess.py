import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils import load_config, setup_logger

def preprocess_data():
    config = load_config()
    logger = setup_logger("PREPROCESSING", os.path.join(config['outputs']['logs'], "preprocessing.log"))
    
    logger.info("Starting Data Preprocessing...")
    
    train_df = pd.read_csv(config['data']['train_path'])
    test_df = pd.read_csv(config['data']['test_path'])
    
    logger.info("Keeping study_hours, class_attendance, and sleep_hours as raw numerical values.")

    # Drop ID column as it has no predictive power
    logger.info("Dropping 'id' column from train and test sets")
    train_df.drop(columns=['id'], inplace=True)
    test_df.drop(columns=['id'], inplace=True)
    
    # Save the cleaned/preprocessed data to artifacts
    train_output = config['outputs']['train_cleaned']
    test_output = config['outputs']['test_cleaned']
    
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    
    logger.info(f"Cleaned data saved to {train_output} and {test_output}")
    logger.info("Data Preprocessing completed.")

if __name__ == "__main__":
    preprocess_data()
