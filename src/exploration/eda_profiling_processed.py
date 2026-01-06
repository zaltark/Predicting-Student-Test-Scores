import pandas as pd
from ydata_profiling import ProfileReport
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import load_config, setup_logger

def run_processed_profiling():
    config = load_config()
    logger = setup_logger("PROFILING_PROCESSED", os.path.join(config['outputs']['logs'], "profiling_processed.log"))
    
    processed_train_path = config['outputs']['processed_train']
    output_report = os.path.join(config['outputs']['visualizations'], "eda_report_processed.html")
    
    logger.info(f"Loading processed data from {processed_train_path}...")
    df = pd.read_csv(processed_train_path)
    
    logger.info("Generating Profile Report for Processed Data (this may take a few minutes)...")
    profile = ProfileReport(df, title="Student Test Scores - Processed EDA Report", explorative=True)
    
    logger.info(f"Saving report to {output_report}...")
    profile.to_file(output_report)
    
    logger.info("Profiling of processed data completed.")

if __name__ == "__main__":
    run_processed_profiling()
