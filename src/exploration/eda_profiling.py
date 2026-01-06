import pandas as pd
from ydata_profiling import ProfileReport
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import load_config, setup_logger

def run_profiling():
    config = load_config()
    logger = setup_logger("PROFILING", os.path.join(config['outputs']['logs'], "profiling.log"))
    
    train_path = config['data']['train_path']
    output_report = os.path.join(config['outputs']['visualizations'], "eda_report.html")
    
    logger.info(f"Loading data from {train_path}...")
    df = pd.read_csv(train_path)
    
    logger.info("Generating Profile Report (this may take a few minutes)...")
    # For large datasets, minimal=True can be used if it takes too long
    profile = ProfileReport(df, title="Student Test Scores - EDA Report", explorative=True)
    
    logger.info(f"Saving report to {output_report}...")
    profile.to_file(output_report)
    
    logger.info("Profiling completed.")

if __name__ == "__main__":
    run_profiling()
