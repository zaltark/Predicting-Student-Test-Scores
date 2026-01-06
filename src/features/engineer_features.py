import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils import load_config, setup_logger

def engineer_features():
    config = load_config()
    logger = setup_logger("FEATURE_ENGINEERING", os.path.join(config['outputs']['logs'], "feature_engineering.log"))
    
    logger.info("Starting Feature Engineering...")
    
    train_df = pd.read_csv(config['outputs']['train_cleaned'])
    test_df = pd.read_csv(config['outputs']['test_cleaned'])
    target_col = config['pipeline']['target']
    
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    
    logger.info(f"Applying Target Encoding to categorical columns: {list(categorical_cols)}")
    
    # Target Encoding with smoothing to prevent leakage
    for col in categorical_cols:
        # Calculate global mean
        global_mean = train_df[target_col].mean()
        
        # Calculate mean target for each category
        agg = train_df.groupby(col)[target_col].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']
        
        # Smoothing (adjust 10 based on category frequency)
        smoothing = 10
        smooth_means = (counts * means + smoothing * global_mean) / (counts + smoothing)
        
        # Map to train and test
        train_df[f"{col}_te"] = train_df[col].map(smooth_means).fillna(global_mean)
        test_df[f"{col}_te"] = test_df[col].map(smooth_means).fillna(global_mean)
        
        # Keep original label encoded version as well? 
        # For now, let's keep the target encoded version and drop original object
        train_df.drop(columns=[col], inplace=True)
        test_df.drop(columns=[col], inplace=True)

    logger.info(f"Saving processed data to {config['outputs']['processed_train']} and {config['outputs']['processed_test']}")
    train_df.to_csv(config['outputs']['processed_train'], index=False)
    test_df.to_csv(config['outputs']['processed_test'], index=False)
    
    logger.info("Feature Engineering completed.")

if __name__ == "__main__":
    engineer_features()
