import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils import load_config, setup_logger

def train_model():
    config = load_config()
    logger = setup_logger("TRAINING", os.path.join(config['outputs']['logs'], "training.log"))
    
    logger.info("Starting Model Training...")
    
    # Load data
    train_path = config['outputs']['processed_train']
    test_path = config['outputs']['processed_test']
    target_col = config['pipeline']['target']
    
    logger.info(f"Loading processed data from {train_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Feature Selection: Best known setup
    selected_features = [
        'sleep_hours', 
        'facility_rating_te', 
        'study_method_te',    
        'sleep_quality_te',   
        'class_attendance', 
        'study_hours'
    ]
    
    X = train_df[selected_features]
    y = train_df[target_col]
    X_test = test_df[selected_features]
    
    # K-Fold Cross Validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config['pipeline']['random_state'])
    
    rmse_scores = []
    
    logger.info(f"Starting {n_splits}-Fold Cross-Validation...")
    
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            random_state=config['pipeline']['random_state'],
            n_jobs=-1,
            early_stopping_rounds=50
        )
        
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )
        
        preds = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        rmse_scores.append(rmse)
        
        logger.info(f"Fold {fold+1} RMSE: {rmse:.4f}")
        
    avg_rmse = np.mean(rmse_scores)
    logger.info(f"Average RMSE: {avg_rmse:.4f}")
    
    # Train on full dataset
    logger.info("Retraining model on full dataset...")
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=config['pipeline']['random_state'],
        n_jobs=-1
    )
    final_model.fit(X, y, verbose=False)
    
    # Save Model
    model_path = config['outputs']['model_path']
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    logger.info(f"Model saved to {model_path}")
    
    # Generate Submission
    logger.info("Generating predictions for test set...")
    test_preds = final_model.predict(X_test)
    
    submission_path = config['outputs']['submission_path']
    
    # We need the IDs for submission. Since we dropped them, we need to read them from original test file
    # Or cleaner: read the sample submission and fill it
    sample_sub_path = config['data']['sample_submission_path']
    if os.path.exists(sample_sub_path):
        submission = pd.read_csv(sample_sub_path)
        submission[target_col] = test_preds
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission file saved to {submission_path}")
    else:
        logger.warning(f"Sample submission not found at {sample_sub_path}. Submission file not generated correctly (missing IDs).")

    logger.info("Model Training completed.")

if __name__ == "__main__":
    train_model()
