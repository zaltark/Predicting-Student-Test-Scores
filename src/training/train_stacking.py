import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils import load_config, setup_logger

def train_stacking():
    config = load_config()
    logger = setup_logger("STACKING", os.path.join(config['outputs']['logs'], "stacking.log"))
    
    logger.info("Starting Stacking Ensemble...")
    
    # Load data
    train_path = config['outputs']['processed_train']
    test_path = config['outputs']['processed_test']
    target_col = config['pipeline']['target']
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Selected Features (Best Set)
    selected_features = [
        'sleep_hours', 
        'facility_rating_te', 
        'study_method_te',    
        'sleep_quality_te',   
        'class_attendance', 
        'study_hours'
    ]
    
    # Handle potentially missing columns in test (e.g. if TE failed)
    # Ideally should be consistent, but let's be safe
    X = train_df[selected_features]
    y = train_df[target_col]
    X_test = test_df[selected_features]
    
    # Define Base Models
    # Using tuned hyperparameters for XGB and LGBM
    xgb_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 2000, # Increased for tuned LR
        'learning_rate': 0.053939076596510226,
        'max_depth': 5,
        'subsample': 0.9562354389688843,
        'colsample_bytree': 0.9265224535978751,
        'reg_alpha': 2.5341184255853644e-07,
        'reg_lambda': 8.695770046722775e-07,
        'min_child_weight': 10,
        'random_state': config['pipeline']['random_state'],
        'n_jobs': -1,
        'early_stopping_rounds': 100
    }

    lgbm_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 2000,
        'learning_rate': 0.0674879984975894,
        'num_leaves': 42,
        'max_depth': 5,
        'subsample': 0.5017400909129055,
        'colsample_bytree': 0.5057717312440626,
        'reg_alpha': 1.8742450132661552e-06,
        'reg_lambda': 1.2472403484004457e-07,
        'min_child_samples': 17,
        'random_state': config['pipeline']['random_state'],
        'n_jobs': -1,
        'verbosity': -1
    }

    base_models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RF": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=config['pipeline']['random_state'], n_jobs=-1),
        "XGB": xgb.XGBRegressor(**xgb_params),
        "LGBM": lgb.LGBMRegressor(**lgbm_params),
        "Cat": CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_seed=config['pipeline']['random_state'], verbose=False, allow_writing_files=False)
    }
    
    # Arrays to hold Level 1 predictions
    train_meta_features = pd.DataFrame(index=X.index)
    test_meta_features = pd.DataFrame(index=X_test.index)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=config['pipeline']['random_state'])
    
    logger.info("Training Base Models (Level 1)...")
    
    for name, model in base_models.items():
        logger.info(f"Training {name}...")
        oof_preds = np.zeros(len(X))
        test_preds_fold = [] # Store predictions from each fold model on test set
        
        # Cross-validation for OOF
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale for linear models
            if name in ["Linear", "Ridge"]:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                model.fit(X_train, y_train)
            
            elif name == "XGB":
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            
            elif name == "LGBM":
                callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='rmse',
                    callbacks=callbacks
                )
                
            else:
                model.fit(X_train, y_train)
            
            # Predict (handle linear scaling for val)
            # wait, X_val is already scaled for linear above
            if name in ["Linear", "Ridge"]:
                 oof_preds[val_idx] = model.predict(X_val)
            else:
                 oof_preds[val_idx] = model.predict(X_val)
            
            # Predict on Test Set (need to scale if linear)
            if name in ["Linear", "Ridge"]:
                # Re-fit scaler on current fold train to transform test
                # Already fit above
                X_test_scaled = scaler.transform(X_test)
                test_preds_fold.append(model.predict(X_test_scaled))
            else:
                test_preds_fold.append(model.predict(X_test))
        
        # Store OOF predictions
        train_meta_features[name] = oof_preds
        # Average test predictions across folds
        test_meta_features[name] = np.mean(test_preds_fold, axis=0)
        
        rmse = np.sqrt(mean_squared_error(y, oof_preds))
        logger.info(f"{name} OOF RMSE: {rmse:.4f}")

    logger.info("Training Meta Model (Level 2)...")
    
    # Simple Ridge or Linear Regression is usually best for stacking to avoid overfitting
    meta_model = Ridge(alpha=10.0) 
    meta_model.fit(train_meta_features, y)
    
    # Coefficients
    logger.info("Meta Model Coefficients:")
    for name, coef in zip(train_meta_features.columns, meta_model.coef_):
        logger.info(f"{name}: {coef:.4f}")
        
    # Evaluate Meta Model on OOF (approximation)
    # Strictly we should CV the meta model too, but this gives a quick check
    meta_oof_preds = meta_model.predict(train_meta_features)
    meta_rmse = np.sqrt(mean_squared_error(y, meta_oof_preds))
    logger.info(f"Stacking Ensemble OOF RMSE: {meta_rmse:.4f}")
    
    # Final Prediction
    final_predictions = meta_model.predict(test_meta_features)
    
    # Save Submission
    submission_path = os.path.join(os.path.dirname(config['outputs']['submission_path']), "submission_stacking.csv")
    sample_sub_path = config['data']['sample_submission_path']
    if os.path.exists(sample_sub_path):
        submission = pd.read_csv(sample_sub_path)
        submission[target_col] = final_predictions
        submission.to_csv(submission_path, index=False)
        logger.info(f"Stacking submission saved to {submission_path}")

if __name__ == "__main__":
    train_stacking()
