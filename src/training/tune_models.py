import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import sys
import os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils import load_config, setup_logger

def tune_models():
    config = load_config()
    logger = setup_logger("TUNING", os.path.join(config['outputs']['logs'], "tuning.log"))
    
    # Load Data
    train_path = config['outputs']['processed_train']
    target_col = config['pipeline']['target']
    train_df = pd.read_csv(train_path)
    
    # Selected Features (Best Set)
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
    
    kf = KFold(n_splits=5, shuffle=True, random_state=config['pipeline']['random_state'])
    
    def objective_xgb(trial):
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': config['pipeline']['random_state'],
            'n_jobs': -1,
            'early_stopping_rounds': 100
        }
        
        rmse_scores = []
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            preds = model.predict(X_val)
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
            
        return np.mean(rmse_scores)

    def objective_lgbm(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'random_state': config['pipeline']['random_state'],
            'n_jobs': -1,
            'verbosity': -1
        }
        
        rmse_scores = []
        # LightGBM requires callbacks for early stopping in sklearn API
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=callbacks
            )
            preds = model.predict(X_val)
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
            
        return np.mean(rmse_scores)

    logger.info("Tuning XGBoost...")
    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(objective_xgb, n_trials=30) # 30 trials for speed, can increase
    logger.info(f"Best XGB Params: {study_xgb.best_params}")
    logger.info(f"Best XGB RMSE: {study_xgb.best_value}")

    logger.info("Tuning LightGBM...")
    study_lgbm = optuna.create_study(direction='minimize')
    study_lgbm.optimize(objective_lgbm, n_trials=30)
    logger.info(f"Best LGBM Params: {study_lgbm.best_params}")
    logger.info(f"Best LGBM RMSE: {study_lgbm.best_value}")
    
    # Save best params
    best_params = {
        'xgboost': study_xgb.best_params,
        'lightgbm': study_lgbm.best_params
    }
    
    with open("best_params.yaml", "w") as f:
        yaml.dump(best_params, f)
    logger.info("Best parameters saved to best_params.yaml")

if __name__ == "__main__":
    tune_models()
