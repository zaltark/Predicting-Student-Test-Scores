import matplotlib.pyplot as plt
import seaborn as sns
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

def compare_models():
    config = load_config()
    logger = setup_logger("MODEL_COMPARISON", os.path.join(config['outputs']['logs'], "model_comparison.log"))
    
    logger.info("Starting Model Comparison...")
    
    # Load data
    train_path = config['outputs']['processed_train']
    target_col = config['pipeline']['target']
    
    logger.info(f"Loading processed data from {train_path}")
    train_df = pd.read_csv(train_path)
    
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    # Define models
    models = {
        "Linear_Regression": LinearRegression(),
        "Ridge_Regression": Ridge(alpha=1.0),
        "Lasso_Regression": Lasso(alpha=0.1),
        "Random_Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=config['pipeline']['random_state'], n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=6, random_state=config['pipeline']['random_state'], n_jobs=-1),
        "LightGBM": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=config['pipeline']['random_state'], n_jobs=-1, verbose=-1),
        "CatBoost": CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_seed=config['pipeline']['random_state'], verbose=False, allow_writing_files=False)
    }
    
    # K-Fold Cross Validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config['pipeline']['random_state'])
    
    results = {}
    oof_preds = {name: np.zeros(len(X)) for name in models.keys()}
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        rmse_scores = []
        
        for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            # Linear models benefit from scaling
            if name in ["Linear_Regression", "Ridge_Regression", "Lasso_Regression"]:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
            
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            
            oof_preds[name][val_index] = preds
            
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            rmse_scores.append(rmse)
        
        avg_rmse = np.mean(rmse_scores)
        results[name] = avg_rmse
        logger.info(f"{name} Average RMSE: {avg_rmse:.4f}")

    logger.info("========================================")
    logger.info("   MODEL COMPARISON RESULTS")
    logger.info("========================================")
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1]))
    
    for name, score in sorted_results.items():
        logger.info(f"{name}: {score:.4f}")
        
    best_model_name = list(sorted_results.keys())[0]
    logger.info(f"Best Model: {best_model_name}")
    
    # Residual Analysis
    logger.info("========================================")
    logger.info("   RESIDUAL CORRELATION ANALYSIS")
    logger.info("========================================")
    
    residuals_df = pd.DataFrame()
    for name, preds in oof_preds.items():
        residuals_df[name] = y - preds
        
    corr_matrix = residuals_df.corr()
    print(corr_matrix)
    logger.info("\n" + str(corr_matrix))
    
    # Plot Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", vmin=0, vmax=1)
    plt.title("Residual Correlation Matrix")
    output_path = os.path.join(config['outputs']['visualizations'], "residual_correlation_matrix.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Residual correlation matrix saved to {output_path}")

if __name__ == "__main__":
    compare_models()
