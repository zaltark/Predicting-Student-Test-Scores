import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils import load_config, setup_logger

def run_shap_analysis():
    config = load_config()
    logger = setup_logger("SHAP_ANALYSIS", os.path.join(config['outputs']['logs'], "shap_analysis.log"))
    
    logger.info("Starting SHAP Analysis...")
    
    # Load Model
    model_path = config['outputs']['model_path']
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # Load Data
    train_path = config['outputs']['processed_train']
    target_col = config['pipeline']['target']
    logger.info(f"Loading processed data from {train_path}")
    train_df = pd.read_csv(train_path)
    
    # Selected Features (must match training)
    selected_features = [
        'sleep_hours', 
        'facility_rating_te', 
        'study_method_te',    
        'sleep_quality_te',   
        'class_attendance', 
        'study_hours'
    ]
    
    # Sample data for SHAP (calculating on full dataset is too slow)
    sample_size = 5000
    if len(train_df) > sample_size:
        logger.info(f"Sampling {sample_size} rows for SHAP analysis")
        shap_df = train_df.sample(n=sample_size, random_state=config['pipeline']['random_state'])
    else:
        shap_df = train_df
        
    X_shap = shap_df[selected_features]
    y_shap = shap_df[target_col]
    
    # Explain predictions using SHAP
    logger.info("Calculating SHAP values...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_shap)
    
    # Save SHAP Summary Plot
    output_dir = config['outputs']['visualizations']
    logger.info("Generating SHAP Summary Plot...")
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary_beeswarm.png"), bbox_inches='tight')
    plt.close()
    
    logger.info("Generating SHAP Bar Plot...")
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.savefig(os.path.join(output_dir, "shap_feature_importance.png"), bbox_inches='tight')
    plt.close()
    
    # Residual Analysis
    logger.info("Calculating Residuals...")
    preds = model.predict(X_shap)
    residuals = y_shap - preds
    
    plt.figure(figsize=(10, 6))
    plt.scatter(preds, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.savefig(os.path.join(output_dir, "residual_plot.png"), bbox_inches='tight')
    plt.close()
    
    logger.info(f"Analysis saved to {output_dir}")
    logger.info("SHAP Analysis completed.")

if __name__ == "__main__":
    run_shap_analysis()
