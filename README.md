# Kaggle Playground Series - S6E1: Predict Student Test Scores

## Executive Summary
This project is a high-performance machine learning system designed to predict students' exam scores based on various academic and lifestyle factors (e.g., study hours, attendance, sleep quality). By employing an iterative, data-driven approach, the system identifies the most influential predictors of success and selects the most accurate mathematical model to forecast outcomes.

**Key Achievement:** Developed a predictive model with an average error of only **8.76 points**, outperforming standard linear approaches by effectively capturing complex, non-linear patterns in student behavior.

---

## The Data Science "Practice"
To ensure reliability and scalability, this project follows industry-standard software engineering and data science practices:

1.  **Modular Pipeline Design:** Instead of one large script, the project is broken into specialized modules (Exploration, Cleaning, Feature Engineering, Training). This makes the code easier to test, maintain, and update.
2.  **Data Auditing & Cleaning:** We perform rigorous "schema identification" to ensure data quality, handling hidden patterns like outliers and distribution shifts that could bias the results.
3.  **Advanced Feature Engineering:** We use **Target Encoding** to translate categorical information (like "Study Method") into numerical signals that the model can understand, significantly boosting the accuracy of simpler models.
4.  **Rigorous Validation (5-Fold CV):** To ensure the model doesn't just "memorize" the training data (overfitting), we test it five separate times on different subsets of the data to guarantee it will perform well on new, unseen students.
5.  **Explainable AI (XAI):** We use **SHAP** analysis to peel back the "black box" of machine learning, providing clear visualizations of *why* the model made a specific prediction.

---

## Technology Stack & Tools

*   **Python:** The core programming language used for all data processing and modeling.
*   **XGBoost / LightGBM / CatBoost:** State-of-the-art "Gradient Boosting" algorithms. These are the "heavy lifters" of the project, capable of learning complex relationships in tabular data.
*   **Scikit-Learn:** The industry-standard library for data preprocessing, linear modeling, and performance evaluation.
*   **SHAP (SHapley Additive exPlanations):** A specialized tool for model interpretability, helping us identify that `study_hours` and `attendance` are the primary drivers of exam scores.
*   **Pandas & NumPy:** Essential tools for high-speed data manipulation and numerical computation.
*   **YData-Profiling:** An automated tool used to generate comprehensive diagnostic reports on the data's health.

---

## Project Structure

This project is organized into a modular pipeline:

- **src/exploration/**: Scripts for initial data analysis, schema identification, and SHAP analysis.
- **src/processing/**: Scripts for data cleaning (log transformation, outlier handling).
- **src/features/**: Scripts for feature engineering (Target Encoding).
- **src/training/**: Scripts for model training (XGBoost, LightGBM, CatBoost, Linear Models) and comparison.
- **src/pipeline.py**: Main orchestration script to run the entire pipeline.
- **config.yaml**: Configuration file for paths and parameters.
- **outputs/**: Directory for storing logs, artifacts, visualizations, and submissions.

## Results

**Champion Model:** Stacking Ensemble (Tuned XGBoost + LightGBM + Linear)
**Best RMSE:** ~8.7333

### Key Configuration
*   **Ensemble Strategy:** Level 1 (XGB, LGBM, CatBoost, RF, Linear) -> Level 2 (Ridge Regression).
*   **Hyperparameter Tuning:** XGBoost and LightGBM optimized via Optuna (100 rounds early stopping).
*   **Selected Features:** `sleep_hours`, `facility_rating_te`, `study_method_te`, `sleep_quality_te`, `class_attendance`, `study_hours`.

### Model Comparison (5-Fold CV)
1.  **Stacking Ensemble (Tuned):** 8.7333
2.  **XGBoost (Tuned):** 8.7465
3.  **LightGBM (Tuned):** 8.7478
4.  **XGBoost (Baseline):** 8.7685
5.  **LightGBM (Baseline):** 8.7746
6.  **CatBoost:** 8.8041
7.  **Linear/Ridge Regression:** 8.9117

### Key Insights
*   **Ensemble Power:** Stacking diverse models (Tree-based + Linear) yields the best performance by allowing the meta-model to correct individual biases.
*   **Tuning Matters:** Optuna tuning improved individual model scores by ~0.02-0.03 RMSE, which directly translated to a better ensemble.
*   **Feature Selection Wins:** Reducing the model to the top 6 most impactful features improved performance by removing noise.
*   **Target Encoding:** Significantly improved the performance of Linear Regression, making it a valuable contributor to the stack.

## Tested Theories & Experiments

| Experiment | Description | Outcome | Decision |
| :--- | :--- | :--- | :--- |
| **Outlier Clipping** | Clipped top 1% of continuous features. | No significant change in RMSE. | Reverted (kept raw/log). |
| **Log Transformation** | Applied `log1p` to `study_hours`, `attendance`. | No significant change in RMSE. | Kept for stability. |
| **Binning** | Binned `sleep_hours` into 5 categories. | RMSE degraded slightly (8.7761). | Reverted. |
| **Interaction Features** | Created simple interactions (e.g., `study_hours * method`). | RMSE degraded slightly (8.7614). | Reverted. |
| **Weighted Training** | Biased LightGBM (low scores) & XGBoost (high scores). | RMSE degraded significantly (~9.12). | Reverted. |
| **Stacking** | Ensembled multiple models with a Ridge meta-learner. | **Best Result (8.7333)**. | **Adopted.** |

## Outputs & Artifacts

- **Submission File:** `outputs/submissions/submission_stacking.csv`
- **EDA Reports:** `outputs/visualizations/eda_report.html` (Raw), `eda_report_processed.html` (Processed)
- **SHAP Analysis:** `outputs/visualizations/shap_summary_beeswarm.png`, `shap_feature_importance.png`
- **Residual Analysis:** `outputs/visualizations/residual_correlation_matrix.png`, `residual_plot.png`
- **Tuned Params:** `best_params.yaml`

## Usage

1.  **Configuration**: Modify `config.yaml` to set your data paths and parameters.
2.  **Data Analysis**: Identify schema and data quality:
    ```bash
    python src/exploration/identify_schema.py
    ```
3.  **Automated EDA**: Generate a comprehensive HTML profiling report:
    ```bash
    python src/exploration/eda_profiling.py
    ```
    The report will be saved to `outputs/visualizations/eda_report.html`.
4.  **Run Pipeline**: Execute the full pipeline using:
    ```bash
    python src/pipeline.py
    ```
    This will run exploration, preprocessing, feature engineering, and training in sequence.
