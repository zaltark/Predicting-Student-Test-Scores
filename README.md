# 🎓 Kaggle Playground S6E1: Predicting Student Test Scores

### **Executive Summary**
**Goal:** Can we predict how well a student will perform on an exam based on their daily habits?  
**Solution:** This project is a high-performance machine learning system that forecasts exam scores by analyzing factors like study hours, sleep quality, and attendance.  
**Result:** The system achieved a prediction error (RMSE) of just **~8.73 points**, outperforming standard statistical methods by detecting complex patterns in student behavior.

---

## 🏗️ The Approach: Engineering Over Scripting
Unlike a typical "messy" data science experiment, this project was built as a software product. We used industry-standard practices to ensure the code is reliable, readable, and ready for production.

### 1. Modular "Lego" Design
Instead of writing one massive, confusing script, the code is broken into specialized building blocks (Modules).
* **Why?** If the "cleaning" step breaks, we can fix it without touching the "training" step. It makes the project easier to test and upgrade.

### 2. The "Team of Experts" Strategy (Ensemble Learning)
Our champion model isn't just one algorithm—it's a **Stacking Ensemble**.
* **How it works:** We trained multiple models (XGBoost, LightGBM, and Linear Regression). Think of these as individual experts. We then used a "Manager" model (Ridge Regression) to listen to their opinions and make a final, more accurate prediction.

### 3. Stress Testing (5-Fold Cross-Validation)
To guarantee the model doesn't just memorize the practice questions (overfitting), we tested it five separate times on different groups of students it had never seen before.

### 4. Explainability (Opening the Black Box)
We don't just want a prediction; we want to know *why*. Using **SHAP analysis**, we visualized exactly which factors drove the score up or down (e.g., confirming that "Study Hours" mattered more than "Sleep").

---

## 📊 Results & Key Findings

| Model | RMSE Score (Lower is Better) | Notes |
| :--- | :--- | :--- |
| **Stacking Ensemble (Champion)** | **8.7333** | **Combined the best of Tree & Linear models.** |
| XGBoost (Tuned) | 8.7465 | Strong individual performer. |
| LightGBM (Tuned) | 8.7478 | Very fast and accurate. |
| Linear Regression | 8.9117 | Good baseline, but missed complex patterns. |

**Key Insights:**
* **Tuning Matters:** Using automated tools (Optuna) to fine-tune our models improved accuracy by ~0.03 points—a significant edge in competitions.
* **Less is More:** We found that feeding the model *every* piece of data actually confused it. Limiting it to the **top 6 most impactful features** improved performance.

---

## 🧪 Experiment Log (What Worked & What Didn't)
Data science is about trial and error. Here is a log of the experiments we ran to optimize the model.

| Experiment | Description | Outcome | Decision |
| :--- | :--- | :--- | :--- |
| **Stacking** | Combining multiple models with a meta-learner. | **Best Result (8.73)** | ✅ **Adopted** |
| **Outlier Clipping** | Removing the top 1% of extreme data points. | No change. | ❌ Reverted |
| **Log Transform** | Mathematically smoothing "Study Hours" data. | No error change, but safer. | ✅ Kept |
| **Interaction Features** | Creating new data like `Study * Sleep`. | Error increased (8.76). | ❌ Reverted |
| **Weighted Training** | Forcing the model to focus on low/high scores. | Error spiked to ~9.12. | ❌ Reverted |

---

## 📂 Project Structure
For developers and engineers, here is how the codebase is organized:

```text
├── config.yaml          # Control Center: Paths and settings
├── src/
│   ├── exploration/     # Data Health Checks & Schema ID
│   ├── processing/      # Cleaning & Transformations (Log, Outliers)
│   ├── features/        # Feature Engineering (Target Encoding)
│   ├── training/        # Model definitions (XGB, LGBM, Stacking)
│   └── pipeline.py      # The "Main" script that runs everything
└── outputs/             # Where reports, charts, and predictions go
