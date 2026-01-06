# Kaggle Playground Series - S6E1: Predict Student Test Scores

## Project Overview
Welcome to the 2026 Kaggle Playground Series! This competition focuses on predicting students' test scores using a synthetically-generated dataset. The goal is to provide a beginner-friendly "sandbox" for practicing machine learning skills on tabular data.

## Goal
The objective is to predict the `exam_score` for each student in the test set.

## Evaluation
Submissions are evaluated using the **Root Mean Squared Error (RMSE)** between the predicted and the observed target.

## Dataset Description
The dataset was generated synthetically from a deep learning model trained on real-world data. While it captures many patterns, it may contain "artifacts" or slight distributional shifts compared to real-world data.

- `train.csv`: The training set, including the target `exam_score`.
- `test.csv`: The test set, for which you must predict the `exam_score`.
- `sample_submission.csv`: A sample submission file in the correct format.

## Submission Format
For each `id` in the test set, you must predict a value for the `exam_score` variable. The file should contain a header and have the following format:

```csv
id,exam_score
630000,97.5
630001,89.2
630002,85.5
...
```

## Timeline
- **Start Date:** January 1, 2026
- **Final Submission Deadline:** January 31, 2026 (11:59 PM UTC)

## Citation
Yao Yan, Walter Reade, Elizabeth Park. Predicting Student Test Scores. https://kaggle.com/competitions/playground-series-s6e1, 2025. Kaggle.

## Project Structure

This project is organized into a modular pipeline:

- **src/exploration/**: Scripts for initial data analysis and visualization.
- **src/processing/**: Scripts for data cleaning and preprocessing.
- **src/features/**: Scripts for feature engineering.
- **src/training/**: Scripts for model training and evaluation.
- **src/pipeline.py**: Main orchestration script to run the entire pipeline.
- **config.yaml**: Configuration file for paths and parameters.
- **outputs/**: Directory for storing logs, artifacts, visualizations, and submissions.

## Usage

1.  **Configuration**: Modify `config.yaml` to set your data paths and parameters.
2.  **Run Pipeline**: Execute the full pipeline using:
    ```bash
    python src/pipeline.py
    ```
    This will run exploration, preprocessing, feature engineering, and training in sequence.
3.  **Individual Steps**: You can also run individual steps, e.g.:
    ```bash
    python src/exploration/explore.py
    ```
