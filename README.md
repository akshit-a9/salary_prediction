# Salary Prediction Challenge – EEE-G513

This repository contains the implementation for the **Salary Prediction Challenge** conducted as part of the *Machine Learning for Electronics Engineers (EEE-G513)* course at BITS Pilani, K.K. Birla Goa Campus.

## Overview
The task is to predict the **average salary (`salary_average`)** across different job roles and global cities, considering variations in the cost of living.  
The project implements a robust regression pipeline combining data preprocessing, feature engineering, and ensemble modeling (LightGBM + CatBoost).

## Dataset
- `train.csv` – Training data containing 6,525 samples across 700 cities  
- `test.csv` – Evaluation data (official test set contains **2,799 rows**)  
- `cost_of_living.csv` – Supplementary data with 52 economic indicators merged via `city_id` or (`city`, `country`)

## Methodology
1. **Data Preparation**
   - Merged cost-of-living indicators into train/test datasets (1:1 per city)
   - Cleaned missing values and removed duplicates
   - Applied log transformation to skewed numerical features
   - Winsorized outliers and standardized numeric ranges

2. **Feature Engineering**
   - Added *target encodings* for country, role, and (country × role) combinations
   - Categorized non-numeric fields for gradient-boosting models
   - Used GroupKFold cross-validation by `city_id` to simulate unseen cities

3. **Modeling**
   - Trained **LightGBM** and **CatBoost** regressors with log-transformed targets
   - Optimized hyperparameters and applied early stopping
   - Blended both models with a weighted average for improved generalization

4. **Evaluation**
   - Metric: **Root Mean Square Percentage Error (RMSPE)**  
   - Achieved **Public RMSE: 1.04975**, currently **Rank 9** on Kaggle Leaderboard

5. **Submission Handling**
   - The official evaluator expects **2,790 rows** with columns: `ID,salary_average`
   - Submission builder enforces exact row count and correct header alignment

## Usage
```bash
git clone https://github.com/akshit-a9/salary_prediction.git
cd salary_prediction
pip install -r requirements.txt
jupyter notebook attempt1.ipynb
```

## Notes
- Ensure the official competition dataset (2,790-row test file) is attached when running the notebook on Kaggle.  
- All predictions are generated reproducibly via cross-validation and blending.
