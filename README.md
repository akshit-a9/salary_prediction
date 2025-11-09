# Salary Prediction Challenge – EEE-G513

This repository contains the implementation for the **Salary Prediction Challenge** conducted as part of the *Machine Learning for Electronics Engineers (EEE-G513)* course.

## Overview
The objective is to predict the average salary of job listings using structured attributes such as location, role, experience, and cost-of-living data. The solution uses feature engineering and regression modeling for accurate prediction.

## Dataset
- `train.csv` – training data with job attributes and target (`salary_average`)
- `test.csv` – evaluation data (2790 rows expected)
- `cost_of_living.csv` *(optional)* – auxiliary dataset merged using `city` and `country` keys

## Methodology
1. **Data Preparation**
   - Read and merge auxiliary cost-of-living data if available
   - Handle missing values and remove duplicate columns
2. **Feature Engineering**
   - Encode categorical variables (e.g., job titles, cities)
   - Normalize numerical features
   - Generate composite and interaction features
3. **Model Development**
   - Evaluate multiple regression models (Linear, Ridge, LightGBM, XGBoost)
   - Select the model with the lowest cross-validation RMSE
4. **Inference**
   - Generate predictions for the official test dataset

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/salary-prediction-eee-g513.git
   cd salary-prediction-eee-g513
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook notebookbf06637a61.ipynb
   ```

## Notes
- Ensure the `train.csv` and `test.csv` files are correctly placed in the input directory before running.
- The official test file must contain exactly 2790 rows.
