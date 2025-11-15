# Salary Prediction Challenge – EEE-G513

This repository contains the work completed for the Salary Prediction Challenge in the Machine Learning for Electronics Engineers course at BITS Pilani, K.K. Birla Goa Campus.

## Overview

The objective is to predict the average salary across global cities while accounting for job role, geography, and cost of living. The project evolved over eight numbered attempts, gradually improving preprocessing, feature engineering, validation, and model design. The final solution is a city-aware, feature-rich ensemble in log-salary space.

## Evolution Across Attempts (1–8)

Below is a concise, chronological summary of how the pipeline changed across attempts. Only the main differences are highlighted.

- **Attempt 1**  
  Initial baseline. Merged `cost_of_living.csv` into train and test, applied basic cleaning, log-transformed the target, and used GroupKFold by city. Trained LightGBM and CatBoost regressors and blended their predictions with a simple weighted average.

- **Attempt 2**  
  Introduced CV-safe smoothed target encodings for key categorical fields (country, state, role, and simple interactions). Added XGBoost to the model stack. Retained GroupKFold-by-city and log target, with a three-model blend on top of the new encodings.

- **Attempt 3**  
  Focused on richer numerical transformations. Added ratio-based features from cost-of-living indicators, along with PCA and KMeans clustering features to capture latent structure across cities. Continued to use LightGBM, CatBoost, and XGBoost with a blended ensemble.

- **Attempt 4**  
  Reworked target encoding into a more systematic, CV-safe module with smoothed encodings for country, state, role, and country × role. Added more domain-informed interaction features from COL (for example PPP vs rent/transport/food). On top of the base models, introduced a Ridge stacking layer that used out-of-fold predictions from LightGBM, CatBoost, and XGBoost as meta-features.

- **Attempt 5**  
  Stabilized training further. Tightened LightGBM and XGBoost hyperparameters (slower learning rates, more conservative leaves and min-data-in-leaf), and used additional scalers (RobustScaler / QuantileTransformer) for the meta-learner. Kept the CV-safe target encodings and stacking but focused on better regularization and more stable OOF behavior.

- **Attempt 6**  
  Enhanced how cost of living is used. Built composite COL aggregates (mean, max, min, range over key indicators such as rent, groceries, restaurants, transport) and an affordability index that normalizes COL by purchasing power. Combined these with smoothed target encodings and a “smart” ensemble that compared pure blends with Ridge stacking based on OOF RMSPE.

- **Attempt 7**  
  Switched to a more global, hierarchical feature view. Constructed country, state, and role level statistics in log-salary space (means, medians, standard deviations, ranges) and mapped them back to rows as `te_*` features. Added COL-adjusted variants (for example salary per COL unit, role and country per COL, and log-transformed TE features). The modeling focused primarily on a strong LightGBM configuration over this expanded feature set, simplifying the ensemble logic.

- **Attempt 8 (Final)**  
  Refactored the full pipeline into a clean script (v3.1-style).  
  - Kept city-aware GroupKFold and log1p target.  
  - Implemented hierarchical encodings for country, state, and role, including counts, deviations, ratios, variance proxies, and COL-adjusted purchasing power features.  
  - Built rich COL aggregations (mean, median, std, range, IQR, skew, category-level proxies) with robust NaN/inf handling.  
  - Trained a four-model ensemble: two differently regularized LightGBM models, one XGBoost model, and one GradientBoostingRegressor.  
  - Used out-of-fold predictions from all models and optimized ensemble weights via SciPy’s SLSQP to directly minimize RMSPE. The optimized weighted ensemble defines the final solution.

The final public notebook for the ensemble corresponds to Attempt 8, and its submission was executed by **Ayushman**.

## Final Methodology (Brief)

- Merge `cost_of_living.csv` into train and test on city-level keys.  
- Work in log1p(salary_average) space and evaluate with RMSPE.  
- Build hierarchical target encodings and statistics over country, state, and role, with fallbacks for sparse combinations.  
- Derive COL-based aggregates (central tendency, dispersion, percentiles, category proxies) and interaction features that relate salary statistics to COL (ratios, adjusted means, purchasing power).  
- Use GroupKFold with city as the grouping variable to simulate unseen-city generalization.  
- Train a four-model ensemble (two LightGBM variants, XGBoost, GradientBoosting) and collect out-of-fold predictions.  
- Optimize model weights with SciPy under a simplex constraint (non-negative, sum to one) to minimize RMSPE on OOF predictions, then apply the weighted ensemble to the test set and exponentiate back to salary space.

## Usage

```bash
git clone https://github.com/akshit-a9/salary_prediction.git
cd salary_prediction
pip install -r requirements.txt
```

Open and run the relevant notebook (for example `attempt8.ipynb`) to reproduce the final pipeline and submission file.

## Team

- **Akshit Sharma** - 2025H1400069G
- **Ayushman** (github.com/ayshmnmm) - 2025H1400065G
