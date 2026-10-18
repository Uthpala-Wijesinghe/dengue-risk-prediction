# Sri Lanka Dengue Risk Prediction with Ensemble Methods

Random Forest and XGBoost, compared against each other, a naive baseline, and two ensembles built from them (soft voting and stacking), for predicting dengue risk in Sri Lanka's 25 districts. Includes single-sample inference from a saved pickle model and a Streamlit front end.

## Problem

Dengue returns to Sri Lanka every year around the monsoons, and some years (2017: over 186,000 cases) it overwhelms hospitals. Given last month's situation in a district, the models predict:

1. **Risk zone** (main task): Safety / Warning / Epidemic, using the WHO Endemic Channel method (district mean and mean + 2 SD, built from past years only)
2. **Epidemic alarm**: Epidemic or not
3. **Case count**: expected cases that month

## Dataset (local, Sri Lankan)

`data/cleaned_dengue_data.csv`: 3,425 district-months (25 districts, Jan 2015 to May 2026). I built this table from three public sources:

| Source | Content |
|---|---|
| Ministry of Health Weekly Epidemiological Reports, via [`denguedatahub`](https://github.com/thiyangt/denguedatahub) | Weekly dengue cases per district (`data/srilanka_weekly_data.csv`, 2006 onwards) |
| [Open-Meteo Historical Weather API](https://open-meteo.com/) (ERA5) | Monthly temperature, rainfall, humidity, wind, elevation per district |
| Census of Population and Housing 2024, via [`nuuuwan/lk_census_2024`](https://github.com/nuuuwan/lk_census_2024) | Population density, urban share, youth share per district |

## Results (test years 2024-2026, trained on 2015-2023)

Models are compared in order of complexity: weak learners → bagging → Random Forest → boosting → voting / stacking.

| Model | Zone: accuracy / macro F1 | Epidemic: ROC-AUC / F1 | Cases: MAE / R² |
|---|---|---|---|
| Naive baseline (zone) / last month's cases | 0.510 / 0.320 | - | 60.9 / 0.756 |
| Decision Tree (tuned with GridSearchCV + pruning) | 0.612 / 0.461 | - | 78.6 / 0.603 |
| Logistic Regression (L2, lbfgs, C tuned) | **0.709 / 0.621** | **0.849** / 0.190 | - |
| Bagging (500 trees, all features) | 0.693 / 0.571 | - | - |
| Random Forest | 0.696 / 0.553 | 0.811 / 0.346 | 60.0 / 0.811 |
| XGBoost | 0.702 / 0.581 | 0.837 / 0.248 | 58.0 / 0.821 |
| **Soft Voting (RF + XGB), deployed** | 0.693 / 0.559 | 0.827 / **0.371** | **57.4 / 0.824** |
| Stacking (RF + XGB → meta-model) | 0.671 / 0.508 | - | 76.7 / 0.704 |
| Published approach (Fan et al. 2025: SMOTE, XGB depth 5) on the same time split | 0.659 / 0.545 | - | - |

- Cross-validation on yearly folds could not separate the main models (differences smaller than the fold-to-fold spread), so by the rule set in the notebook the **soft voting** ensemble is deployed. RF vs XGBoost: McNemar p = 0.69.
- **Logistic Regression was the best model on the test years**, but not significantly better than voting (McNemar p = 0.34). Ensembles are not automatically better.
- Stacking was best in cross-validation but the worst ensemble on test.
- Reproducing the published approach shows that a random 80/20 split inflates accuracy on this time-ordered data (73.9% vs 65.9% on a time split).
- Last month's cases and the neighbouring districts' cases drive the predictions far more than weather (impurity and permutation importance, SHAP, partial dependence plots).

## Hyperparameters (no defaults)

Every parameter was chosen from the data's characteristics and explained in notebook Section 6, then checked with an OOB-error vs test-error curve, depth sweeps on year-by-year validation folds, and a GridSearchCV around the designed values (kept, since the tuned gains were within fold-to-fold noise).

- **Decision Tree:** GridSearchCV over criterion, max_depth, min_samples_split, min_samples_leaf → `entropy, 6, 40, 10`; cost-complexity pruning curve.
- **Logistic Regression:** `solver="lbfgs", penalty="l2", C=1.0` (C from GridSearchCV), `max_iter=2000`.
- **Random Forest:** `n_estimators=500, criterion="gini", max_depth=10, min_samples_split=10, min_samples_leaf=5, max_features=0.4, max_samples=0.8, oob_score=True`; regression forest uses Breiman's `max_features=1/3, min_samples_leaf=5`.
- **XGBoost:** `learning_rate=0.03, max_depth=4, min_child_weight=3, subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, reg_alpha=0.1, gamma=0.1`, `n_estimators` from early stopping on 2023 (143 trees); `count:poisson` objective for case counts, `scale_pos_weight` for the Epidemic alarm.

## Repository structure

```
├── Dengue_Risk_Prediction.ipynb       # full analysis, run top to bottom
├── data/
│   ├── cleaned_dengue_data.csv        # modelling table
│   ├── model_ready.csv                # features + all three targets (written by the notebook, used by the dashboard)
│   └── srilanka_weekly_data.csv       # raw weekly cases (for the Endemic Channel baselines)
├── models/
│   ├── model.pkl                      # saved pipelines (written by the notebook)
│   ├── holdout_sample.csv             # the held-out Colombo row used for inference
│   └── results.json                   # notebook result tables shown in the dashboard
├── app/app.py                         # Streamlit dashboard (Predict / Train & Compare / Notebook Results)
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt

# 1. the notebook (about 3-4 minutes); recreates models/model.pkl, results.json, data/model_ready.csv
jupyter notebook Dengue_Risk_Prediction.ipynb

# 2. the dashboard
streamlit run app/app.py
```

In the app, click **Fill in the held-out example** and then **Predict** to reproduce the notebook's single-sample inference (Colombo, May 2026: predicted Warning, ~1,030 cases; actual Safety, 636 cases, see notebook Section 13 for why). The **Train & Compare** tab lets you upload a CSV (or use the dengue table), pick a target and models, adjust the RF/XGBoost settings and compare them.

## Front end (dashboard)

`app/app.py` loads the saved pipelines from `models/model.pkl` and has three tabs: **Predict** (enter a district-month and get the risk zone, Epidemic probability and expected cases), **Train & Compare** (retrain and compare the models on the dengue table or your own CSV) and **Notebook Results** (the result tables from the notebook).

## References

- World Health Organization (2009). *Dengue: Guidelines for Diagnosis, Treatment, Prevention and Control.*
- Bortman, M. (1999). Elaboración de corredores o canales endémicos mediante planillas de cálculo. *Rev Panam Salud Publica*, 5(1).
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16.*
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5-32.
- Fan, X. et al. (2025). Prediction of outpatient rehabilitation patient preferences and optimization of graded diagnosis and treatment based on XGBoost machine learning algorithm. *Frontiers in Artificial Intelligence*, 7:1473837.
