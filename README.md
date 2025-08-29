
# Customer Churn Prediction

A complete, reproducible baseline to predict customer churn for a telecom-style dataset.

## Files
- `churn_data.csv` — synthetic dataset
- `churn_model.py` — trains Logistic Regression and Random Forest, prints metrics, saves a feature importance chart

## How to Run
```bash
pip install pandas scikit-learn matplotlib
python churn_model.py
```

## What You'll See
- Classification report (precision/recall/F1)
- ROC AUC for both models
- `feature_importance.png` saved in the repo

## Next Ideas
- SMOTE for class imbalance
- Hyperparameter tuning (GridSearchCV/Optuna)
- Calibration (CalibratedClassifierCV)
