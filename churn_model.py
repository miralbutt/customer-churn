
"""
Customer Churn Prediction
-------------------------
Train a simple baseline model (Logistic Regression + RandomForest)
and output metrics and a feature importance chart.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("churn_data.csv")

X = df.drop(columns=["churn", "customer_id"])
y = df["churn"]

categorical = X.select_dtypes(include=["object"]).columns.tolist()
numeric = X.select_dtypes(exclude=["object"]).columns.tolist()

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", "passthrough", numeric)
])

# Logistic Regression pipeline
logit_clf = Pipeline([
    ("prep", pre),
    ("clf", LogisticRegression(max_iter=1000))
])

# Random Forest pipeline
rf_clf = Pipeline([
    ("prep", pre),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

for name, model in [("LogisticRegression", logit_clf), ("RandomForest", rf_clf)]:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]
    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, digits=3))
    print("ROC AUC:", roc_auc_score(y_test, proba))

# Plot feature importances for RandomForest
rf = rf_clf.named_steps["clf"]
# Get feature names after one-hot
ohe = rf_clf.named_steps["prep"].transformers_[0][1]
cat_cols = ohe.get_feature_names_out(input_features=categorical)
feature_names = list(cat_cols) + numeric
importances = rf.feature_importances_

fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
plt.figure()
fi[::-1].plot(kind="barh")
plt.title("Top 20 Feature Importances (RandomForest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("\nSaved chart: feature_importance.png")
