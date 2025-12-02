import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from scipy.stats import randint
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np
import joblib

print(" Training started...")

#LOAD DATA
df = pd.read_csv("creditcard.csv")

# Sort chronologically
df = df.sort_values(by="Time")

# Split features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Time-based split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Remove Time column from final features
X_train = X_train.drop(columns=["Time"])
X_test = X_test.drop(columns=["Time"])

tscv = TimeSeriesSplit(n_splits=5)

# RANDOM FOREST MODEL 
rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    ))
])

rf_param_dist = {
    "model__max_depth": [10, 15, 20],
    "model__min_samples_split": randint(2, 6),
    "model__min_samples_leaf": randint(1, 4),
}

rf_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_param_dist,
    n_iter=5,
    scoring="roc_auc",
    cv=tscv,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print("\n Training Random Forest...")
rf_search.fit(X_train, y_train)
rf_score = rf_search.best_score_
print(f" RF Best ROC-AUC = {rf_score:.4f}")

#XGBOOST MODEL
xgb_pipeline = Pipeline([
    ("model", XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,
        random_state=42,
        eval_metric="logloss"
    ))
])

xgb_param_dist = {
    "model__max_depth": [3, 4, 5],
    "model__learning_rate": [0.01, 0.05, 0.1],
}

xgb_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=xgb_param_dist,
    n_iter=5,
    scoring="roc_auc",
    cv=tscv,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print("\n Training XGBoost...")
xgb_search.fit(X_train, y_train)
xgb_score = xgb_search.best_score_
print(f" XGB Best ROC-AUC = {xgb_score:.4f}")

#SELECT BEST MODEL
if xgb_score > rf_score:
    best_model = xgb_search.best_estimator_
    best_name = "XGBoost"
    best_score = xgb_score
else:
    best_model = rf_search.best_estimator_
    best_name = "RandomForest"
    best_score = rf_score

print("\n BEST MODEL SELECTED:", best_name)

#THRESHOLD OPTIMIZATION
y_proba_test = best_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_test)

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"🎯 Optimal Threshold = {best_threshold:.4f}")

# SAVE MODEL & THRESHOLD 
joblib.dump(
    {"model": best_model, "X_test": X_test, "y_test": y_test, "threshold": float(best_threshold)},
    "fraud_model_final.pkl"
)

print("💾 Model saved to fraud_model_final.pkl")
print("🎉 Training complete")


