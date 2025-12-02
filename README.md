# Credit-Fraud-Detection
Credit Card Fraud Detection (End-to-End ML System)
An end-to-end machine learning project to detect fraudulent credit card transactions using real-world banking dataset (284,807 records). Designed a full pipeline for model training, evaluation, and deployment with real-time prediction support.

🔍 Key Highlights

- Built automated ML pipeline for fraud detection with extensive preprocessing & feature handling.

- Experimented with multiple models — Logistic Regression, KNN, XGBoost, Random Forest — and selected the best using RandomizedSearchCV.

- Achieved industry-relevant performance using class imbalance handling via class weighting, threshold optimization & performance tuning.

- Final chosen model: Random Forest Classifier due to superior precision and balanced performance.

- Final model results:
  Accuracy: 99.96% | Precision: 0.92 | Recall: 0.76 | F1-Score: 0.83 | ROC-AUC: 0.99 | PR-AUC: 0.81

- Added custom decision threshold logic to reduce false positives real-world usage.

- Developed interactive web dashboard enabling single & batch transaction prediction and model performance analysis.
