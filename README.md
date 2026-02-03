# Credit-Fraud-Detection
Credit Card Fraud Detection (End-to-End ML System)
An end-to-end machine learning system for detecting fraudulent credit card transactions using a real-world banking dataset (284,807 transactions). The project combines supervised classification and unsupervised anomaly detection in a complete pipeline covering data preprocessing, time-aware training, model evaluation, threshold tuning, and deployment, with support for real-time and batch predictions.

🔍 Key Highlights

Built a fully automated ML pipeline for fraud detection with extensive preprocessing, scaling, and imbalance handling.

Implemented supervised models including Logistic Regression, KNN, Random Forest, and XGBoost, with model selection via RandomizedSearchCV.

Added unsupervised anomaly detection using clustering-based methods (K-Means / Hierarchical) to capture previously unseen fraud patterns.

Designed an ensemble decision strategy combining supervised probabilities and unsupervised anomaly signals.

Applied class weighting and decision threshold optimization to balance precision and recall under extreme class imbalance (0.17% fraud).

Achieved strong performance: Accuracy 99.96% | Precision 0.92 | Recall 0.76 | F1-score 0.83 | ROC-AUC 0.99 | PR-AUC 0.81.

Developed an interactive web dashboard for single and batch transaction prediction with model performance insights.
