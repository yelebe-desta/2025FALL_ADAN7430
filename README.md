Data Preprocessing

  Cleaned missing values and encoded categorical features using one-hot encoding.

A  ddressed severe class imbalance (only ~4% stroke cases) by applying class_weight='balanced' in Logistic Regression to give more importance to the minority class during training.

Model Training and Evaluation

  Trained Logistic Regression on the original training data (no oversampling).

  Evaluated performance on a held-out validation set using multiple metrics.

Logistic Regression Accuracy Score: 0.8640
Logistic Regression F1 Score: 0.2053
Logistic Regression Precision Score: 0.1352
Logistic Regression Recall Score: 0.4257
Logistic Regression AUC: 0.6543

