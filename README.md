# SECOM Datset ML Project

### Decisions

- **Addressing columns with high null percentage**
  - Remove >90% missing columns
  - Median imputation

- **Addressing correlated features**
  - Remove zero variance features
    - checking the lognormal distribution of variances does not show a clear cutoff to remove obvious non-zero variance features
  - Correlation with target (mututal information GRC check)
  - Remove highly correlated sensor duplicates

- **Train baseline models**
  - Train L1 Logistic regression model and check coefficients
  - Train XGBoost model and feature importance scores/ SHAP
  - If non-linear model has improved performance, interactions matter
  - Re-train model with reduced feature set using import features to simplify model, if possible
