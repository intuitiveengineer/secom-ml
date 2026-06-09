# SECOM Manufacturing Failure Prediction

## Objective

Develop machine learning models to predict semiconductor manufacturing failures using the UCI SECOM dataset.

The SECOM dataset contains 1,567 manufacturing observations with 590 process sensor measurements and a highly imbalanced target variable (~6.6% failure rate).

---

## Dataset Challenges

- High-dimensional feature space (590 sensors)
- Significant missing data
- Strong feature correlation and redundancy
- Severe class imbalance
- Limited number of failure examples

---

## Preprocessing

### Missing Data Handling

- Removed features with >90% missing values
- Applied median imputation to remaining missing values

### Feature Reduction

- Removed constant and near-constant features
- Removed highly correlated features
- Evaluated feature relevance using mutual information and model-based importance measures

### Model Evaluation

- Stratified train/test splitting
- Cross-validation
- ROC-AUC evaluation
- Precision-Recall AUC evaluation
- Threshold analysis for imbalanced classification

---

## Models Explored

- L1-Regularized Logistic Regression
- Random Forest
- XGBoost

---

## Current Results

Current models achieve ROC-AUC values in the ~0.70–0.80 range, demonstrating meaningful predictive signal despite the challenging class imbalance and high-dimensional feature space.

---

## Current Research

Areas currently being explored:

- Leakage-free preprocessing pipelines
- Feature engineering
- PCA and latent variable methods
- Partial Least Squares (PLS) and PLS-DA
- Threshold optimization for failure detection
- Model interpretation and feature importance analysis

---

## Future Work

- Robust nested cross-validation
- Advanced feature selection techniques
- Ensemble methods and stacking
- Time-dependent process trajectory analysis
- Manufacturing domain interpretation of key sensors
