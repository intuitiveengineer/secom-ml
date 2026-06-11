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

## Exploratory Data Analysis Findings

A comprehensive exploratory analysis was performed to better understand the structure of the SECOM dataset before model development.

Methods explored included:

- Principal Component Analysis (PCA)
- Partial Least Squares (PLS)
- UMAP
- t-SNE
- Kernel PCA
- Gaussian Mixture Models (GMM)
- Sensor-level risk analysis
- Correlation analysis
- Mutual information analysis

Several consistent findings emerged:

- No clear failure cluster was observed in PCA, UMAP, t-SNE, Kernel PCA, or GMM visualizations.
- Failures appear throughout the sensor space rather than forming a distinct population.
- Multiple supervised and unsupervised methods repeatedly identified a small group of sensors as being associated with elevated failure risk.
- Certain regions of sensor space showed failure rates 2-3× higher than the baseline failure rate, indicating localized process regimes with increased risk.
- A Gaussian Mixture Model identified an elevated-risk process regime with approximately twice the baseline failure rate, further supporting the presence of probabilistic rather than deterministic failure behavior.

Overall, the evidence suggests that manufacturing failures are not driven by a single separable failure mode. Instead, failures occur across overlapping process conditions where risk increases in specific regions of the sensor space.

---

## Current Research

Current efforts are focused on building a leakage-free modeling pipeline and evaluating model performance under rigorous cross-validation.

Areas being explored include:

- Leakage-free preprocessing pipelines
- Correlation filtering and feature selection within cross-validation
- Mutual information based feature screening
- L1-regularized logistic regression
- Random Forest and XGBoost models
- PCA, PLS, and latent variable methods
- Threshold optimization for imbalanced classification
- Feature stability and model interpretability
