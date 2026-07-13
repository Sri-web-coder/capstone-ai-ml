# Part 3: Advanced Modeling – Ensembles, Tuning, and Full ML Pipeline

## Project Summary

This part of the project focuses on building and comparing advanced machine learning models using the cleaned dataset from Part 2. Different ensemble techniques, hyperparameter tuning, cross-validation, feature importance analysis, learning curves, and model serialization were performed to identify the most reliable classification model for deployment.

## Dataset

- Dataset: Cleaned Student Mental Health Dataset
- Input file: `cleaned_data.csv`
- Training and testing datasets from Part 2 were reused.
- Classification target: **Mental_Health_Risk**
- Models were trained using the preprocessed features generated in Part 2.

## Setup

Install the required libraries:

pip install pandas numpy scikit-learn matplotlib joblib

No `.env` file is required.

## Run Commands

To reproduce the results using Jupyter Notebook:

1. Open the Jupyter Notebook interface.
2. Open the Part 3 notebook (`.ipynb`).
3. Run all cells sequentially from top to bottom.

## Methodology

- Loaded the processed training and testing datasets from Part 2.
- Built an unconstrained Decision Tree classifier.
- Built a controlled Decision Tree using `max_depth=5` and `min_samples_split=20`.
- Compared Gini and Entropy splitting criteria.
- Trained a Random Forest classifier and analysed feature importance.
- Trained a Gradient Boosting classifier.
- Performed a feature ablation study by removing the five least important features.
- Compared multiple models using 5-fold Stratified Cross Validation.
- Tuned Random Forest hyperparameters using GridSearchCV.
- Generated a manual learning curve using different training sizes.
- Saved the best-performing pipeline as `best_model.pkl`.
- Reloaded the saved model and verified predictions.

## Key Findings

- The baseline Decision Tree achieved **100% training accuracy** but only **72.00% test accuracy**, indicating overfitting.
- The controlled Decision Tree improved generalization with **82.50% training accuracy** and **74.75% test accuracy**.
- The Gini criterion slightly outperformed Entropy on the test dataset.
- Random Forest achieved **78.25% test accuracy** with a **ROC-AUC of 0.8968**.
- Gradient Boosting produced the highest test accuracy (**81.25%**) with a **ROC-AUC of 0.9012**.
- The five most important features were:
  - Sleep_Hours
  - Financial_Stress
  - Study_Hours_Per_Week
  - Family_Support
  - Attendance_Rate
- Removing the five least important features caused only a very small decrease in ROC-AUC (0.8968 → 0.8922), indicating they contributed very little to prediction.
- Logistic Regression achieved the highest average cross-validation ROC-AUC (0.9364).
- GridSearchCV selected the best Random Forest model with:
  - `n_estimators = 200`
  - `max_depth = None`
  - `min_samples_leaf = 5`
- The learning curve showed that test performance improved as more training data became available before stabilizing.
- The best trained model was successfully saved and reloaded without errors.

## Design Decisions

- A constrained Decision Tree was used to reduce overfitting.
- Gini and Entropy were compared to evaluate different impurity measures.
- Random Forest was selected because ensemble learning reduces variance compared to a single decision tree.
- Feature importance was used to identify the most influential variables.
- Feature ablation was performed to evaluate whether removing less useful features affected model performance.
- Stratified Cross Validation was used to obtain a more reliable estimate of model performance.
- GridSearchCV was applied to systematically identify the best Random Forest hyperparameters.
- The final model was serialized using Joblib for future deployment.

## Model Performance

| Model | Test Accuracy | ROC-AUC |
|--------|--------------:|--------:|
| Decision Tree (Baseline) | 0.7200 | - |
| Controlled Decision Tree | 0.7475 | - |
| Random Forest | 0.7825 | 0.8968 |
| Gradient Boosting | 0.8125 | 0.9012 |

## Cross-Validation Results

| Model | Mean ROC-AUC | Standard Deviation |
|--------|-------------:|-------------------:|
| Logistic Regression | 0.9364 | 0.0134 |
| Controlled Decision Tree | 0.8128 | 0.0275 |
| Random Forest | 0.9112 | 0.0101 |
| Gradient Boosting | 0.9214 | 0.0097 |

## Hyperparameter Tuning

**Best Parameters**

- `n_estimators = 200`
- `max_depth = None`
- `min_samples_leaf = 5`

**Best Cross-Validation Score**

- **ROC-AUC = 0.9051**

A total of **18 parameter combinations** were evaluated using **5-fold cross-validation**, resulting in **90 model evaluations**.

## Learning Curve Summary

The training AUC decreased slightly as more training samples were used, while the test AUC improved and then stabilized around 0.91. This indicates that adding more data improved generalization initially, and the model is approaching its performance limit.

## Model Recommendation

Among all evaluated models, **Logistic Regression** achieved the highest average cross-validation ROC-AUC, while **Gradient Boosting** achieved the highest test accuracy. Considering overall stability, cross-validation performance, and generalization, Logistic Regression remains the recommended model for this dataset. It provides strong predictive performance while remaining simple, interpretable, and computationally efficient.

## Output Files

- `best_model.pkl`

