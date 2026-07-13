# Part 2: Supervised Machine Learning Model — Build, Train, and Evaluate

## Project Summary

This part of the capstone builds on the cleaned dataset from Part 1 to create predictive machine learning models. I prepared the data by defining target variables, encoding categorical features, and splitting the data safely to avoid leakage. Then, I trained a regression model to predict student GPA and a classification model to predict Mental Health Risk. Finally, I evaluated these models using various metrics, tested different decision thresholds, and ran a regularization experiment to see how it affected performance.

## Dataset
- **Input Data:** `cleaned_data.csv` (generated from Part 1).
- **Target Definitions:**
  - `y_reg` (Regression Label): `GPA` (Continuous numerical column).
  - `y_clf` (Classification Label): `Mental_Health_Risk` (Binarized using a map: High = 1, Low/Medium = 0).

## Setup

Install the required libraries:
`pip install pandas numpy matplotlib seaborn scikit-learn joblib`
No `.env` file is required.

## Run Commands

To run the code and reproduce my analysis using Jupyter Notebook:
1. Open the Jupyter Notebook interface.
2. Open the `.ipynb` file containing this project.
3. Run the cells sequentially from the top to ensure all tasks and visualizations are executed in order.

## Methodology

- Loaded `cleaned_data.csv` and separated the features (X) and targets (`y_reg` and `y_clf`).
- Applied Label Encoding for ordinal categories (Physical Activity, Diet Quality, Risk) and One-Hot Encoding for nominal columns (Gender, Job, History).
- Split the dataset into 80% training and 20% testing sets.
- Scaled features using `StandardScaler`.
- Trained Linear Regression and Ridge Regression for GPA prediction.
- Trained Logistic Regression for Mental Health Risk prediction and plotted the ROC curve.
- Tested how different decision thresholds impact Precision, Recall, and F1-score.
- Conducted a regularization experiment (C=0.01) and used Bootstrapping (500 samples) to verify if the performance difference was statistically significant.
- Saved the final scaled arrays and classification test sets as `.pkl` files.

## Key Findings & Interpretations

**1. Data Leakage Prevention:**

I fitted the `StandardScaler` *only* on the training data. Fitting it on the entire dataset would cause data leakage because the scaler would calculate the mean and variance using test set data, effectively allowing the model to "peek" at the test data during training.

**2. Encoding Choices:**

I used Label Encoding for ordinal features (like Diet: Poor < Average < Good) to preserve their natural rank. For nominal features (like Gender), I used One-Hot Encoding and dropped the first column. This avoids the false-ordinal-relationship problem, as assigning 1, 2, 3 to nominal categories would trick the model into thinking one category is "greater" than another.

**3. Regression Results (GPA Prediction):**

The top 3 features with the largest absolute coefficients were:
1. Study_Hours_Per_Week (0.158)
2. Sleep_Hours (0.068)
3. Attendance_Rate (0.068)
*Interpretation:* A large positive coefficient means that as the feature increases, the predicted GPA increases. For example, a higher scaled value of study hours strongly pushes the predicted GPA up. A negative coefficient would mean the opposite.

**4. Ridge vs. OLS Linear Regression Comparison:**

| Model | MSE | R² |
| :--- | :--- | :--- |
| Linear Regression | 0.0874 | 0.2936 |
| Ridge Regression (alpha=1.0)| 0.0874 | 0.2936 |
*Explanation:* Ridge introduces an L2 penalty to shrink coefficients and prevent overfitting, controlled by the `alpha` parameter (higher alpha = stronger penalty). Here, the dataset is simple enough that both models performed identically, meaning plain OLS wasn't severely overfitting.

**5. Classification Metrics (Mental Health Risk):**

- **Precision Formula:** TP / (TP + FP)
- **Recall Formula:** TP / (TP + FN)
- **Important Metric:** For this specific task, **Recall** is more important. Missing a student who actually has a high mental health risk (False Negative) is much more costly than falsely flagging a healthy student (False Positive).
- **AUC Meaning:** The model achieved an AUC of 0.9219. This means there is a 92.19% chance that the model will correctly rank a randomly chosen high-risk student higher than a randomly chosen low-risk student. It indicates excellent separation between the classes.

**6. Decision-Threshold Sensitivity:**

| Threshold | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- |
| 0.30 | 0.7118 | 0.8462 | 0.7732 |
| 0.40 | 0.7651 | 0.7972 | 0.7808 |
| 0.50 | 0.8030 | 0.7413 | 0.7709 |
| 0.60 | 0.8261 | 0.6643 | 0.7364 |
| 0.70 | 0.8710 | 0.5664 | 0.6864 |

*Conclusion:* The threshold that maximizes the F1-score is **0.40**. However, to optimize for Recall (our most important metric), I would **lower** the threshold (e.g., to 0.30). The cost of doing this is a drop in Precision—meaning we will have more false alarms where we investigate students who are actually fine.

**7. Regularization Experiment (C Parameter):**

In Logistic Regression, the `C` parameter controls the inverse of regularization strength (smaller `C` means stronger penalty). Reducing `C` to 0.01 slightly worsened the AUC performance on this dataset (from 0.9219 to 0.9182), showing that heavily penalizing the model wasn't necessary.

**8. Bootstrap Confidence Interval:**

- Mean AUC Difference: 0.0036
- 95% Confidence Interval: [-0.0003, 0.0079]
*Interpretation:* Because the 95% confidence interval **includes zero**, the performance difference between the standard model and the strongly regularized model may not be statistically reliable across different data samples.

## Output Files

- `X_train_scaled.pkl`
- `X_test_scaled.pkl`
- `y_clf_train.pkl`
- `y_clf_test.pkl`
