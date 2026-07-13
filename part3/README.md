

# Part 3: Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline

## Project Summary

In this part of the project, I moved beyond basic models to build, evaluate, and tune advanced ensemble models. I trained Decision Trees, Random Forests, and Gradient Boosting models, and compared them against our baseline Logistic Regression from Part 2. I also explored feature importance, conducted an ablation study to see what happens when we drop weak features, built a reproducible Machine Learning Pipeline, and tuned hyperparameters using GridSearchCV. And I saved the best performing pipeline for future deployment.

## Setup & Run Commands

To reproduce the results using Jupyter Notebook:

1. Open the Jupyter Notebook interface.
2. Open the Part 2 notebook (`.ipynb`).
3. Run all cells sequentially from top to bottom.

## Methodology & Key Findings

**1. Decision Tree Baseline vs. Controlled Tree:**

* **Baseline:** The unconstrained Decision Tree heavily overfit the data (Training Accuracy: 1.0000, Test Accuracy: 0.7200). Decision trees are "high-variance" models because they greedily create rules to perfectly fit the training data at every split, which means they memorize the noise instead of learning the underlying pattern.
* **Controlled Tree:** By setting `max_depth=5` and `min_samples_split=20`, the gap closed significantly (Train: 0.8250, Test: 0.7475). `max_depth` limits how deep the tree can grow (reducing variance/overfitting), while `min_samples_split` prevents the tree from creating highly specific rules for tiny, noisy groups of data.

**2. Gini vs. Entropy:**

* **Formulas:** * Gini Impurity = 1 - Σ(p_i)²
    * Entropy = -Σ(p_i * log2(p_i))
* Both methods measure the "impurity" of a node. If a node has a **Gini = 0**, it means the node is perfectly pure—all the samples in that node belong to exactly one class. Both criteria performed very similarly on this dataset.

**3. Random Forest & Bagging:**

* **Top 5 Features:** Sleep_Hours, Financial_Stress, Study_Hours_Per_Week, Family_Support, and Attendance_Rate.
* **Feature Importance vs. Linear Coefficients:** Random Forest calculates feature importance by looking at how much a feature reduces Gini impurity across all the splits in all the trees. This is different from a linear regression coefficient, which represents the direct mathematical slope (how much Y changes for a one-unit change in X).
* **Bagging Explained:** "Bagging" (Bootstrap Aggregating) means building multiple decision trees using random samples of the training data (with replacement). Also, at each split, the tree is only allowed to look at a random subset of features. By averaging the predictions of all these uniquely flawed trees, the ensemble smooths out mistakes, significantly reducing the high variance seen in a single deep decision tree.

**4. Feature Ablation Study (Production Trade-offs):**

I removed the 5 least important features (mostly gender and job-related columns) and retrained the Random Forest. The ROC-AUC dropped slightly from 0.8968 to 0.8922. Because the score dropped, these features weren't entirely uninformative—they were contributing a little bit. However, in a real production environment, a tiny drop in AUC might be totally acceptable if it means deploying a simpler, faster model that costs less to run and maintain.

**5. Cross-Validation:**

I used 5-fold cross-validation instead of relying solely on our Train/Test split. Cross-validation is much more reliable because it tests the model on 5 different, rotating slices of the data. This proves the model's performance is consistent and didn't just get "lucky" with one specific random split.

**6. Hyperparameter Tuning (GridSearchCV):**

* I evaluated a total of **90 model configurations** (3 depths × 3 estimator counts × 2 leaf samples = 18 configurations, multiplied by 5 cross-validation folds).

* **Grid Search vs. Randomized Search:** Grid Search exhaustively tries *every single combination* in the parameter grid. It guarantees you find the best combination you asked for, but it is very slow. Randomized Search only tries a random selection of combinations, which is much faster but might miss the absolute perfect setup.

**7. Manual Learning Curve:**

| Training Fraction | Training AUC | Test AUC |
| :--- | :--- | :--- |
| 0.2 | 0.9907 | 0.8907 |
| 0.4 | 0.9892 | 0.9018 |
| 0.6 | 0.9885 | 0.9075 |
| 0.8 | 0.9887 | 0.9136 |
| 1.0 | 0.9873 | 0.9123 |

*Interpretation:* As expected with high-variance models, the Training AUC slowly decreased as the dataset grew (it's harder to perfectly memorize more data). The Test AUC generally increased but completely plateaued around 80%-100% of the data. My conclusion is that this model is currently limited by **model capacity**, not data quantity. Feeding it more data probably won't improve the score much further.

## Summary Comparison Table & Recommendation

| Model | 5-Fold CV Mean AUC | 5-Fold CV Std AUC | Test-Set AUC |
| :--- | :--- | :--- | :--- |
| Logistic Regression (Part 2) | **0.9364** | 0.0134 | **0.9219** |
| Controlled Decision Tree | 0.8128 | 0.0275 | ~0.7475 (Accuracy) |
| Random Forest | 0.9112 | 0.0101 | 0.8968 |
| Gradient Boosting | 0.9214 | 0.0097 | 0.9012 |

**Final Recommendation:**

I recommend deploying the **Logistic Regression** model to the client. Despite being the simplest model, it achieved the highest Mean AUC during cross-validation (0.9364) and performed the best on the unseen test set (0.9219). Complex ensemble models like Random Forest and Gradient Boosting did not provide any performance boost on this specific dataset, making the Logistic Regression the most robust, interpretable, and cost-effective choice for production.

## Output Files

- `best_model.pkl` 
