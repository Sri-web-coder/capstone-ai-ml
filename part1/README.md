# Part 1: Data Acquisition, Cleaning, and Exploratory Analysis

## Project Summary

This Part of the project focuses on cleaning and exploring the Student Mental Health dataset before building machine learning models. The dataset was cleaned by handling missing values, checking duplicates, converting data types, detecting outliers, and creating visualizations. The cleaned dataset was then saved for the next stages of the project.

## Dataset

- Dataset: Student Mental Health Dataset
- Rows: 2000
- Columns: 17
- Includes both numerical and categorical features related to students' academics, lifestyle, and mental health.
- I sourced the student_mental_health.csv dataset from Kaggle for this project.

## Setup

Install the required libraries:

"pip install pandas numpy matplotlib seaborn"

No ".env" file is required.

## Run Commands

To reproduce the results using Jupyter Notebook:

1. Open the Jupyter Notebook interface.
2. Open the Part 1 notebook (`.ipynb`).
3. Run all cells sequentially from top to bottom.

## Methodology

- Loaded the dataset using Pandas.
- Checked data types, missing values, and duplicates.
- Filled numerical missing values using the median and categorical values using the mode.
- Converted repeated string columns to the category data type to reduce memory usage.
- Detected outliers using the IQR method.
- Performed descriptive statistics, skewness analysis, and correlation analysis.
- Created line plot, bar chart, histogram, scatter plot, box plot, and correlation heatmap.
- Saved the cleaned dataset as "cleaned_data.csv".

## Key Findings

- No duplicate records were found.
- All missing values were handled successfully.
- Memory usage decreased after converting categorical columns.
- Family_Support showed the highest skewness, so the median was used for numerical imputation.
- Only a few outliers were detected and retained for future modelling.
- A positive relationship was observed between Social_Media_Hours and Screen_Time_Hours.
- Students with Low Mental Health Risk generally had higher average sleep hours.
- Pearson and Spearman correlations were similar, indicating mostly linear relationships.

## Design Decisions

- Used median instead of mean for numerical missing values because it is more robust to skewed data and outliers.
- Retained outliers since they may represent genuine student behaviour.
- Used the category data type to improve memory efficiency.

## Output Files

- "cleaned_data.csv"
- "line_plot.png"
- "bar_chart.png"
- "histogram.png"
- "scatter_plot.png"
- "box_plot.png"
- "correlation_heatmap.png"


