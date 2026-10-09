# ML Internship, Week 1: ML Fundamentals \& Basic Supervised Models

This repository contains my Week 1 work for my machine learning internship: a hands-on notebook covering core ML concepts and two basic supervised models, Linear Regression and Logistic Regression.

**Author:** Maryam ([GitHub: Maryamahsan78](https://github.com/Maryamahsan78))

## Repository contents

|File|Description|
|-|-|
|`Week1(ML-Internship).ipynb`|The full notebook with code, plots and written explanations|
|`README.md`|This file|

## What the notebook covers

1. **Exploratory data analysis (EDA)** on two datasets, with pandas and seaborn:

   * Iris (classification)
   * California Housing (regression)
2. **Gradient descent from scratch:** batch vs stochastic gradient descent, and how the learning rate affects convergence
3. **Linear Regression** on California Housing, evaluated with MSE, RMSE and R², with an actual-vs-predicted plot and residual plot
4. **Bias-variance tradeoff:** underfitting vs overfitting shown with polynomial regression, plus regularisation (Ridge)
5. **Logistic Regression** on Iris, evaluated with a confusion matrix, accuracy, precision, recall and F1, with a decision boundary plot
6. **Linear vs Logistic Regression** compared on the same binary problem

## Results

* Linear Regression (California Housing): test R² = 0.576 , RMSE = 0.746
* Logistic Regression (Iris, 3 classes): test accuracy = 0.9210526315789473

## Tools and libraries

Python, pandas, NumPy, seaborn, matplotlib, scikit-learn, Google Colab

## How to run

1. Open [Google Colab](https://colab.research.google.com) and choose **File > Upload notebook**, then select `Week1(ML-Internship).ipynb`. Alternatively, open it from GitHub through Colab.
2. Click **Connect**, then run the cells from top to bottom (**Runtime > Run all**).
3. An internet connection is needed the first time to download the California Housing dataset.

## Notes

* The Boston Housing dataset was removed from scikit-learn (version 1.2 onwards), so California Housing is used for the regression task.

