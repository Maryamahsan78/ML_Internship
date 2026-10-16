# ML Internship, Week 2: Core Classification Models \& Data Preprocessing

Hands-on notebooks comparing three classification algorithms inside a reusable preprocessing pipeline.

**Author:** Maryam ([GitHub: Maryamahsan78](https://github.com/Maryamahsan78))

## Notebooks

|File|Topic|
|-|-|
|`preprocessing.ipynb`|Missing data, encoding, scaling, train/test split, cross-validation (Titanic)|
|`trees\\\_knn.ipynb`|Decision Trees and k-Nearest Neighbors (Wine, Breast Cancer)|
|`naive\\\_bayes\\\_pipelines.ipynb`|Naive Bayes, Pipeline and ColumnTransformer|
|`comparison.ipynb`|Feature selection vs PCA, and the full 3-model comparison with confusion matrices|

## What this covers

* **Preprocessing:** imputing missing values, one-hot encoding, feature scaling, all inside a `Pipeline` so nothing leaks from the test data
* **Validation:** train/test split and stratified 5-fold cross-validation
* **Models:** Decision Tree, k-Nearest Neighbors, Gaussian Naive Bayes
* **Feature selection vs dimensionality reduction:** `SelectKBest` and PCA on the Wine dataset
* **Evaluation:** accuracy, precision, recall, F1, and confusion matrices for every model on every dataset

## Datasets

Titanic (seaborn), Wine and Breast Cancer (scikit-learn)

## Results (5-fold CV accuracy, fill in from your own run)

|Model|Titanic|Wine|Breast Cancer|
|-|-|-|-|
|Decision Tree|0.818 ± 0.016|0.888 ± 0.031|0.923 ± 0.030|
|kNN (k=5)|0.813 ± 0.018|0.972 ± 0.018|0.963 ± 0.018|
|Naive Bayes|0.787 ± 0.017|0.977 ± 0.028|0.930 ± 0.020|

## Summary

I compared a Decision Tree (max depth 4), k-Nearest Neighbors (k=5) and Gaussian Naive Bayes, each inside the same pipeline (imputation, one-hot encoding, scaling), on Titanic, Wine and Breast Cancer using 5-fold cross-validation

**- Titanic:** Decision Tree (0.818) and kNN (0.813) performed almost equally, with a difference smaller than the fold-to-fold variation. Naive Bayes scored 0.787. All three models outperformed the majority-class baseline of 0.616.

**- Wine:** Naive Bayes (0.977) and kNN (0.972) were effectively tied and performed better than the Decision Tree (0.888). The performance gap was larger than the fold-to-fold variation.

**- Breast Cancer:** kNN achieved the highest accuracy (0.963), followed by Naive Bayes (0.930) and Decision Tree (0.923). Naive Bayes and Decision Tree performed relatively similarly.

Most important metric: For Breast Cancer, recall for the malignant class is especially important because failing to identify a malignant tumour can have serious consequences. On the held-out test set, kNN missed 3 malignant cases, Decision Tree missed 4, and Naive Bayes missed 6.

## **Tools**

Python, pandas, NumPy, seaborn, matplotlib, scikit-learn, Google Collab

## How to run

Open each notebook in [Google Colab](https://colab.research.google.com) (File > Upload notebook), click **Connect**, then **Runtime > Run all**. An internet connection is needed to load the Titanic dataset.

