# ML Internship, Week 3: Unsupervised Learning \& Hyperparameter Tuning

Four notebooks covering customer segmentation with clustering and hyperparameter tuning of classifiers.

**Author:** Maryam ([GitHub: Maryamahsan78](https://github.com/Maryamahsan78))

## Notebooks

|File|Topic|Dataset|
|-|-|-|
|`kmeans.ipynb`|K-Means, Elbow method, Silhouette score, cluster profiling, why scaling matters|Mall Customers|
|`hierarchical\_pca.ipynb`|Hierarchical clustering (dendrogram, linkage), K-Means vs hierarchical, four-feature clustering with PCA visualisation|Mall Customers|
|`gridsearch.ipynb`|GridSearchCV on kNN, Decision Tree and SVM, default vs tuned scores, tuning for malignant recall|Breast Cancer|
|`randomsearch.ipynb`|RandomizedSearchCV, grid vs random, tuning plots, comparison table, reading tuning results honestly|Breast Cancer|

## Part 1: Clustering (Monday and Tuesday notebooks)

* **Preprocessing:** scaled the features, because K-Means and hierarchical clustering are distance based
* **Choosing k:** Elbow method and Silhouette score
* **Algorithms:** K-Means (including a from-scratch version showing the assign-and-move loop) and hierarchical (agglomerative) clustering with different linkages
* **Visualisation:** cluster plots with centroids, a dendrogram with the cut line, and PCA projections of clusters built on four features
* **Interpretation:** profiled each cluster (mean Age, Income, Spending Score) and named the customer segments

### Results

**K-Means on Annual Income and Spending Score (scaled)**

* Chosen k: **5**. The Elbow method flattens after k = 5 (inertia drops 43.4 going from 4 to 5 clusters, then only 10.5 from 5 to 6), and the Silhouette score is highest at k = 5.
* Silhouette score of the final model: **0.555** (k = 6 was close at 0.540).
* Customer segments found (cluster sizes in brackets):

  * **High income, high spending** (39): mean age 33, income 86.5k$, spending score 82
  * **High income, low spending** (35): mean age 41, income 88.2k$, spending score 17
  * **Low income, high spending** (22): mean age 25, income 25.7k$, spending score 79
  * **Low income, low spending** (23): mean age 45, income 26.3k$, spending score 21
  * **Middle income, middle spending** (81): mean age 43, income 55.3k$, spending score 50

**Hierarchical clustering (same two features, k = 5)**

* The dendrogram's biggest jump in merge distance (from 4.35 to 9.45) comes right before 5 clusters.
* Silhouette by linkage: ward 0.554, complete 0.553, average 0.479, single 0.276 (single linkage chained one cluster of 195 customers).
* K-Means vs hierarchical (ward) agreement, Adjusted Rand Index: **0.942** (only about 4 customers were grouped differently).

**Four features (Age, Income, Spending Score, Gender) with PCA**

* K-Means on four features: the best silhouette was at k = 10, with a lower score of **0.421**, so the clusters are less well separated than with two features.
* The first 2 principal components keep **59.9 %** of the variance (PC1 33.7 %, PC2 26.2 %).
* PC1 is driven by Age (+0.69) and Spending Score (-0.69); PC2 by Income (+0.77) and Gender (+0.63).

## Part 2: Hyperparameter tuning (Wednesday and Thursday notebooks)

* **Search methods:** `GridSearchCV` and `RandomizedSearchCV`, with cross-validation inside a scikit-learn `Pipeline` (scaler plus model)
* **Models tuned:** kNN, Decision Tree, SVM, Random Forest
* **Honest evaluation:** a held-out test set that the searches never saw, compared against default settings
* **Visualisation:** a heatmap of CV accuracy for the SVM (C vs gamma), the random-search trials, and a kNN tuning curve with the fold-to-fold spread
* **Extras:** tuning for malignant recall instead of accuracy, an edge-of-grid check, and a shuffled-label demo of why `best\_score\_` is optimistic

### Results

Breast Cancer data, 426 training rows and 143 held-out test rows, 5-fold CV. One test row is about 0.007 accuracy.

|Model|Search|Default CV|Tuned CV|Default test|Tuned test|Best parameters|
|-|-|-|-|-|-|-|
|kNN|Grid|0.967|0.972|0.979|0.965|`n\_neighbors=10`, `weights=uniform`, `metric=euclidean`|
|Decision Tree|Grid|0.925|0.941|0.923|0.895|`max\_depth=6`, `min\_samples\_leaf=2`, `criterion=gini`|
|SVM (rbf)|Grid|0.967|0.974|0.979|0.979|`C=10`, `gamma=0.001`|
|SVM (rbf)|Random|0.967|0.972|0.979|0.979|`C=10.32`, `gamma=0.0151`|
|Random Forest|Random|0.962|0.958|0.958|0.958|`n\_estimators=245`, `max\_depth=12`, `max\_features=log2`, `min\_samples\_split=5`, `min\_samples\_leaf=1`|

Extra results:

* **Grid vs random (SVM):** the 30-combination grid took 3.06 s and reached CV 0.9742. Random search with 10 trials took 0.46 s and reached 0.9719. Across 8 different seeds, random search scored 0.9695 to 0.9766, so it matched the grid with a third of the fits.
* **Random Forest:** the space has 1,440 combinations (7,200 fits). Random search tried 40 (200 fits, 45.9 s) and did not beat the default forest.
* **Tuning for malignant recall (kNN):** the search chose `n\_neighbors=2`. Malignant recall on the test set went from 0.943 (default) and 0.925 (tuned for accuracy) to 0.962, and missed malignant tumours dropped from 3 and 4 to 2. The price was lower accuracy (0.937).
* **Edge check:** all best values were inside the tried range, except `min\_samples\_leaf=1` for the forest, which is the smallest allowed value anyway.
* **Shuffled-label demo:** with random labels (majority-class rate 0.627), the best of 120 settings scored 0.629 while the average setting scored 0.579. The best score from a search is optimistic.

### Explanation of the tuning results

I tuned kNN, a Decision Tree, an SVM and a Random Forest with GridSearchCV and RandomizedSearchCV, always inside a Pipeline with a scaler. Tuning raised the cross-validation score a little for kNN (0.967 to 0.972), the Decision Tree (0.925 to 0.941) and the SVM (0.967 to 0.974), but the Random Forest's CV score did not improve (0.962 to 0.958). On the held-out test set, the SVM and the forest stayed the same, while kNN fell by 0.014 and the Decision Tree by 0.028. That is 2 and 4 test rows, which is within the noise of a 143-row test set. The reason is that `best\_score\_` is chosen as the best out of many tries on the same folds, so it is slightly optimistic. The dataset is also easy, because the default models already scored 0.96 to 0.98. So tuning was not worth much extra compute here, apart from random search, which found a good SVM about 6 times faster than the grid. Tuning for malignant recall did change the result in a useful way, because the metric you optimise decides what "best" means.

## Key takeaways

* Tuning gives small gains on an easy dataset, and a gain is only real if it is bigger than the noise (one test row is about 0.007, CV std is about 0.005 to 0.02).
* Judge a tuned model on a held-out test set, not on `best\_score\_`, which is optimistic (the shuffled-label demo shows this).
* RandomizedSearchCV reaches almost the same score as GridSearchCV with far fewer fits, and it matters most when a model has many hyperparameters.
* The `scoring` metric decides what "best" means: tuning kNN for malignant recall found fewer missed tumours at the cost of some accuracy.

## Tools

Python, pandas, NumPy, seaborn, matplotlib, scikit-learn, SciPy, Google Colab

## How to run

1. Open a notebook in [Google Colab](https://colab.research.google.com) (File > Upload notebook, or open it from GitHub) and click **Connect**.
2. The two clustering notebooks need `Mall\_Customers.csv`: upload it with the folder icon on the left. Without it, they fall back to made-up practice data and print a warning.
3. The two tuning notebooks use a built-in scikit-learn dataset, so nothing needs uploading.
4. Run **Runtime > Run all**.

