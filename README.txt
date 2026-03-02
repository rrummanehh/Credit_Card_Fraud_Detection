Credit Card Fraud Detection

Overview

This project applies unsupervised machine learning to detect fraudulent credit card transactions. Since the features are the result of PCA (anonymized), the focus is on clustering and anomaly detection rather than feature interpretation. The project covers data cleaning, feature engineering, skewness correction, scaling, dimensionality reduction, clustering, and anomaly detection — both on the original imbalanced dataset and on a balanced version using SMOTE and undersampling.

This was a group project completed during second year in university.
Group members: Tareq M. Hab-El-Rumman, Gina R. Maayah.
---------------------------------------

Dataset

The dataset contains credit card transactions made by European cardholders in September 2013. It is highly imbalanced — fraud cases make up only about 0.17% of all transactions. The features V1–V28 are the result of PCA transformation and cannot be directly interpreted. The only non-PCA features are Time, Amount, and Class (the target).

The dataset file (creditcard.csv) is approximately 150MB and is too large to include in this repository.
Download it from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
And Place it in the same folder as the notebook before running (The one you downloaded from the repository).
---------------------------------------

What We Did

1. Data Cleaning

. Verified no missing values
. Removed duplicate legitimate transactions (kept all fraud rows to preserve fraud patterns)
. Dropping all duplicates would have caused bias in density-based models like DBSCAN

2. Feature Engineering

. Extracted Hour from the Time column (hour of day the transaction occurred)
. Dropped the original Time column after extraction
. Visualized fraud transactions by hour of day to identify high-risk time windows

3. Skewness Correction

. Applied PowerTransformer (Yeo-Johnson method) to Amount, Hour, and selected V features
. Visualized distributions after transformation using KDE plots

4. Splitting and Scaling

. Split data before scaling to prevent data leakage
. Applied StandardScaler to all V features (already PCA-centered)
. Applied RobustScaler to Amount and Hour (more robust to outliers)

5. Dimensionality Reduction

. PCA retaining 95% variance — did not reduce dimensions significantly due to the nature of PCA-transformed features
. 2D PCA — used for visualization before and after scaling
. T-SNE — applied on a 5,000-sample subset for memory efficiency; fraud points (red) were clearly isolated from normal transactions (blue)

6. Clustering (on imbalanced dataset)

Three clustering algorithms were compared across K values from 2 to 20:

. KMeans (random init) — Best K: 11
. KMeans++ — Best K: 10
. MiniBatchKMeans — Best K: 3

Best K was selected based on the highest silhouette score.
DBSCAN was also applied with a grid search over eps and min_samples.

7. Anomaly Detection (on imbalanced dataset)

. Isolation Forest (contamination=0.02)
. One-Class SVM (trained only on legitimate transactions)

One-Class SVM outperformed Isolation Forest on this dataset.

8. Balanced Dataset (SMOTE + Undersampling)

. Applied SMOTE to oversample the minority (fraud) class
. Applied RandomUnderSampler to reduce the majority (legitimate) class
. Re-ran the full clustering and anomaly detection pipeline on the balanced dataset
. On the balanced dataset, best K dropped to 2 for all three KMeans variants, reflecting the cleaner class separation
---------------------------------------

A Note on Why We Used Undersampling for Anomaly Detection

For clustering, oversampling or undersampling is not needed — fraud is supposed to be rare, and altering that ratio would distort distance and density calculations. However, for anomaly detection models like Isolation Forest and One-Class SVM, working on a balanced subset helps avoid the model predicting "legit" for everything and achieving misleadingly high accuracy while completely missing fraud cases.
Results
---------------------------------------

Results

Anomaly Detection — Imbalanced Dataset:

 Model                    Precision    Recall    F1 Score
. Isolation Forest        moderate     moderate  moderate
. One-Class SVM           higher       higher    higher

One-Class SVM was the better choice for anomaly detection on this dataset.

Clustering — Balanced Dataset:

 Model                Best K
. KMeans (random)     2
. KMeans++            2
. MiniBatchKMeans     2

KMeans++ produced the tightest and best-positioned clusters. MiniBatchKMeans was faster but slightly weaker in performance.
Visualizations
---------------------------------------

. Fraud transactions by hour of day (bar chart)

. Correlation heatmap of all features with the target (Class)

. KDE plots after skewness correction

. Class distribution (fraud vs. normal)

. PCA explained variance plot

. 2D PCA scatter plots before and after scaling (fraud highlighted in red)

. T-SNE scatter plots (fraud in red, normal in blue)

. Elbow curves (inertia vs. K) for all three KMeans variants

. Silhouette score vs. K for all three KMeans variants

. Silhouette plots for selected K values

. DBSCAN clustering plot (grey points = outliers)

. Side-by-side T-SNE comparison of all three clustering methods

. Confusion matrices for Isolation Forest and One-Class SVM

. All of the above repeated on the balanced dataset