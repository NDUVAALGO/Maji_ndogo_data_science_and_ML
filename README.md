# Maji_ndogo_data_science_and_ML

A collection of end-to-end data science and machine learning projects set in **Maji Ndogo**, a fictional country whose Ministry of Agriculture and Department of Energy need data-driven answers. Each notebook starts from a stakeholder question and ends with a concrete, usable output.

The projects cover the full arc of applied ML: regression, classification, model selection, unsupervised learning, and recommendation systems.

## Projects

### 1. Regression: predicting numbers

| Notebook | Question | Techniques |
|----------|----------|------------|
| [`Predicting_Crop_Yield_in_Maji_Ndogo_A_Simple_Linear_Regression_Baseline.py`](./Predicting_Crop_Yield_in_Maji_Ndogo_A_Simple_Linear_Regression_Baseline.py) | How well does rainfall alone predict farm yield? | Simple linear regression, Pearson correlation, train/test split, R², MAE, RMSE, residual analysis |
| [`Multiple_Linear_Regression.py`](./Multiple_Linear_Regression.py) | Can several weather drivers explain the energy shortfall better than one? | Multiple linear regression, Variance Inflation Factor (multicollinearity), Durbin-Watson test, homoscedasticity check |
| [`Forecasting_Maji_Ndogo_s_Energy_Shortfall_with_Regularized_Regression.py`](./Forecasting_Maji_Ndogo_s_Energy_Shortfall_with_Regularized_Regression.py) | Can we build a deployable energy shortfall forecaster? | Correlation-based feature pruning, StandardScaler, Ridge (L2), LASSO (L1), pickle persistence, test-set predictions scored on RMSE |

### 2. Classification: predicting categories

| Notebook | Question | Techniques |
|----------|----------|------------|
| [`Classifying_Countries_by_Population_Size_with_Decision_Trees_A_Bias-Variance_Study.py`](./Classifying_Countries_by_Population_Size_with_Decision_Trees_A_Bias-Variance_Study.py) | Does a decision tree work before we trust it with the fields? | `DecisionTreeClassifier`, tree visualization, precision/recall/F1, bias-variance sweep over `max_depth`, pruning validation |
| [`Random_Forests_and_Logistic_Regression_Model_Stability_and_the_Precision-Recall_Trad.py`](./Random_Forests_and_Logistic_Regression_Model_Stability_and_the_Precision-Recall_Trad.py) | Is a forest more stable than a single tree, and which fields are heavily polluted? | Random Forest, feature importance, stability across 30 random seeds, logistic regression, decision-threshold tuning for the precision-recall trade-off |
| [`Crop_Classification_for_Maji_Ndogo_Multiclass_Models__Class_Imbalance_and_Per-C.py`](./Crop_Classification_for_Maji_Ndogo_Multiclass_Models__Class_Imbalance_and_Per-C.py) | What should we plant here, and how sure is the model? | Multiclass logistic regression, confusion matrix, class-imbalance upsampling, logistic regression vs Random Forest, per-crop confidence report |
| [`Digitizing_Paper_Records_Handwritten_Digit_Recognition_with_Random_Forests.py`](./Digitizing_Paper_Records_Handwritten_Digit_Recognition_with_Random_Forests.py) | Can we read handwritten digits from scanned paper survey forms? | Random Forest image classifier on 35,000 digit images, accuracy, classification report |
| [`Maji_ndogo_brew_quality_svm.py`](./Maji_ndogo_brew_quality_svm.py) | How do we tune an SVM for brew quality? | `SVC`, custom log-loss scorer, `GridSearchCV` over `C` and `gamma` |
| [`Maji_ndogo_crop_recommendation.py`](./Maji_ndogo_crop_recommendation.py) | Which model is best for recommending crops? | Data cleaning and encoding, Keras neural network, SVM, KNN, AdaBoost, Random Forest and logistic regression compared with 5-fold stratified cross-validation |

### 3. Unsupervised learning: finding structure without labels

| Notebook | Question | Techniques |
|----------|----------|------------|
| [`maji-ndogo-unsupervised-landscape.py`](./maji-ndogo-unsupervised-landscape.py) | What structure exists in 5,654 farm fields, and which are unusual? | PCA (cumulative explained variance), t-SNE, Isolation Forest |
| [`maji-ndogo-gmm-zones.py`](./maji-ndogo-gmm-zones.py) | Where are the agricultural zones, and how confident is the model? | Gaussian Mixture Models, soft cluster membership, BIC and silhouette, GeoPandas maps (`EPSG:4326`), boundary-field inspection shortlist |

### 4. Recommendation systems

| Notebook | Question | Techniques |
|----------|----------|------------|
| [`shamba-ndogo-recommender.py`](./shamba-ndogo-recommender.py) | Which crop varieties should we recommend to a farmer? | TF-IDF content-based filtering, user-based collaborative filtering with mean-centered cosine similarity, RMSE evaluation, cold-start comparison |

## Skills demonstrated

- **Regression:** simple and multiple linear regression, regularization (Ridge, LASSO), diagnostics (VIF, Durbin-Watson, residual plots), model persistence
- **Classification:** decision trees, ensembles, logistic regression, SVMs, neural networks, class imbalance, threshold tuning, model selection with cross-validation
- **Evaluation:** confusion matrices, precision/recall/F1, bias-variance analysis, RMSE and R², custom scoring functions
- **Unsupervised learning:** dimensionality reduction, soft clustering, anomaly detection, model-order selection
- **Applied ML:** geospatial visualization, image classification, recommender systems and cold-start analysis
- **Communication:** each project frames a stakeholder question first and ends with an actionable result

## Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `TensorFlow/Keras` · `statsmodels` · `SciPy` · `GeoPandas` · `matplotlib` · `seaborn` · `SQLite` · `marimo`

## Running the notebooks

The notebooks are [marimo](https://marimo.io) notebooks (plain `.py` files).

```bash
pip install marimo pandas numpy scikit-learn matplotlib seaborn statsmodels scipy geopandas tensorflow
marimo edit <notebook>.py
```

## Data

The datasets are not included in this repository. Most farm-survey notebooks read `Maji_Ndogo_farm_survey_small.db`, which is available from the [Explore AI public data repo](https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Maji_Ndogo_farm_survey_small.db). Other notebooks expect their own files (for example `world_population.csv`, `brew_quality.csv`, `df_train.csv` and `df_test.csv`) in the same folder as the notebook.

## Notes

These notebooks were completed as part of a structured data science programme. They are shared as a portfolio of my own work and are not an official solutions repository.