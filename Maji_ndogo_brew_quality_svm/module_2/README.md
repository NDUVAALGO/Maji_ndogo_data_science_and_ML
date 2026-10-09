# Maji Ndogo Brew Quality: SVM Hyperparameter Tuning

A classification project that builds and tunes a **Support Vector Classifier (SVC)** on chemical assay data from the fictional Maji Ndogo Brew Quality Bank. The work is a [marimo](https://marimo.io) notebook covering preprocessing, a baseline model, a custom log-loss scorer, and a cross-validated hyperparameter search with `GridSearchCV`.

## The problem

The Maji Ndogo Ministry of Agriculture wants to classify complex, multi-dimensional chemical profiles, and soil chemistry data is on the way. As a structurally similar practice problem, this notebook classifies the quality of a fermented grain brew from lab measurements, framed as a binary question: **is this batch acceptable or not?**

## Dataset

`brew_quality.csv` holds one row per brew batch.

| Feature | Description |
|---|---|
| `fixed acidity`, `volatile acidity`, `citric acid` | Acid concentrations |
| `residual sugar` | Sugar remaining after fermentation |
| `chlorides` | Salt concentration |
| `free sulfur dioxide`, `total sulfur dioxide` | SO₂ levels |
| `density`, `pH`, `sulphates`, `alcohol` | Physical and chemical readings |
| `type` | Grain base (0 = sorghum, 1 = millet) |
| `quality` | Panel score from 3 to 9 (the target) |

The target is converted to a binary label: quality of 4 or below becomes **0** (lower quality) and quality of 5 or above becomes **1** (higher quality).

## Approach

1. **Preprocessing:** binarise the target, fill missing values with zero, standardise features with `StandardScaler`, and split 75/25 (`random_state=42`).
2. **Baseline model:** an SVC with default settings, `gamma='auto'` and `random_state=40`.
3. **Custom scoring function:** a binary log-loss function with predictions clipped at `1e-15` to avoid `log(0)`.
4. **Inspecting hyperparameters:** a reusable helper that lists an estimator's tunable parameters through `get_params()`.
5. **Hyperparameter search:** `GridSearchCV` with 5-fold cross-validation over `C` in `[0.1, 1, 10]` and `gamma` in `[0.01, 0.1, 1]`, scored with the custom log-loss through `make_scorer(greater_is_better=False)`.
6. **Extracting the best parameters:** a helper that returns the winning parameters as a dictionary.

## Results

| Model | Log-loss | Accuracy |
|---|---|---|
| Baseline SVC (`gamma='auto'`) | 1.0202 | 97.05% |
| Tuned SVC (`C=10`, `gamma=0.01`) | 0.9565 | 97.23% |

Tuning the hyperparameters gave a modest improvement in both metrics without changing the model type.

## Tech stack

Python, marimo, pandas, NumPy, scikit-learn, Matplotlib, Seaborn.

## Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install marimo pandas numpy matplotlib seaborn scikit-learn
marimo edit notebook.py
```

Place `brew_quality.csv` in the same folder as `notebook.py` before running it.

## Project structure

```
.
├── notebook.py        # marimo notebook with all six challenges
├── brew_quality.csv   # dataset (add this file yourself)
└── README.md
```

## Credits

The challenge and dataset come from the Explore AI data science programme.