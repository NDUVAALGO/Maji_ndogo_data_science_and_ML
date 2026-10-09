# Maji Ndogo Crop Yield Regression (Part A)

A first supervised machine learning project: predicting crop yield on the fictional **Maji Ndogo** farm survey using simple linear regression. The work is a [marimo](https://marimo.io) notebook that moves from a single scatter plot through to model evaluation and residual diagnostics, and it sets an honest baseline for the multiple-regression work that follows.

## The question

The Maji Ndogo Ministry of Agriculture wants to forecast crop yield so farmers can plan ahead and investment can go to the factors that matter most. Part A asks the simplest version of that question: **how much of the yield can a single feature explain?**

## Dataset

The data is the Maji Ndogo farm survey, a SQLite database where each row is one field. The notebook joins four tables on `Field_ID`: geographic, weather, soil and crop, and farm management features.

| Group | Features |
|---|---|
| Target | `Standard_yield` (normalised yield, 0 to 1) |
| Environmental | `Rainfall`, `Ave_temps`, `Elevation`, `Slope` |
| Soil | `Soil_fertility`, `pH`, `Pollution_level` |
| Farm | `Plot_size`, `Crop_type`, `Location`, `Annual_yield` |

Cleaning steps applied in the notebook: fixing swapped `Annual_yield` and `Crop_type` column names, taking the absolute value of `Elevation`, correcting misspelled crop names (`cassaval`, `wheatn`, `teaa`), and dropping rows with missing values.

## What the notebook covers

1. **Visualising a relationship:** scatter plot of `Ave_temps` against `Standard_yield`, with the Pearson correlation coefficient.
2. **Fitting a line:** least-squares regression of yield on `Rainfall` with scikit-learn, returning the slope and intercept.
3. **Evaluating the model:** R², MAE, MSE and RMSE on the full dataset.
4. **Train-test split:** an 80/20 split (`random_state=42`) to check the model generalises to unseen fields.
5. **Residual analysis:** histogram of residuals, plus their mean and standard deviation, to check the model's assumptions.

## Results

| Step | Result |
|---|---|
| Correlation, temperature vs yield | 0.0068 |
| Rainfall model, slope / intercept | 8.77e-06 / 0.5239 |
| Full dataset, R² / MAE / MSE / RMSE | 0.0015 / 0.0877 / 0.0125 / 0.1117 |
| Test set, R² / MAE / MSE / RMSE | 0.00005 / 0.0900 / 0.0133 / 0.1154 |
| Test residuals, mean / std | 0.0072 / 0.1152 |

**Takeaway:** neither temperature nor rainfall explains much of the variation in yield on its own. Rainfall has a weak positive slope, but R² is close to zero. Train and test performance are similar, so the model is consistently weak rather than overfitted, and its residuals show no strong bias. A single predictor is not enough, which is the motivation for adding more features in Part B.

## Tech stack

Python, marimo, pandas, NumPy, SciPy, scikit-learn, Matplotlib, SQLite.

## Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install marimo pandas numpy matplotlib scipy scikit-learn
marimo edit notebook.py
```


## Roadmap

| Part | Scope | Status |
|---|---|---|
| **A** | Single-predictor simple linear regression | Complete |
| **B** | Multiple linear regression with feature selection | Upcoming |
| **C** | Regularisation (Ridge and LASSO) and model persistence | Upcoming |

## Credits

The challenge and data come from the Explore AI data science programme. The dataset is from the [Explore-AI Public-Data](https://github.com/Explore-AI/Public-Data) repository.