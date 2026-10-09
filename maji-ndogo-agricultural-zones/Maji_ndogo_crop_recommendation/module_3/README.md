# Maji Ndogo Crop Recommendation: Neural Network vs Classical Classifiers

A multi-class classification and model-selection project that predicts **which crop a field is growing** from its terrain, soil, weather and management data. A Keras neural network is compared with five classical scikit-learn classifiers under 5-fold stratified cross-validation, and the chosen model is then checked on a held-out test set. The work is a [marimo](https://marimo.io) notebook.

## The question

The Maji Ndogo Ministry of Agriculture wants a crop recommendation system: given what is known about a field, what should be planted there? The real brief is not just to build a model but to show **which model is steady enough to rely on**, rather than picking the first classifier that happens to work.

## Dataset

The Maji Ndogo farm survey is a SQLite database, `Maji_Ndogo_farm_survey_small.db`, spread across four tables that share a `Field_ID`. The notebook joins them into one row per field: **5,654 fields, 11 features, 8 crop classes**.

| Group | Features |
|---|---|
| Terrain | `Elevation`, `Slope` |
| Management | `Plot_size`, `Pollution_level` |
| Soil | `pH`, `Soil_fertility`, `Soil_type` |
| Weather | `Rainfall`, `Min_temperature_C`, `Max_temperature_C`, `Ave_temps` |
| **Target** | `Crop_type`: banana, cassava, coffee, maize, potato, rice, tea, wheat |

The crop names are stored in the survey's `Annual_yield` column, so the notebook aliases it to `Crop_type` on load. The labels are hand-entered and messy: 14 raw strings (such as `'wheat '`, `'wheatn'`, `'teaa'`) collapse to 8 real crops once whitespace and typos are cleaned.

## Approach

1. **Data preparation:** drop missing values, clean the crop labels, drop `Field_ID`, label-encode `Soil_type` and the target, scale features with `StandardScaler`, and split 80/20 (`random_state=42`).
2. **Neural network:** a Keras feed-forward model (Dense 128, Dropout 0.3, Dense 64, Dropout 0.3, Softmax over 8 classes), trained for 50 epochs with a 20% validation split, with training curves checked for overfitting.
3. **Classical classifiers:** Logistic Regression, K-Nearest Neighbors, SVM (RBF), Random Forest and AdaBoost.
4. **Cross-validation:** 5-fold `StratifiedKFold` (shuffled, `random_state=42`) comparing mean accuracy and standard deviation across folds.
5. **Visualisation:** a bar chart of the cross-validation results with error bars, with the neural network's accuracy as a reference line.
6. **Final evaluation:** the top cross-validation model is refitted on the full training set and scored on the held-out test set, with a per-crop classification report.

## Results

**Cross-validation on the training set (5 folds)**

| Classifier | Mean accuracy | Std |
|---|---|---|
| Support Vector Machine | 0.4519 | 0.0125 |
| Random Forest | 0.4519 | 0.0076 |
| Logistic Regression | 0.4477 | 0.0194 |
| AdaBoost | 0.4287 | 0.0098 |
| K-Nearest Neighbors | 0.3909 | 0.0099 |

**Held-out test set (1,131 fields)**

| Model | Test accuracy |
|---|---|
| Neural network | 0.4757 |
| SVM (selected by cross-validation) | 0.4695 |

**Takeaways**
- Accuracy is modest, below 50% for every model, so the structured survey features only partly separate eight overlapping crops. A reasonable next step is richer features, such as the weather-station logs, more data, or better labels, rather than a fancier model.
- The neural network and the best classical model finish within about one percentage point of each other, so the added complexity did not clearly pay off.
- Performance varies a lot by crop. Tea is predicted well (F1 about 0.76), while maize and rice are rarely identified correctly (maize F1 0.00 with the SVM).

## Tech stack

Python, marimo, pandas, NumPy, scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn, SQLite.

## Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install marimo pandas numpy matplotlib seaborn scikit-learn tensorflow
marimo edit notebook.py
```

Place `Maji_Ndogo_farm_survey_small.db` in the same folder as `notebook.py` before running. The notebook stops with a clear error if the file is missing. The database is available from the [Explore-AI Public-Data](https://github.com/Explore-AI/Public-Data) repository.

## Project structure

```
.
├── notebook.py                          # marimo notebook with all seven challenges
├── Maji_Ndogo_farm_survey_small.db      # dataset (download separately)
└── README.md
```

## Credits

The challenge and data come from the Explore AI data science programme.