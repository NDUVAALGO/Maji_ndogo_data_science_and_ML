# maji-ndogo-unsupervised-landscape

**Compressing the landscape of Maji Ndogo: PCA, t-SNE, and Isolation Forest applied to 5,654 farm fields, with no labels at all.**

Every earlier model in the Maji Ndogo series started with a target: yields to predict, crops to classify. This project drops the labels and asks a different question: *what structure exists in this data?* Using only raw environmental measurements, it compresses the data, maps it in 2D, and produces a prioritized list of unusual fields for the Ministry of Agriculture to inspect.

## What it does

| Step | Technique | Question it answers |
|------|-----------|---------------------|
| 1 | **PCA** | How many dimensions do we really need? |
| 2 | **t-SNE** | Do natural agricultural zones show up visually? |
| 3 | **Isolation Forest** | Which fields are unusual enough to inspect? |

### 1. PCA: fighting the curse of dimensionality
Scales the eight features, fits PCA, and tracks the **cumulative explained variance**. Six components capture at least 85% of the variance, so the eight features contain real redundancy. That makes agronomic sense, since rainfall, temperature, and elevation move together across the country.

### 2. t-SNE: a 2D map of the fields
Projects the scaled data to two dimensions with `random_state=42` for reproducibility. Because t-SNE is non-linear, it can reveal groupings that PCA's straight-line projections smooth over. The resulting map is the baseline for deciding how many zones to look for in clustering.

### 3. Isolation Forest: the inspection shortlist
Flags the most anomalous 5% of fields (`contamination=0.05`), which is 283 of 5,654. The anomalies are plotted on the t-SNE map to check whether they cluster, which would point to a systematic problem such as a miscalibrated station, or scatter, which would suggest individual errors or genuine micro-climates.

## Data

Eight features per field, joined across four tables of a SQLite database on `Field_ID`: `Elevation`, `Rainfall`, `Ave_temps`, `Soil_fertility`, `pH`, `Pollution_level`, `Plot_size`, `Standard_yield`.

The database (`Maji_Ndogo_farm_survey_small.db`) is not included in this repo. Download it from the [Explore AI public data repo](https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Maji_Ndogo_farm_survey_small.db) and place it next to `notebook.py`.

## Key takeaways

- **Scale before PCA and t-SNE.** Both are sensitive to feature scale. Isolation Forest is not, so it runs on the raw data.
- **Unsupervised results need interpretation.** Whether the 283 flagged fields are errors or exceptional environments is a question for the Ministry to settle on the ground.

## Tech stack

`pandas` · `numpy` · `scikit-learn` · `matplotlib` · `SQLAlchemy` · `marimo`

## Getting started

```bash
pip install marimo pandas numpy scikit-learn matplotlib sqlalchemy
marimo edit notebook.py
```

## Part of the Maji Ndogo series

This project is one of several data science notebooks set in Maji Ndogo.