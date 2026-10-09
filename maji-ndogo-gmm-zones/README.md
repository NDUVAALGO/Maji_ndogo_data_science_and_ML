# maji-ndogo-gmm-zones

**Mapping the hidden geography of Maji Ndogo: soft clustering of 5,654 farm fields with Gaussian Mixture Models, projected onto real maps with GeoPandas.**

The Ministry of Agriculture had a map on the wall and a database on a server, and the two had never been connected. K-Means can say "this field belongs to zone 3," but it can't say *how sure* it is. This project replaces hard labels with a probabilistic `GaussianMixture` model, so every field gets a probability for every zone. It then puts those zones on a coordinate-aware map and produces an inspection shortlist of the fields the model is least sure about.

## What it does

1. **Scales** eight environmental features (elevation, rainfall, temperature, soil fertility, pH, pollution, plot size, yield) so no single feature dominates.
2. **Fits a GMM** with full covariance, so zones can be rotated ellipsoids where features move together.
3. **Extracts soft assignments**: a full probability distribution per field, not just one label.
4. **Chooses the number of zones** by sweeping component counts and comparing BIC and silhouette score.
5. **Profiles each zone** in real agricultural units (mean feature values per cluster).
6. **Maps the zones** with GeoPandas using `EPSG:4326` latitude/longitude.
7. **Builds a regional view** where position, marker size, and color encode region center, field count, and mean yield.
8. **Flags boundary fields** whose top probability is below 0.7 (about 442 fields), sorted most uncertain first.

## Key ideas

- **Soft membership is the point.** A field at 55% / 40% across two zones sits on a transition, and a hard cluster label would hide that.
- **The map is an independent check.** Latitude and longitude are held out of the model, yet the zones still form coherent geographic regions. That's evidence they're real and not an artifact of the algorithm.
- **Model selection is a judgment call.** Five components were chosen because Maji Ndogo has five named regions, then tested against BIC and silhouette rather than just assumed.

## Tech stack

`pandas` · `numpy` · `scikit-learn` · `GeoPandas` · `Shapely` · `matplotlib` · `SQLAlchemy` · `marimo`

## Data

The notebook reads `Maji_Ndogo_farm_survey_small.db`, a four-table SQLite database joined on `Field_ID`: geographic, weather, soil and crop, and farm management features. The database is not included in this repo, so place it next to `notebook.py` before running.

## Getting started

```bash
pip install marimo pandas numpy scikit-learn geopandas shapely matplotlib sqlalchemy
marimo edit notebook.py
```

## Part of the Maji Ndogo series

This project is one of several data science notebooks set in Maji Ndogo.