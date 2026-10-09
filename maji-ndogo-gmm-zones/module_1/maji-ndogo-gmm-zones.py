import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div align="center" style=" font-size: 80%; text-align: center; margin: 0 auto">
    <img src="https://raw.githubusercontent.com/Explore-AI/Pictures/refs/heads/master/Python-Notebook-Banners/Code_challenge.png"  style="display: block; margin-left: auto; margin-right: auto;";/>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mapping the Hidden Geography of Maji Ndogo
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Geospatial intelligence requires an honest treatment of systemic ambiguity. When physical boundaries overlap, forced assignments mask the actual transition risks faced by administrative planning teams. In this project, you map the hidden geography of Maji Ndogo by replacing hard-coded partitions with a probabilistic `GaussianMixture` model (GMM) framework. You will scale multi-dimensional environmental matrices, track soft cluster membership distributions to isolate boundary transition zones, and leverage `GeoPandas` to project abstract multi-variable similarity onto explicit coordinate reference systems (`EPSG:4326`). This bridges the gap between raw database infrastructure and spatial decision maps, delivering an actionable inspection checklist anchored directly in model uncertainty.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **This challenge is not graded.** It is here to build your footing before the graded work begins.

    ### Instructions

    - Do not add or remove cells in this notebook.
    - Answer the questions according to the specifications provided.
    - Use the provided **Expected output** blocks and test cells to verify your work before continuing.
    - The use of StackOverflow, Google, Generative AI tools, and any other online resources is permitted. Use AI to help you understand — not to shortcut the thinking. [Read the honor code here](https://drive.google.com/file/d/1atFOPUQRLz5slb4Q1ASXh8QQfKyXVqrw/preview).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    The Ministry of Agriculture in Maji Ndogo has a map of the country pinned to its wall, and a database of 5,654 farm fields sitting in a server. The two have never been connected.

    For weeks you have worked through that database as an international data-science consultant — searching it, cleaning it, validating it, predicting yields, and classifying crops. In the previous project you went further and let the data group *itself*, with no labels at all. The plots showed clear structure. But your client, Sanaa Lewis, wasn't satisfied:

    > *"You showed me clusters. But a visualization doesn't hand me memberships, and it never tells me how confident the model is that a field belongs to one zone versus another. If a field sits exactly on the line between two agricultural zones, I want to see that uncertainty — not have it quietly forced into one bucket. And then I want it on a map. Tell me **where** these zones are."*

    That single request defines this project, and it splits cleanly in two:

    1. **Find the zones, with honesty about uncertainty.** A **Gaussian Mixture Model (GMM)** treats each field as being *generated* by one of several underlying environmental profiles. Instead of K-Means' hard "you belong to cluster 3, full stop," a GMM returns a probability for every zone — a field can be 70% one zone and 30% another. That soft membership is exactly the uncertainty Sanaa asked for.

    2. **Put the zones on the ground.** The survey database carries real `Latitude` and `Longitude` for every field. Using **GeoPandas**, you'll turn those coordinates into a true spatial map of the discovered zones, build a multidimensional regional view that encodes a third and fourth variable through color and marker size, and finally hand the Ministry a short list of the genuinely ambiguous boundary fields worth sending an inspector to.

    By the end, you'll have a deliverable a ministry could act on — and a portfolio piece that shows you can take a model from raw database rows all the way to a map and a decision. Let's get to work. 🌍🌱

    **A note on what this project covers.** This week is strictly about *unsupervised* structure and geography: Gaussian Mixture Models, soft clustering, model selection, and geospatial visualization. No supervised prediction, no recommendation engines — those come later in the course. Everything you build here is a foundation for the similarity-based reasoning that recommendation systems rely on.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sqlalchemy import create_engine
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    import geopandas as gpd
    from shapely.geometry import Point
    import warnings
    import sqlite3
    warnings.filterwarnings('ignore')
    return (
        GaussianMixture,
        Point,
        StandardScaler,
        create_engine,
        gpd,
        np,
        pd,
        plt,
        silhouette_score,
        sqlite3,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Dataset

    We work from the Maji Ndogo farm survey database (`Maji_Ndogo_farm_survey_small.db`), the same four-table source used earlier in the course. Each of the 5,654 fields has one row spread across four tables, joined on `Field_ID`:

    | Table | What it holds |
    |-------|---------------|
    | `geographic_features` | `Field_ID`, `Elevation`, **`Latitude`**, **`Longitude`**, `Location`, `Slope` |
    | `weather_features` | `Rainfall`, `Ave_temps`, min/max temperatures |
    | `soil_and_crop_features` | `Soil_fertility`, `Soil_type`, `pH` |
    | `farm_management_features` | `Pollution_level`, `Plot_size`, `Standard_yield`, ... |

    The crucial columns for this project are the real `Latitude` and `Longitude` — they let us move from abstract feature space onto an actual map — and `Location`, the named region each field falls in (e.g. `Rural_Kilimani`). We keep both alongside the environmental features we'll cluster on.
    """)
    return


@app.cell
def _(create_engine, pd, sqlite3):
    engine = create_engine('sqlite:///Maji_Ndogo_farm_survey_small.db')

    sql_query = """
    SELECT geographic_features.Field_ID,
           Elevation, Latitude, Longitude, Location, Slope,
           Rainfall, Ave_temps,
           Soil_fertility, pH,
           Pollution_level, Plot_size, Standard_yield
    FROM geographic_features
    LEFT JOIN weather_features      USING (Field_ID)
    LEFT JOIN soil_and_crop_features USING (Field_ID)
    LEFT JOIN farm_management_features USING (Field_ID)
    """

    with sqlite3.connect("Maji_Ndogo_farm_survey_small.db") as conn:
        df = pd.read_sql_query(sql_query, conn)
    df = df.dropna().reset_index(drop=True)

    print(f"Dataset shape: {df.shape}")
    print(f"Regions: {sorted(df['Location'].unique().tolist())}")
    df.head()
    return (df,)


@app.cell
def _():
    # The eight environmental features we cluster on.
    # (Latitude / Longitude / Location are geography, NOT clustering inputs —
    #  we hold them aside so the map is an independent check on the clusters.)
    feature_cols = [
        'Elevation', 'Rainfall', 'Ave_temps', 'Soil_fertility',
        'pH', 'Pollution_level', 'Plot_size', 'Standard_yield'
    ]
    feature_cols
    return (feature_cols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 1: Scaling the Environment

    A GMM evaluates how far each field sits from each Gaussian component, so the *scale* of every feature matters. `Rainfall` runs into the thousands of millimeters while `pH` lives between roughly 4 and 8. Left unscaled, rainfall would dominate the distance calculations and the model would barely notice soil fertility or temperature. Standardizing every feature to zero mean and unit variance puts them on equal footing.

    ### Task
    Write a function `scale_features` that standardizes the feature columns and returns both the scaled array and the fitted scaler.

    **Function specifications:**
    - Takes a DataFrame `df` and a list of feature column names `feature_cols`.
    - Scales **only** the feature columns using `StandardScaler` (zero mean, unit variance).
    - Returns a tuple `(X_scaled, scaler)`:
      - `X_scaled` — a NumPy array of shape `(n_fields, n_features)`.
      - `scaler` — the fitted `StandardScaler` object.

    > **Why return the scaler?** We keep the fitted scaler so we can convert cluster centers back into real-world units later. Returning it now saves us refitting it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    Scaled array shape: (5654, 8)
    Mean of first feature (should be ~0): 0.0000
    Std of first feature  (should be ~1): 1.0000
    ```
    """)
    return


@app.cell
def _(StandardScaler):
    ### START FUNCTION
    def scale_features(df, feature_cols):
        # Your code here
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feature_cols])
        return X_scaled, scaler
    ### END FUNCTION
    return (scale_features,)


@app.cell
def _(df, feature_cols, scale_features):
    X_scaled, scaler = scale_features(df, feature_cols)

    print(f"Scaled array shape: {X_scaled.shape}")
    print(f"Mean of first feature (should be ~0): {X_scaled[:, 0].mean():.4f}")
    print(f"Std of first feature  (should be ~1): {X_scaled[:, 0].std():.4f}")
    return (X_scaled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 2: Fitting a Gaussian Mixture Model

    A GMM assumes the data was produced by a mixture of several Gaussian "blobs," each with its own mean and covariance. The **Expectation–Maximization (EM)** algorithm alternates between estimating which component each point likely came from (the **E-step**) and re-estimating each component's mean and covariance from those soft assignments (the **M-step**), repeating until the fit stops improving.

    The `covariance_type` controls the shape each component is allowed to take:
    - `'spherical'` — every component is a sphere of equal width in all directions.
    - `'diag'` — axis-aligned ellipsoids (each feature gets its own width).
    - `'full'` — any ellipsoid, including rotated ones, so features can correlate within a cluster.

    Maji Ndogo's agricultural zones are defined by *interacting* conditions — highland rainfall and temperature move together; lowland fertility and pollution move together — so `'full'` covariance is the realistic default.
    ### Task
    Write a function `fit_gmm` that fits and returns a `GaussianMixture` model.

    **Function specifications:**
    - Takes `X_scaled` (NumPy array), `n_components` (int), `covariance_type` (str, default `'full'`), and `random_state` (int, default `42`).
    - Fits a `GaussianMixture` with those settings.
    - Returns the fitted model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    Converged: True
    Iterations to convergence: 22
    Number of components: 5
    ```
    *(The exact iteration count can vary slightly across library versions, but it should converge in well under 100 iterations.)*
    """)
    return


@app.cell
def _(GaussianMixture):
    ### START FUNCTION
    def fit_gmm(X_scaled, n_components, covariance_type='full', random_state=42):
        # Your code here
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
        gmm.fit(X_scaled)
        return gmm

    ### END FUNCTION
    return (fit_gmm,)


@app.cell
def _(X_scaled, fit_gmm):
    gmm = fit_gmm(X_scaled, n_components=5)

    print(f"Converged: {gmm.converged_}")
    print(f"Iterations to convergence: {gmm.n_iter_}")
    print(f"Number of components: {gmm.n_components}")
    return (gmm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 3: Soft Cluster Assignments

    This is the heart of why Sanaa wanted a GMM. Rather than a single hard label per field, the model returns a full probability distribution across the components. A field deep inside one zone scores near 1.0 for that zone; a field on a boundary might split 0.55 / 0.40 across two zones — a genuine ambiguity that hard clustering would hide.

    ### Task
    Write a function `get_soft_assignments` that returns both the hard labels and the full probability matrix.

    **Function specifications:**
    - Takes a fitted `gmm` and `X_scaled`.
    - Returns a tuple `(labels, proba_matrix)`:
      - `labels` — integer array of shape `(n_fields,)`, the highest-probability component per field (use `gmm.predict`).
      - `proba_matrix` — float array of shape `(n_fields, n_components)`, the soft memberships (use `gmm.predict_proba`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    Labels shape: (5654,)
    Probability matrix shape: (5654, 5)
    Probabilities for field 0: [0.1126 0.8128 0.     0.0378 0.0368]
    They sum to (should be 1.0): 1.0000
    Unique cluster labels: [0, 1, 2, 3, 4]
    ```
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def get_soft_assignments(gmm, X_scaled):
        # Your code here
    
        labels = gmm.predict(X_scaled)          # hard label = argmax of proba
        proba_matrix = gmm.predict_proba(X_scaled)
        return labels, proba_matrix
   
    ### END FUNCTION
    return (get_soft_assignments,)


@app.cell
def _(X_scaled, get_soft_assignments, gmm, np):
    labels, proba_matrix = get_soft_assignments(gmm, X_scaled)

    print(f"Labels shape: {labels.shape}")
    print(f"Probability matrix shape: {proba_matrix.shape}")
    print(f"Probabilities for field 0: {proba_matrix[0].round(4)}")
    print(f"They sum to (should be 1.0): {proba_matrix[0].sum():.4f}")
    print(f"Unique cluster labels: {sorted(np.unique(labels).tolist())}")
    return labels, proba_matrix


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 4: How Many Zones, Really?

    We started with five components because Maji Ndogo has five named regions — but that's an assumption, not a finding. Two standard tools let us interrogate it:

    - **BIC (Bayesian Information Criterion):** rewards fit, penalizes complexity. **Lower is better.**
    - **Silhouette score:** how well each point sits inside its own cluster versus the nearest other cluster, from –1 to 1. **Higher is better.**

    ### Task
    Write a function `evaluate_n_components` that sweeps a range of component counts, records both metrics, plots them on a twin-axis figure, and returns a summary DataFrame.

    **Function specifications:**
    - Takes `X_scaled` (NumPy array) and `n_range` (list of ints).
    - For each `n`, fit a GMM (`covariance_type='full'`, `random_state=42`), then record:
      - `bic` via `gmm.bic(X_scaled)`.
      - `silhouette` via `silhouette_score(X_scaled, labels, sample_size=1000, random_state=42)` on the hard labels.
    - Plot BIC and silhouette against `n` on a twin-axis (`ax.twinx()`) figure.
    - Return a DataFrame with columns `['n_components', 'bic', 'silhouette']`, one row per `n`.

    > **Note on `sample_size`:** the silhouette score is O(n²); sampling 1,000 points keeps it fast and, with a fixed `random_state`, reproducible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    (a figure, then a table close to):
    ```
     n_components       bic  silhouette
                2 110002.07        0.20
                3 103764.62        0.19
                4 101267.17        0.17
                5  99968.12        0.17
                6  98265.64        0.14
                7  96096.32        0.13
                8  95545.25        0.11
    ```
    """)
    return


@app.cell
def _(GaussianMixture, pd, plt, silhouette_score):
    ### START FUNCTION
    def evaluate_n_components(X_scaled, n_range):
        # Your code here
        results = []
        for n in n_range:
            g = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
            g.fit(X_scaled)
            lbl = g.predict(X_scaled)
            results.append({
                'n_components': n,
                'bic': g.bic(X_scaled),
                'silhouette': silhouette_score(X_scaled, lbl, sample_size=1000, random_state=42)
            })
        result_df = pd.DataFrame(results)

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        ax1.plot(result_df['n_components'], result_df['bic'], 'b-o')
        ax2.plot(result_df['n_components'], result_df['silhouette'], 'r-o')
        ax1.set_xlabel('Number of components'); ax1.set_ylabel('BIC', color='b')
        ax2.set_ylabel('Silhouette score', color='r')
        plt.title('Model selection: BIC vs Silhouette')
        plt.show()
        return result_df
    ### END FUNCTION
    return (evaluate_n_components,)


@app.cell
def _(X_scaled, evaluate_n_components):
    n_range = list(range(2, 9))
    eval_df = evaluate_n_components(X_scaled, n_range)
    print(eval_df.round(2).to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Read this carefully — it is the interesting part.** BIC keeps falling as we add components, and the silhouette score *declines*. There is no clean "elbow" at five. Statistically, the environment doesn't divide into five tidy spheres. That isn't a failure: it tells us the zones overlap and blend, which is precisely why soft memberships (Challenge 3) and boundary analysis (Challenge 7) matter here. We keep five components because they map to Sanaa's five administrative regions and stay interpretable for the Ministry — a decision driven by use, made *with* the statistics in view, not against them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 5: Interpreting the Zones

    Cluster labels are just integers until someone explains what each one *means* on a farm. To brief the Ministry, Sanaa needs each zone described in real agricultural units — average elevation, rainfall, soil fertility, and so on.

    ### Task
    Write a function `describe_clusters` that returns the mean of each feature per cluster, in original (unscaled) units.

    **Function specifications:**
    - Takes `df` (the original unscaled DataFrame), `feature_cols` (list of strings), and `labels` (integer array).
    - Works on a **copy** — it must not modify the input `df`.
    - Adds the labels as a `'Cluster'` column, groups by `'Cluster'`, and takes the mean of each feature column.
    - Returns the grouped DataFrame (one row per cluster, one column per feature).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    (values close to):
    ```
             Elevation  Rainfall  Ave_temps  Soil_fertility    pH  Pollution_level  Plot_size  Standard_yield
    Cluster
    0           748.11   1573.41      13.52            0.67  4.93             0.18       2.20            0.65
    1           678.49   1110.91      13.28            0.61  5.56             0.51       2.76            0.46
    2           619.94   1296.06      13.26            0.60  5.60             0.21      10.46            0.54
    3           436.75   1606.18      13.21            0.65  6.09             0.06       2.67            0.49
    4           704.16    624.34      13.01            0.58  5.55             0.22       2.43            0.56
    ```
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def describe_clusters(df, feature_cols, labels):
        # Your code here
        df_copy = df.copy()
        df_copy['Cluster'] = labels
        return df_copy.groupby('Cluster')[feature_cols].mean()
    ### END FUNCTION
    return (describe_clusters,)


@app.cell
def _(describe_clusters, df, feature_cols, labels):
    cluster_profiles = describe_clusters(df, feature_cols, labels)
    print(cluster_profiles.round(2).to_string())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each row is now a readable agricultural profile. For example, one cluster pairs high rainfall with low pH (acidic highland), another is defined by unusually large plots, and another by high pollution and depressed yield. *Your* cluster numbers may be permuted — GMM labels are arbitrary — but the same five profiles should appear.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 6: Putting the Zones on the Map

    Now we honor the second half of Sanaa's request: *where* are these zones? Every field carries a real `Latitude` and `Longitude`, so we can build a true **GeoDataFrame** and plot each field as a point on the map of Maji Ndogo, colored by its GMM zone.

    This is genuine geospatial work — we convert raw coordinate columns into Shapely `Point` geometries, attach a coordinate reference system (CRS), and let GeoPandas handle the plotting.

    ### Task
    Write a function `build_geodataframe` that turns the labeled DataFrame into a GeoDataFrame and plots it.

    **Function specifications:**
    - Takes `df_labeled` — a DataFrame containing `Longitude`, `Latitude`, and `Cluster` columns.
    - Builds a list of `shapely.geometry.Point` objects as `Point(longitude, latitude)` (note: **longitude first**, the x–y convention).
    - Wraps it in `gpd.GeoDataFrame(df_labeled, geometry=..., crs='EPSG:4326')` (EPSG:4326 is standard latitude/longitude).
    - Plots the GeoDataFrame colored by `'Cluster'` (categorical, with a legend and a title).
    - Returns the GeoDataFrame.

    > **Note:** `EPSG:4326` is the WGS-84 latitude/longitude system used by GPS. Use `gdf.plot(column='Cluster', categorical=True, legend=True, ...)`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    (a map, then):
    ```
    GeoDataFrame shape: (5654, 15)
    CRS: EPSG:4326
    Geometry type: ['Point']
    ```
    """)
    return


@app.cell
def _(Point, gpd, plt):
    ### START FUNCTION
    def build_geodataframe(df_labeled):
        # Your code here
        geometry = [Point(lon, lat) for lon, lat in zip(df_labeled['Longitude'], df_labeled['Latitude'])]
        gdf = gpd.GeoDataFrame(df_labeled, geometry=geometry, crs='EPSG:4326')
        fig, ax = plt.subplots(figsize=(8, 8))
        gdf.plot(column='Cluster', categorical=True, legend=True, ax=ax, markersize=5)
        ax.set_title('GMM Zones of Maji Ndogo')
        plt.show()
        return gdf
    ### END FUNCTION
    return (build_geodataframe,)


@app.cell
def _(build_geodataframe, df, labels):
    df_labeled = df.copy()
    df_labeled['Cluster'] = labels

    gdf = build_geodataframe(df_labeled)

    print(f"GeoDataFrame shape: {gdf.shape}")
    print(f"CRS: {gdf.crs}")
    print(f"Geometry type: {gdf.geometry.geom_type.unique().tolist()}")
    return (df_labeled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at the map. The GMM never saw a single coordinate — it clustered purely on environmental features — yet the colors form coherent geographic regions. That spatial coherence is independent evidence that the zones are real and not an artifact of the algorithm. Where colors interleave, you're seeing genuine transition land between zones.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 7: A Multidimensional Regional View

    A scatter of 5,654 points shows individual fields but buries the regional story. The planning brief asks specifically for **multidimensional plots that encode extra variables through size and color** — so here we summarize each named `Location` into a single marker whose **position** is the region's center, whose **size** encodes how many fields it contains, and whose **color** encodes its average yield. Four variables, one readable plot.

    ### Task
    Write a function `regional_summary` that aggregates fields by region and plots the multidimensional view.

    **Function specifications:**
    - Takes `df_labeled` (must contain `Location`, `Latitude`, `Longitude`, and `Standard_yield`).
    - Groups by `'Location'` and computes, per region:
      - `lat` = mean `Latitude`, `lon` = mean `Longitude`
      - `field_count` = number of fields
      - `mean_yield` = mean `Standard_yield`
    - Produces a scatter plot where marker **position** is `(lon, lat)`, marker **size** scales with `field_count`, and marker **color** encodes `mean_yield` (include a colorbar).
    - Returns the summary DataFrame with columns `['Location', 'lat', 'lon', 'field_count', 'mean_yield']` (in any column order).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    (a bubble map, then values close to):
    ```
          Location    lat    lon  field_count  mean_yield
      Rural_Akatsi -9.917 -7.984          727       0.523
      Rural_Amanzi -6.050 -7.320          259       0.518
     Rural_Hawassa -9.663 -4.933         1734       0.516
    Rural_Kilimani -3.174 -3.915         2020       0.532
      Rural_Sokoto -8.516 -0.572          914       0.588
    ```
    """)
    return


@app.cell
def _(plt):
    ### START FUNCTION
    def regional_summary(df_labeled):
        # Your code here
        summary = df_labeled.groupby('Location').agg(
            lat=('Latitude', 'mean'),
            lon=('Longitude', 'mean'),
            field_count=('Location', 'size'),
            mean_yield=('Standard_yield', 'mean')
        ).reset_index()

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(summary['lon'], summary['lat'],
                              s=summary['field_count'] / 5,
                              c=summary['mean_yield'], cmap='viridis')
        plt.colorbar(scatter, label='Mean yield')
        for _, row in summary.iterrows():
            ax.annotate(row['Location'], (row['lon'], row['lat']))
        plt.show()
        return summary
    ### END FUNCTION
    return (regional_summary,)


@app.cell
def _(df_labeled, regional_summary):
    region_df = regional_summary(df_labeled)
    print(region_df.round(3).to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In one figure you can now read where each region sits, how many fields it holds (bubble size), and how productive it is on average (color). Sokoto stands out as the smaller-but-higher-yielding region — the kind of pattern that earns a second look from the Ministry.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 8: Flagging the Boundary Fields

    Finally, the deliverable Sanaa asked for by name: the fields the model is genuinely *unsure* about. These are where soft membership earns its keep — a field whose top zone probability is only, say, 0.45 is sitting on a transition and deserves a human inspection before any planting decision is locked in.

    ### Task
    Write a function `find_boundary_fields` that returns the most ambiguous fields.

    **Function specifications:**
    - Takes `proba_matrix` (NumPy array), `df` (DataFrame containing `Field_ID`), and `threshold` (float, default `0.7`).
    - For each field, find its maximum probability across clusters and the cluster that achieves it.
    - Keep only fields whose maximum probability is **strictly below** `threshold`.
    - Return a DataFrame with columns `['Field_ID', 'Max_probability', 'Best_cluster']`, sorted ascending by `Max_probability` (most uncertain first) with a reset index.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    (count and exact IDs depend on label ordering, but close to):
    ```
    Fields below the 0.7 confidence threshold: 442

    Ten most uncertain fields:
     Field_ID  Max_probability  Best_cluster
         8474         0.344607             1
        18349         0.352651             3
           98         0.360504             1
        45083         0.385918             1
        43130         0.390089             0
         8909         0.394064             1
        43369         0.403487             3
        42988         0.404002             1
        32661         0.406500             1
        37810         0.409989             1
    ```
    """)
    return


@app.cell
def _(pd):
    ### START FUNCTION
    def find_boundary_fields(proba_matrix, df, threshold=0.7):
        # Your code here
        max_proba = proba_matrix.max(axis=1)
        best_cluster = proba_matrix.argmax(axis=1)
        result = pd.DataFrame({
            'Field_ID': df['Field_ID'],
            'Max_probability': max_proba,
            'Best_cluster': best_cluster
        })
        result = result[result['Max_probability'] < threshold]
        return result.sort_values('Max_probability').reset_index(drop=True)
    ### END FUNCTION
    return (find_boundary_fields,)


@app.cell
def _(df, find_boundary_fields, proba_matrix):
    boundary_df = find_boundary_fields(proba_matrix, df)

    print(f"Fields below the 0.7 confidence threshold: {len(boundary_df)}")
    print("\nTen most uncertain fields:")
    print(boundary_df.head(10).to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These few hundred fields are the Ministry's inspection shortlist — the places where the data itself admits it can't decide. That honesty is the whole point of choosing a GMM.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Wrapping up

    Starting from 5,654 rows spread across four database tables, you delivered exactly what Sanaa asked for:

    1. **Scaled** eight environmental features so distances were meaningful.
    2. **Fit a GMM** with the EM algorithm, obtaining soft memberships rather than hard labels.
    3. **Extracted soft assignments** — a full probability distribution per field.
    4. **Interrogated the cluster count** with BIC and silhouette, and made a defensible, use-driven choice in full view of the statistics.
    5. **Profiled each zone** in real agricultural units.
    6. **Mapped the zones** with GeoPandas using true coordinates — and found the clusters were spatially coherent without ever seeing a coordinate.
    7. **Built a multidimensional regional view** encoding count and yield through size and color.
    8. **Flagged the boundary fields** — a concrete, prioritized inspection list for the Ministry.

    You took a model from raw rows all the way to a map and a decision. That arc — environmental similarity → soft membership → actionable shortlist — is the same reasoning you'll soon apply to people instead of fields: *which users resemble each other, and how confident are we?* That is where recommendation systems begin, and it's where this course goes next.

    Until then.

    — Sanaa
    """)
    return


@app.cell
def cell_readiness():
    import inspect as _inspect
    import ast as _ast
    _spec = {'build_geodataframe': {'expected_params': ['df_labeled']}, 'describe_clusters': {'expected_params': ['df', 'feature_cols', 'labels']}, 'evaluate_n_components': {'expected_params': ['X_scaled', 'n_range']}, 'find_boundary_fields': {'expected_params': ['proba_matrix', 'df', 'threshold']}, 'fit_gmm': {'expected_params': ['X_scaled', 'n_components', 'covariance_type', 'random_state']}, 'get_soft_assignments': {'expected_params': ['gmm', 'X_scaled']}, 'regional_summary': {'expected_params': ['df_labeled']}, 'scale_features': {'expected_params': ['df', 'feature_cols']}}
    _msgs = []
    _missing = 0
    _stubs = 0
    _mismatches = 0
    for _name, _checks in _spec.items():
        try:
            _fn = eval(_name)
        except NameError:
            _msgs.append(f"❌ `{_name}` — not defined yet")
            _missing += 1
            continue
        if not callable(_fn):
            _msgs.append(f"❌ `{_name}` — exists but is not callable")
            _missing += 1
            continue
        try:
            _sig = _inspect.signature(_fn)
            _actual_params = list(_sig.parameters.keys())
        except (ValueError, TypeError):
            _msgs.append(f"✓ `{_name}` — defined (signature unavailable)")
            continue
        _expected = _checks.get("expected_params")
        if _expected and _actual_params != _expected:
            _msgs.append(
                f"⚠️ `{_name}` — parameter mismatch: expected {_expected}, got {_actual_params}"
            )
            _mismatches += 1
            continue
        # AST check — flag empty stubs (function body is just `pass` or `return None`)
        _is_stub = False
        try:
            import textwrap as _textwrap
            _src = _textwrap.dedent(_inspect.getsource(_fn))
            _tree = _ast.parse(_src)
            _body = _tree.body[0].body if _tree.body else []
            # Strip any leading docstring (an Expr with a Constant str)
            _real_body = [
                _stmt for _stmt in _body
                if not (
                    isinstance(_stmt, _ast.Expr)
                    and isinstance(_stmt.value, _ast.Constant)
                    and isinstance(_stmt.value.value, str)
                )
            ]
            _is_stub = (
                not _real_body
                or (
                    len(_real_body) == 1
                    and (
                        isinstance(_real_body[0], _ast.Pass)
                        or (isinstance(_real_body[0], _ast.Return) and _real_body[0].value is None)
                    )
                )
            )
        except (OSError, TypeError, SyntaxError, IndexError):
            pass  # source unavailable; skip stub check
        if _is_stub:
            _msgs.append(
                f"⚠️ `{_name}` — body looks empty (just `pass` or bare `return`). Did you implement it?"
            )
            _stubs += 1
            continue
        _msgs.append(f"✓ `{_name}` — defined with parameters {_actual_params}")
    print("Readiness check:")
    for _m in _msgs:
        print(f"  {_m}")
    print()
    if _missing == len(_spec):
        print("It looks like none of the required functions are defined yet.")
        print("Click 'Run all cells' (the ▶▶ button at the top of the page),")
        print("or run each function-defining cell individually first.")
    elif _missing > 0:
        print(f"{_missing} function(s) not yet defined — run the cells that define them and re-run this check.")
    elif _stubs > 0 or _mismatches > 0:
        print(f"Some functions need fixing before submitting ({_stubs} stub(s), {_mismatches} signature mismatch(es)).")
    else:
        print("All required functions are present and ready. You can submit.")
    return


if __name__ == "__main__":
    app.run()
