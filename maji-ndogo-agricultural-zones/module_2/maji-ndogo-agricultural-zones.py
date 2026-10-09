import marimo

__generated_with = "0.24.2"
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
    # Mapping Agricultural Zones
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Data compression is complete; now it's time to group. To help the Ministry of Agriculture establish data-driven **Agricultural Zones** across Maji Ndogo, you will transition from exploring individual field anomalies to uncovering macro-level regional structures. In this project, you will master the core algorithms of unsupervised clustering using scikit-learn and SciPy to partition 5,654 fields based on shared environmental features. You will implement **K-Means Clustering** and use the Elbow Method to determine the optimal number of operational zones, construct an agglomerative **Hierarchical Clustering** dendrogram to map out relationships between clusters, and extract **Cluster Centroids** to translate raw statistical outputs into actionable, human-readable zone personas. It is the definitive grouping baseline that anchors an unsupervised clustering portfolio.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ⚠️ **This challenge is not graded.** It is here to build your footing before the graded work begins and part of your data science portfolio.

    ---

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
    Last week we compressed Maji Ndogo's landscape. We proved that eight environmental features contain real redundancy, visualized the hidden structure of 5,654 fields with t-SNE, and handed Sanaa a list of anomalous fields for manual inspection.

    This week, we group.

    The Ministry of Agriculture wants to define **Agricultural Zones** — regions that share similar soil chemistry, climate, and terrain characteristics regardless of which administrative province they fall within. This is a different question from crop classification: we are not asking *what crop grows here*, we are asking *which fields are fundamentally similar to each other?*

    Zones defined this way can drive practical policy: shared irrigation infrastructure, bulk fertilizer procurement, pest management programs, and climate resilience planning — all calibrated to the actual agricultural character of the land rather than historical administrative boundaries.

    Your tools this week are:

    - **K-Means Clustering:** Find the optimal number of zones using the Elbow Method, then group all 5,654 fields into those zones.
    - **Hierarchical Clustering:** Build a dendrogram that shows how zones relate to each other — which zones are more similar, and how they would merge if we needed fewer zones.
    - **Cluster Centroid Interpretation:** Translate the mathematical zone descriptions into human-readable Agricultural Zone personas that the ministry can use in planning documents.

    Ready to draw some boundaries? 🌱🗺️

    > **AI assist:** *"I have K-Means cluster centroids for agricultural data. Here are the feature means per cluster: [paste table]. Act as an expert agronomist and name each cluster, describe its typical field conditions, and suggest suitable crops."* Use this after completing Challenge 3.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Imports
    """)
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sqlalchemy import create_engine, text
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import scipy.cluster.hierarchy as sch
    from scipy.cluster.hierarchy import dendrogram

    return KMeans, StandardScaler, create_engine, dendrogram, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## The Dataset

    We use the same query as Week 1, loading eight environmental features for all 5,654 fields from the farm survey database.

    **Download the database:** [Maji_Ndogo_farm_survey_small.db](https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Maji_Ndogo_farm_survey_small.db)
    """)
    return


@app.cell
def _(create_engine, pd):
    import os

    db_file = 'Maji_Ndogo_farm_survey_small.db'
    if not os.path.exists(db_file):
        raise FileNotFoundError(
            f"Database file '{db_file}' not found. "
            "Please place Maji_Ndogo_farm_survey_small.db in the same folder as this notebook."
        )

    engine = create_engine(f'sqlite:///{db_file}')
    conn = engine.raw_connection()

    sql_query = """
    SELECT Elevation, Rainfall, Ave_temps, Soil_fertility, pH, Pollution_level, Plot_size, Standard_yield
    FROM geographic_features
    LEFT JOIN weather_features USING (Field_ID)
    LEFT JOIN soil_and_crop_features USING (Field_ID)
    LEFT JOIN farm_management_features USING (Field_ID)
    """

    df = pd.read_sql_query(sql_query, conn).dropna()
    conn.close()

    print(f'Dataset shape: {df.shape}')
    print(df.head())
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 1: Finding the Elbow (K-Means)

    Before we can cluster, we need to decide how many clusters to use. Asking K-Means for 2 zones when 5 exist naturally will give us meaningless groupings. Asking for 20 when the data has only 4 natural groups will give us zones too small to be useful for policy.

    The **Elbow Method** helps us find the sweet spot. We run K-Means for values of k from 1 to 10 and record the **inertia** — the sum of squared distances from each point to its assigned cluster center — at each k. As k increases, inertia always decreases (more clusters means each center is closer to its members). The "elbow" is the point where adding another cluster stops giving meaningful gains: the curve bends sharply and then flattens.

    ### Task
    Create a function `get_k_means_inertia` that returns a list of inertia values for each k in the specified range.

    **Function specifications:**
    - Takes a DataFrame `df` and an iterable `k_range` as input.
    - Scales the data using `StandardScaler`.
    - For each k in `k_range`, fits a `KMeans` model with `n_clusters=k`, `random_state=42`, and `n_init=10`.
    - Returns a **list** of inertia values, one per k, in the order of `k_range`.

    > **Note:** `n_init=10` runs K-Means 10 times with different initializations and returns the best result. This avoids the algorithm getting stuck in a local minimum.

    ### Expected Output
    ```
    Inertia values (k=1 to 10):
      k=1: 45232.00
      k=2: 36307.24
      k=3: 31194.68
      k=4: 27338.84
      k=5: 24644.77
      k=6: 22664.10
      k=7: 21271.08
      k=8: 20338.30
      k=9: 19441.30
      k=10: 18725.28
    ```
    """)
    return


@app.cell
def _(KMeans, StandardScaler):
    ### START FUNCTION
    def get_k_means_inertia(df, k_range):
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        # For each k in k_range, fit KMeans and record inertia
        inertias = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(scaled_data)
            inertias.append(km.inertia_)
        return inertias
    ### END FUNCTION
    return (get_k_means_inertia,)


@app.cell
def _(df, get_k_means_inertia):
    k_range = range(1, 11)
    inertia_values = get_k_means_inertia(df, k_range)
    print('Inertia values (k=1 to 10):')
    for _k, inertia in zip(k_range, inertia_values):
        print(f'  k={_k}: {inertia:.2f}')
    return inertia_values, k_range


@app.cell
def _(inertia_values, k_range, plt):
    # Plot the Elbow Curve
    plt.figure(figsize=(9, 5))
    plt.plot(list(k_range), inertia_values, marker='o', linewidth=2, color='steelblue')
    plt.title('Elbow Method — Optimal Number of Agricultural Zones', fontsize=13)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    plt.xticks(list(k_range))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at the curve. Where does it bend? The rate of improvement slows noticeably at a particular value of k — that is where adding another zone gives diminishing returns. Use this to decide how many Agricultural Zones to use in Challenges 2 and 3.

    If the curve is very gradual with no sharp bend, that tells you something too: it suggests the data does not have a small number of strongly separated natural clusters. In that case, the choice of k is a policy decision as much as a statistical one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 2: The Family Tree (Hierarchical Clustering)

    K-Means assigns each field to exactly one zone. But it cannot tell us how those zones relate to each other. Are zones 1 and 3 more similar than zones 1 and 4? Which zones would merge first if we needed to consolidate from five zones to four?

    **Hierarchical clustering** answers these questions. It builds a tree — called a **dendrogram** — by progressively merging the most similar observations. We use **agglomerative clustering** (bottom-up): every field starts in its own cluster, and at each step the two most similar clusters are merged until only one cluster remains.

    **Ward's linkage** is the merging rule we use: at each step, merge the two clusters that minimize the increase in total within-cluster variance. This tends to produce compact, well-separated clusters and is the standard choice for most applications.

    Because hierarchical clustering computes a distance matrix between all pairs of points, it becomes slow on very large datasets. We work with a **sample of 500 fields** for the dendrogram — enough to reveal the structure without the computational cost.

    ### Task
    Create a function `get_linkage_matrix` that returns a Ward linkage matrix from SciPy.

    **Function specifications:**
    - Takes a DataFrame `df` as input (this will be a 500-row sample — see the call cell below).
    - Scales the data using `StandardScaler`.
    - Computes and returns the linkage matrix using `scipy.cluster.hierarchy.linkage` with `method='ward'`.
    - The returned array has shape `(n-1, 4)` for an input of n points.

    ### Expected Output
    ```
    Linkage matrix shape: (499, 4)
    Last merge (root of dendrogram): [995.     997.      36.3652 500.    ]
    ```
    """)
    return


@app.cell
def _(StandardScaler):
    from scipy.cluster.hierarchy import linkage

    ### START FUNCTION
    def get_linkage_matrix(df):
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        # Compute and return the Ward linkage matrix using scipy
        Z = linkage(scaled_data, method='ward')
        return Z
    ### END FUNCTION
    return (get_linkage_matrix,)


@app.cell
def _(df, get_linkage_matrix):
    # We sample 500 fields for the dendrogram — the full dataset is computationally expensive
    df_sample = df.sample(500, random_state=42)

    Z = get_linkage_matrix(df_sample)
    print(f'Linkage matrix shape: {Z.shape}')
    print(f'Last merge (root of dendrogram): {Z[-1].round(4)}')
    return (Z,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The linkage matrix has one row per merge. The four columns are: index of cluster 1, index of cluster 2, distance at which they merged, and total number of points in the merged cluster. The last row is always the root — the final merge that combines everything into one cluster.
    """)
    return


@app.cell
def _(Z, dendrogram, plt):
    # Plot the dendrogram
    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=90,
               leaf_font_size=10, show_contracted=True,
               color_threshold=0.7 * max(Z[:, 2]))
    plt.title('Dendrogram of Maji Ndogo Agricultural Fields (500-field sample, Ward linkage)', fontsize=13)
    plt.xlabel('Cluster (number of fields in parentheses)')
    plt.ylabel('Ward Distance')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the dendrogram from the bottom up. Short vertical lines represent merges of very similar fields. Tall vertical lines represent merges of dissimilar groups — a tall line tells you that two groups were far apart before being forced together.

    Draw a horizontal line across the dendrogram at a height where it cuts through the fewest vertical lines. The number of lines it crosses is your suggested number of clusters. Does this match your elbow curve result?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 3: Describing the Zones (Cluster Centroids)

    K-Means gives us a zone label for every field. But a number — "Zone 2" — means nothing to a minister or a field agronomist. We need to translate the mathematics into language.

    The centroid of each cluster is the mean value of every feature across all fields assigned to that cluster. It is the "typical" field for that zone. If Zone 3 has a mean elevation of 1,800 m, mean rainfall of 2,200 mm, and mean pH of 5.8, an agronomist immediately recognizes that as a high-altitude tea-growing environment.

    Once we have centroids, we can pass them to an AI and ask it to generate **Agricultural Zone Personas** — descriptive names and summaries that the ministry can use in planning documents and communications.

    ### Task
    Create a function `get_cluster_centroids` that returns the mean feature values for each cluster.

    **Function specifications:**
    - Takes a DataFrame `df` and a NumPy array `labels` (cluster assignments, one per row) as input.
    - Adds a `'Cluster'` column to a **copy** of the DataFrame (do not modify the original).
    - Groups by `'Cluster'` and returns the mean of all original features.
    - Returns a DataFrame indexed by cluster label, with the same columns as `df`.

    ### Expected Output (for k=4)
    ```
             Elevation  Rainfall  Ave_temps  Soil_fertility    pH  \
    Cluster
    0           699.78    750.68      13.04            0.59  5.58
    1           668.50   1130.76      13.29            0.61  5.46
    2           742.81   1680.02      13.69            0.66  4.91
    3           385.21   1627.78      13.13            0.65  6.28

             Pollution_level  Plot_size  Standard_yield
    Cluster
    0                   0.21       3.92            0.55
    1                   0.67       3.94            0.42
    2                   0.16       3.64            0.64
    3                   0.06       4.14            0.49
    ```
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def get_cluster_centroids(df, labels):
        # Add cluster labels to a copy of df
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = labels

        # Group by cluster and return the mean
        return df_with_clusters.groupby('Cluster').mean()
    ### END FUNCTION
    return (get_cluster_centroids,)


@app.cell
def _(KMeans, StandardScaler, df, get_cluster_centroids):
    # Fit K-Means with your chosen k (adjust based on your elbow analysis above)
    _k = 4
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    kmeans = KMeans(n_clusters=_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_data)
    centroids = get_cluster_centroids(df, labels)
    print(f'Cluster centroids (k={_k}):')
    print(centroids.round(2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare the centroids across zones. Which zone has the highest rainfall? Which has the lowest pollution? Which has the best soil fertility? These differences are what distinguish each Agricultural Zone and will drive different policy prescriptions.

    If two zones have very similar centroids across all features, consider whether k was too high — they might be one natural zone that K-Means split unnecessarily.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Wrapping up

    We have now completed the unsupervised learning journey for Maji Ndogo — and delivered something the ministry can actually use.

    **K-Means** gave us Agricultural Zones: a data-driven partition of 5,654 fields into groups that share similar environmental profiles. The Elbow Method told us how many zones were statistically appropriate.

    **Hierarchical clustering** gave us the family tree of those zones: which zones are most similar to each other, and at what cost they could be merged or split. The dendrogram is a powerful communication tool — a ministry official with no statistics background can look at it and understand that some zones are closely related while others are fundamentally different.

    **Persona generation** bridged the gap between mathematical output and actionable insight. A K-Means centroid is a row of numbers. An "Agricultural Zone Persona" is something a minister can present at a budget meeting.
    """)
    return


@app.cell
def cell_readiness():
    import inspect as _inspect
    import ast as _ast
    _spec = {'get_cluster_centroids': {'expected_params': ['df', 'labels']}, 'get_k_means_inertia': {'expected_params': ['df', 'k_range']}, 'get_linkage_matrix': {'expected_params': ['df']}}
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
