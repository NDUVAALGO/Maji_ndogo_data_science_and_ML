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
    # Compressing the Landscape
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Data ingestion is complete; now it’s time to discover. To help the Ministry of Agriculture uncover the hidden structure of Maji Ndogo's farmlands, you will shift from predicting known labels to exploring raw, unsupervised data. In this project, you will build the foundational toolkit for unsupervised machine learning using scikit-learn to compress and map 5,654 fields across 8 environmental features. You will implement **Principal Component Analysis (PCA)** to combat the Curse of Dimensionality, leverage **t-SNE** to project complex interactions into a visual 2D map, and deploy an **Isolation Forest** to isolate the top 5% most anomalous fields for targeted quality control. This is the critical exploratory baseline that anchors an unsupervised learning portfolio and sets the stage for advanced clustering.
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
    For weeks we have been predicting yields and recommending crops for Maji Ndogo — but every model we built started from the same premise: we already knew what we were looking for. We had labels. We had targets. We had answers.

    That changes today.

    The Ministry of Agriculture has asked Sanaa to do something different: make sense of 5,654 fields **without using any labels at all**. No crop types. No yield categories. Just raw measurements — elevation, rainfall, temperature, soil fertility, pH, pollution level, plot size, and standard yield — and the question: *what structure, if any, exists in this data?*

    This is the domain of **unsupervised learning**. We are not teaching the model what to find. We are asking it to find things we have not thought to look for.

    This week your tools are:

    - **PCA (Principal Component Analysis):** Compress our eight-dimensional environment into fewer dimensions while preserving as much information as possible. This is how we fight the Curse of Dimensionality — the tendency of patterns to dissolve in high-dimensional space.
    - **t-SNE (t-Distributed Stochastic Neighbor Embedding):** A non-linear technique for projecting data down to 2D for visualization, revealing clusters and groupings that PCA cannot show.
    - **Isolation Forest:** An anomaly detection algorithm that flags fields that behave very differently from the rest — potential sensor errors, unique micro-climates, or fields worth investigating in person.

    By the end of this project, Sanaa will have a compressed representation of Maji Ndogo's agricultural landscape and a prioritized list of fields the Ministry should inspect more closely. This work will also inform Week 2's clustering task — the t-SNE map we build here will tell us how many natural zones to look for.

    Let's get to work. 🌱🔬

    > **AI assist:** *"Explain the difference between PCA and t-SNE in terms of what each one preserves and what it sacrifices."* *"What does an Isolation Forest actually do to find anomalies? Walk me through the algorithm step by step."*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## The Dataset

    We load our data directly from the Maji Ndogo farm survey database. The query joins four tables — geographic features, weather features, soil and crop features, and farm management features — on `Field_ID`, giving us a single modeling-ready DataFrame with one row per field.

    After dropping any rows with missing values, we are working with **5,654 fields** and **8 environmental features**.

    **Download the database:** [Maji_Ndogo_farm_survey_small.db](https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Maji_Ndogo_farm_survey_small.db)
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
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.ensemble import IsolationForest

    return (
        IsolationForest,
        PCA,
        StandardScaler,
        TSNE,
        create_engine,
        np,
        pd,
        plt,
    )


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
    ## Challenge 1: The Essence of the Environment (PCA)

    Our 8-feature dataset lives in an 8-dimensional space. PCA finds the directions — called **principal components** — along which the data varies the most, then projects everything down into those directions. The result is a lower-dimensional representation that preserves as much information as possible.

    The key question we need to answer for Sanaa is: **how many components do we actually need?** We track this using the **cumulative explained variance ratio** — the proportion of total variance accounted for as we add more components.

    If 6 components capture 85% of the variance, we can discard 2 dimensions and lose only 15% of the information. That is the trade-off PCA makes transparent.

    ### Task
    Create a function `apply_pca` that scales the data, applies PCA with the specified number of components, and returns the cumulative explained variance ratio as a NumPy array.

    **Function specifications:**
    - Takes a DataFrame `df` and an integer `n_components` as input.
    - Scales the data using `StandardScaler` (fit and transform on the full DataFrame).
    - Applies `PCA` with `n_components` components.
    - Returns the **cumulative** explained variance ratio as a NumPy array (use `np.cumsum`).

    > **Note:** PCA is sensitive to feature scale. Always scale first.

    ### Expected Output
    ```
    Cumulative explained variance (8 components):
    [0.2514 0.4455 0.6005 0.7385 0.8372 0.9117 0.9821 1.    ]

    Components needed for ≥85% variance: 6
    ```
    """)
    return


@app.cell
def _(PCA, StandardScaler, np):
    ### START FUNCTION
    def apply_pca(df, n_components):
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        # Apply PCA
        pca = PCA(n_components=n_components)
        pca.fit(scaled_data)

        # Return the cumulative explained variance ratio
        return np.cumsum(pca.explained_variance_ratio_)
    ### END FUNCTION
    return (apply_pca,)


@app.cell
def _(apply_pca, df, np):
    cumvar = apply_pca(df, 8)
    print('Cumulative explained variance (8 components):')
    print(cumvar.round(4))

    # How many components are needed to explain at least 85% of the variance?
    n_for_85 = np.argmax(cumvar >= 0.85) + 1
    print(f'\nComponents needed for \u226585% variance: {n_for_85}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Reflect on this result. With 8 features, we might expect 8 fully independent dimensions of information — but only 6 are needed to capture 85% of the variance. The last two components contribute very little. This tells us our 8 environmental features are not fully independent: some combinations of them carry overlapping information.

    In Maji Ndogo's context, this makes sense. Elevation, temperature, and rainfall are related — higher elevations tend to be cooler, and certain rainfall patterns correlate with temperature bands. PCA finds and compresses these redundancies automatically.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 2: Visualizing the Hidden Structure (t-SNE)

    PCA is a linear technique — it projects data along straight-line directions. But agricultural patterns in the real world are rarely linear. Soil type, micro-climate, and elevation interact in complex, curved ways.

    **t-SNE (t-Distributed Stochastic Neighbor Embedding)** is a non-linear technique designed specifically for visualization. It places data points in 2D space such that points that are similar in the original high-dimensional space end up close together. It is exceptionally good at revealing clusters and groupings that PCA would smooth over.

    We will use it to generate a 2D map of Maji Ndogo's 5,654 fields. If natural agricultural zones exist in this data, t-SNE will reveal them visually — even though we have never told the algorithm anything about crop types or provinces.

    ### Task
    Create a function `run_tsne` that reduces the dataset to 2 dimensions using t-SNE and returns the coordinates as a NumPy array.

    **Function specifications:**
    - Takes a DataFrame `df` as input.
    - Scales the data using `StandardScaler`.
    - Applies `TSNE` with `n_components=2` and `random_state=42`.
    - Returns the resulting 2D array (shape: `(n_fields, 2)`, dtype: `float32`).

    > **Note:** t-SNE is stochastic but fixing `random_state` makes results reproducible. On 5,654 rows it will take 30–60 seconds to run — this is normal.

    ### Expected Output
    ```
    t-SNE output shape: (5654, 2)
    dtype: float32
    ```
    """)
    return


@app.cell
def _(StandardScaler, TSNE):
    ### START FUNCTION
    def run_tsne(df):
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        # Run t-SNE with 2 components and random_state=42
        tsne = TSNE(n_components=2, random_state=42)
        coords = tsne.fit_transform(scaled_data)

        return coords.astype('float32')
    ### END FUNCTION
    return (run_tsne,)


@app.cell
def _(df, run_tsne):
    tsne_coords = run_tsne(df)
    print(f't-SNE output shape: {tsne_coords.shape}')
    print(f'dtype: {tsne_coords.dtype}')
    return (tsne_coords,)


@app.cell
def _(plt, tsne_coords):
    # Visualize the t-SNE projection
    plt.figure(figsize=(10, 7))
    plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1], s=5, alpha=0.4, c='steelblue')
    plt.title("t-SNE Projection of Maji Ndogo's 5,654 Fields", fontsize=14)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at the plot. Do you see distinct clusters or does the data form one continuous cloud? The answer directly shapes how we approach Week 2's clustering task. If clear groupings appear here, K-Means will find them readily. If the structure is subtler, we will need to be careful about how many clusters we request.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 3: Finding the Outliers (Isolation Forest)

    Not every field in Maji Ndogo behaves like its neighbors. Some fields have unusual combinations of features — a very high pH paired with very low rainfall, or an anomalous pollution reading inconsistent with any nearby station. These could be:

    - **Sensor errors:** A malfunctioning weather station or soil probe.
    - **Data entry mistakes:** A field ID linked to the wrong set of measurements.
    - **Genuine micro-climates:** Rare agricultural environments that may deserve special attention.

    **Isolation Forest** finds these outliers by randomly partitioning the data into trees. Outliers are the points that get isolated quickly — they require fewer splits to separate from the rest because they sit far from the main body of data.

    We set `contamination=0.05`, which tells the model to flag the most anomalous 5% of fields — approximately 283 out of 5,654.

    ### Task
    Create a function `detect_anomalies` that fits an Isolation Forest and returns the indices of anomalous fields.

    **Function specifications:**
    - Takes a DataFrame `df` as input (unscaled — the Isolation Forest handles this internally).
    - Initializes `IsolationForest` with `contamination=0.05` and `random_state=42`.
    - Fits and predicts on the full DataFrame. Isolation Forest labels inliers as `1` and outliers as `-1`.
    - Returns a **list** of DataFrame indices where the prediction is `-1`.

    > **Note:** Return `df.index[preds == -1].tolist()` — these are the *row labels* (DataFrame index), not positional integers.

    ### Expected Output
    ```
    Anomalies detected: 283
    First five anomalous field indices: [15, 54, 78, 86, 99]
    ```
    """)
    return


@app.cell
def _(IsolationForest):
    ### START FUNCTION
    def detect_anomalies(df):
        # Initialize IsolationForest with contamination=0.05 and random_state=42
        iso_forest = IsolationForest(contamination=0.05, random_state=42)

        # Fit and predict
        preds = iso_forest.fit_predict(df)

        # Return a list of indices where prediction == -1
        return df.index[preds == -1].tolist()
    ### END FUNCTION
    return (detect_anomalies,)


@app.cell
def _(detect_anomalies, df):
    anomaly_indices = detect_anomalies(df)
    print(f'Anomalies detected: {len(anomaly_indices)}')
    print(f'First five anomalous field indices: {anomaly_indices[:5]}')
    return (anomaly_indices,)


@app.cell
def _(anomaly_indices, df, np, plt, tsne_coords):
    # Visualize anomalies on the t-SNE map
    is_anomaly = np.isin(df.index, anomaly_indices)

    plt.figure(figsize=(10, 7))
    plt.scatter(tsne_coords[~is_anomaly, 0], tsne_coords[~is_anomaly, 1],
                s=5, alpha=0.3, c='steelblue', label='Normal')
    plt.scatter(tsne_coords[is_anomaly, 0], tsne_coords[is_anomaly, 1],
                s=20, alpha=0.8, c='crimson', label='Anomaly')
    plt.title('Anomalous Fields Highlighted on t-SNE Map', fontsize=14)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Do the anomalies cluster in a specific region of the t-SNE map, or are they scattered throughout? If they cluster, it suggests a systematic issue — for example, a particular weather station that is consistently miscalibrated. If they are spread out, the anomalies are more likely to be individual data quality issues or genuine environmental outliers.

    Sanaa will use this list to flag fields for manual inspection before any further modeling is done on the full dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Wrapping up

    This week we applied three unsupervised techniques to Maji Ndogo's agricultural data — and we did so **without labels**. That is the key shift from everything we have done before in this course series.

    **PCA** showed us that our eight environmental features contain real redundancy: only six dimensions are needed to capture 85% of the variance. Our features are correlated, which makes agronomic sense — rainfall, temperature, and elevation co-vary in Maji Ndogo's geography.

    **t-SNE** gave us a visual map of the landscape. The clusters you see (or do not see) in that plot will directly inform how many zones we ask K-Means to find in Week 2.

    **Isolation Forest** gave Sanaa her first actionable output from DS-7: 283 fields to investigate. Before the Ministry builds a national crop recommendation system on this data, it should know whether those 283 fields represent errors or genuinely exceptional environments.
    """)
    return


@app.cell
def cell_readiness():
    import inspect as _inspect
    import ast as _ast
    _spec = {'apply_pca': {'expected_params': ['df', 'n_components']}, 'detect_anomalies': {'expected_params': ['df']}, 'run_tsne': {'expected_params': ['df']}}
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
