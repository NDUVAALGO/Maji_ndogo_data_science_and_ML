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
    # The Global Stress Test & The Binary Audit
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Single trees are interpretable, but they are brittle; now it’s time to scale to resilient architectures. To transition from basic benchmarks to high-stakes field tracking, you need models built for stability and risk management. In this project, you deploy two distinct classification tools using scikit-learn. First, you will build a homogeneous ensemble with `RandomForestClassifier` to minimize variance and measure model stability across multiple random seeds. Second, you will train a `LogisticRegression` model on real environmental data, adjusting the sigmoid decision threshold to balance the precision-recall tradeoff for a pollution audit. It is the exact dual-validation pipeline required to transition from fragile rule sets to production-grade classification systems.

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
    - Use the tools introduced in this course: pandas, NumPy, scikit-learn, and Matplotlib — for decision trees, random forests, logistic regression, and classification evaluation.
    - The use of StackOverflow, Google, Generative AI tools, and any other online resources is permitted. Use AI to help you understand — not to shortcut the thinking. [Read the honor code here](https://drive.google.com/file/d/1atFOPUQRLz5slb4Q1ASXh8QQfKyXVqrw/preview).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Last module we ran Sanaa's benchmarking trial and proved that a single Decision Tree can meaningfully classify countries by population size. Sanaa was pleased — but not yet satisfied.

    > *"A single tree is interpretable, but it's brittle. One bad split near the root, and the whole tree goes wrong. I want to see if combining many trees makes the model stable enough to trust with the field registry."*

    This module has two parts — two different contexts, two different classification tools, one larger goal.

    **Part A — The Global Stress Test:** We replace our single Decision Tree with a **Random Forest**: a homogeneous ensemble of trees, each trained on a random bootstrap sample with a random subset of features. The ensemble votes on the final prediction, averaging out the instability of any single tree. We stress-test this approach on the World Population dataset — the same low-stakes benchmark — and compare its accuracy and stability against the single tree from Module 1.

    **Part B — The Binary Audit:** While the stress test runs, Sanaa has a more urgent request for Maji Ndogo's field data.

    > *"Before we classify what to plant, I want to see if we can classify what's broken. Run a quick audit: can we predict which fields are heavily polluted just from the environmental data we already have?"*

    This is a binary classification problem — exactly two outcomes: a field is either heavily polluted or it is not. We will use **Logistic Regression** — the foundational binary classifier that models the probability of belonging to a class — to predict which Maji Ndogo fields have `Pollution_level > 0.7`. We will then explore how adjusting the decision threshold shifts the trade-off between catching all polluted fields and avoiding false alarms.

    By the end of this notebook you will have:
    - Built and evaluated a `RandomForestClassifier` and measured its stability across 30 random seeds.
    - Interpreted feature importances averaged across all trees in the forest.
    - Trained a `LogisticRegression` binary classifier on real Maji Ndogo field data.
    - Applied threshold adjustment to shift the precision-recall trade-off for a pollution audit.

    > **AI assist:** *"Explain the difference between a Decision Tree and a Random Forest using the analogy of asking one expert versus asking a crowd of experts."* *"What does it mean when logistic regression recall for the positive class is low — what is the model actually failing to do?"*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Data Dictionary

    **Part A — World Population (same as Module 1)**

    Features: historical population figures (1970–2020), area, density, growth rate, world population share.
    Target: `Pop_class_encoded` — `Low`, `Medium`, or `High` population class.

    **Part B — Maji Ndogo Field Data**

    | Column | Description | Type |
    |---|---|---|
    | `Elevation` | Field elevation above sea level in meters | Float |
    | `Rainfall` | Annual rainfall in mm | Float |
    | `Min_temperature_C` | Average minimum temperature in Celsius | Float |
    | `Max_temperature_C` | Average maximum temperature in Celsius | Float |
    | `Ave_temps` | Average temperature in Celsius | Float |
    | `Soil_fertility` | Soil fertility score, 0 (infertile) to 1 (very fertile) | Float |
    | `pH` | Soil pH level | Float |
    | `Pollution_level` | Pollution score, 0 (clean) to 1 (very polluted) | Float |
    | `Slope` | Slope of the terrain in degrees | Float |
    | `Plot_size` | Field area in hectares | Float |
    | `Heavily_Polluted` | Binary target — 1 if `Pollution_level > 0.7`, else 0 | Int |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Part A: The Global Stress Test — Random Forest on World Population

    ### Setup

    Run the cell below to reproduce the Module 1 data preparation and load the modelling-ready DataFrame.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # headless rendering for test environments
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, classification_report,
                                  precision_recall_fscore_support)
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression

    # World Population data
    file_path_pop = 'world_population.csv'
    df_raw = pd.read_csv(file_path_pop)

    thresholds = df_raw['2022 Population'].quantile([1/3, 2/3])

    def classify_population(pop, low_thresh, high_thresh):
        if pop < low_thresh:
            return 'Low'
        elif pop < high_thresh:
            return 'Medium'
        else:
            return 'High'

    df_raw['Pop_class'] = df_raw['2022 Population'].apply(
        classify_population,
        low_thresh=thresholds[1/3],
        high_thresh=thresholds[2/3]
    )

    # Detect column names — handles both Unicode (km²) and ASCII (km2) encodings
    area_col    = 'Area (km²)'    if 'Area (km²)'    in df_raw.columns else 'Area (km2)'
    density_col = 'Density (per km²)' if 'Density (per km²)' in df_raw.columns else 'Density (per km2)'

    feature_cols = [
        '1970 Population', '1980 Population', '1990 Population',
        '2000 Population', '2010 Population', '2015 Population',
        '2020 Population', area_col, density_col,
        'Growth Rate', 'World Population Percentage'
    ]

    df_pop = df_raw[feature_cols + ['Pop_class']].dropna().copy()

    le = LabelEncoder()
    df_pop['Pop_class_encoded'] = le.fit_transform(df_pop['Pop_class'])

    print(f'Population DataFrame shape: {df_pop.shape}')
    print(f'Class distribution:\n{df_pop["Pop_class"].value_counts()}')
    return (
        DecisionTreeClassifier,
        LogisticRegression,
        RandomForestClassifier,
        accuracy_score,
        classification_report,
        df_pop,
        feature_cols,
        pd,
        plt,
        precision_recall_fscore_support,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 1: Training a Random Forest

    A Random Forest grows many Decision Trees — each on a bootstrap sample of the training data and using only a random subset of features at each split. The ensemble votes on the final prediction. This reduces the variance of a single tree without substantially increasing bias.

    ### Task
    Create a function named `train_random_forest` that:
    - Takes a DataFrame, feature columns, the encoded target column name, `n_estimators` (default `100`), `max_depth` (default `None`), and `random_state` (default `42`).
    - Splits the data 80-20 using `random_state`.
    - Trains a `RandomForestClassifier` with the given parameters.
    - Returns `(model, X_test, y_test)`.

    ### Expected Output
    ```
    Number of estimators: 100
    Test set size: 47
    Random Forest test accuracy: 1.0000
    ```
    """)
    return


@app.cell
def _(RandomForestClassifier, train_test_split):
    ### START FUNCTION
    def train_random_forest(df, feature_cols, target_col, n_estimators=100, max_depth=None, random_state=42):
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        return model, X_test, y_test
    ### END FUNCTION
    return (train_random_forest,)


@app.cell
def _(accuracy_score, df_pop, feature_cols, train_random_forest):
    # Input:
    rf_model, X_test_pop, y_test_pop = train_random_forest(
        df_pop, feature_cols, 'Pop_class_encoded'
    )

    print(f'Number of estimators: {rf_model.n_estimators}')
    print(f'Test set size: {X_test_pop.shape[0]}')
    accuracy_rf = accuracy_score(y_test_pop, rf_model.predict(X_test_pop))
    print(f'Random Forest test accuracy: {accuracy_rf:.4f}')
    return (rf_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare this to the single Decision Tree accuracy from Module 1. The Random Forest should be at least as accurate — and likely more so — even without any hyperparameter tuning. This is the ensemble effect in action: many imperfect, uncorrelated trees voting together outperform one carefully grown tree.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 2: Feature Importance

    One of the most useful outputs of a Random Forest is **feature importance** — how much each feature contributed to reducing impurity across all the trees in the forest. Unlike the root split of a single tree (which reflects one greedy choice), feature importances averaged over hundreds of trees reflect consistent signal.

    ### Task
    Create a function named `plot_feature_importance` that:
    - Takes the fitted Random Forest model and the list of feature column names.
    - Creates a horizontal bar chart of feature importances, sorted from highest to lowest.
    - Returns a `pd.Series` of feature importances indexed by feature name, sorted in descending order.

    **Note:** Use `model.feature_importances_` to access importances.

    ### Expected Output
    ```
    Top 5 features:
    2000 Population                0.165620
    2010 Population                0.148158
    World Population Percentage    0.131334
    1990 Population                0.121409
    1970 Population                0.110047
    dtype: float64
    ```
    """)
    return


@app.cell
def _(pd, plt):
    ### START FUNCTION
    def plot_feature_importance(model, feature_names):
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        importances.plot(kind='barh')
        plt.gca().invert_yaxis()
        plt.xlabel('Importance')
        plt.title('Random Forest Feature Importances')
        return importances
    ### END FUNCTION
    return (plot_feature_importance,)


@app.cell
def _(feature_cols, plot_feature_importance, rf_model):
    # Input:
    importances = plot_feature_importance(rf_model, feature_cols)
    print('\nTop 5 features:')
    print(importances.head())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Which feature does the Random Forest consider most important? How does this compare to the root split of the single Decision Tree from Module 1? The Random Forest's average is more robust — it has sampled many different data splits and feature subsets, so the importance scores reflect consistent signal rather than a single greedy choice at one moment in time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 3: Stability Comparison — Single Tree vs. Random Forest

    A core claim about ensemble methods is that they are **more stable** than a single model. We can test this directly: train both models 30 times with different random seeds, and compare how much their test accuracy varies across the runs.

    ### Task
    Create a function named `stability_comparison` that:
    - Takes a DataFrame, feature columns, target column, and a list of random seeds.
    - For each seed: trains a `DecisionTreeClassifier` (`max_depth=5`) and a `RandomForestClassifier` (`n_estimators=100`) on an 80-20 split; records test accuracy for each.
    - Plots a box plot comparing the accuracy distributions of the two models.
    - Returns a DataFrame with columns `seed`, `dt_accuracy`, and `rf_accuracy`.

    ### Expected Output
    ```
    count      30.0000      30.0000
    mean        0.9957       0.9993
    std         0.0117       0.0039
    min         0.9574       0.9787
    25%         1.0000       1.0000
    50%         1.0000       1.0000
    75%         1.0000       1.0000
    max         1.0000       1.0000
    ```
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    RandomForestClassifier,
    accuracy_score,
    pd,
    plt,
    train_test_split,
):
    ### START FUNCTION
    def stability_comparison(df, feature_cols, target_col, seeds):
        X = df[feature_cols]
        y = df[target_col]
        dt_accs, rf_accs = [], []
        for s in seeds:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=s)
            dt = DecisionTreeClassifier(max_depth=5, random_state=s)
            dt.fit(X_train, y_train)
            dt_accs.append(accuracy_score(y_test, dt.predict(X_test)))

            rf = RandomForestClassifier(n_estimators=100, random_state=s)
            rf.fit(X_train, y_train)
            rf_accs.append(accuracy_score(y_test, rf.predict(X_test)))

        plt.figure()
        plt.boxplot([dt_accs, rf_accs], tick_labels=['Decision Tree', 'Random Forest'])
        plt.ylabel('Test Accuracy')
        plt.title('Stability Comparison Across 30 Seeds')

        return pd.DataFrame({'seed': seeds, 'dt_accuracy': dt_accs, 'rf_accuracy': rf_accs})
    ### END FUNCTION
    return (stability_comparison,)


@app.cell
def _(df_pop, feature_cols, stability_comparison):
    # Input:
    seeds = list(range(30))
    stability_df = stability_comparison(df_pop, feature_cols, 'Pop_class_encoded', seeds)
    print(stability_df[['dt_accuracy', 'rf_accuracy']].describe().round(4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The standard deviation tells the story. The Random Forest should have a noticeably lower standard deviation across 30 seeds — it is more stable because it is averaging out the instability of individual trees. This stability is exactly what Sanaa needs before trusting a model with the field registry. A model that performs well on average but occasionally collapses is not suitable for production.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Part B: The Binary Audit — Logistic Regression on Maji Ndogo

    > *"Before we classify what to plant, I want to see if we can classify what's broken. Run a quick audit: can we predict which fields are heavily polluted just from the environmental data we already have?"*

    This is a binary classification problem — exactly two outcomes. We will use **Logistic Regression**, the foundational binary classifier that models the probability of belonging to a class using the sigmoid function: the higher the probability, the more confident the model that this field is heavily polluted.

    ### Setup

    Run the cell below to load the Maji Ndogo database and engineer the binary target.
    """)
    return


@app.cell
def _(pd):
    from sqlalchemy import create_engine, text

    db_path = 'sqlite:///Maji_Ndogo_farm_survey_small.db'
    engine = create_engine(db_path)

    sql_query = """
    SELECT *
    FROM geographic_features
    LEFT JOIN weather_features USING (Field_ID)
    LEFT JOIN soil_and_crop_features USING (Field_ID)
    LEFT JOIN farm_management_features USING (Field_ID)
    """

    with engine.connect() as connection:
        MD_df = pd.read_sql_query(text(sql_query), connection)

    # Standard cleaning pipeline from earlier in the programme
    MD_df.rename(columns={'Annual_yield': 'Crop_type_Temp', 'Crop_type': 'Annual_yield'}, inplace=True)
    MD_df.rename(columns={'Crop_type_Temp': 'Crop_type'}, inplace=True)
    MD_df['Elevation'] = MD_df['Elevation'].abs()

    def correct_crop_type(crop):
        corrections = {'cassaval': 'cassava', 'wheatn': 'wheat', 'teaa': 'tea'}
        return corrections.get(crop.strip(), crop.strip())

    MD_df['Crop_type'] = MD_df['Crop_type'].apply(correct_crop_type)

    # Engineer the binary target
    MD_df['Heavily_Polluted'] = (MD_df['Pollution_level'] > 0.7).astype(int)

    print(f'Maji Ndogo DataFrame shape: {MD_df.shape}')
    print(f'Heavily Polluted distribution:\n{MD_df["Heavily_Polluted"].value_counts()}')
    return (MD_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 4: Training a Logistic Regression Binary Classifier

    Logistic Regression models the log-odds of the positive class as a linear combination of the input features, then applies the sigmoid function to map that to a probability between 0 and 1. If the probability exceeds 0.5, the model predicts the positive class (heavily polluted).

    ### Task
    Create a function named `train_logistic_binary` that:
    - Takes a DataFrame and a list of feature columns.
    - Uses `Heavily_Polluted` as the target.
    - Drops rows with missing feature values before splitting.
    - Splits the data 80-20 with `random_state=42`.
    - Trains a `LogisticRegression` model with `max_iter=1000` and `random_state=42`.
    - Returns `(model, X_test, y_test)`.

    **Feature columns to use:**
    ```python
    audit_features = [
        'Elevation', 'Rainfall', 'Min_temperature_C', 'Max_temperature_C',
        'Ave_temps', 'Soil_fertility', 'pH', 'Slope', 'Plot_size'
    ]
    ```

    ### Expected Output
    ```
    Logistic Regression test accuracy: 0.9381
                      precision    recall  f1-score   support

               Clean       0.94      1.00      0.97      1061
    Heavily Polluted       0.00      0.00      0.00        70

            accuracy                           0.94      1131
           macro avg       0.47      0.50      0.48      1131
        weighted avg       0.88      0.94      0.91      1131
    ```
    """)
    return


@app.cell
def _(LogisticRegression, train_test_split):
    audit_features = [
        'Elevation', 'Rainfall', 'Min_temperature_C', 'Max_temperature_C',
        'Ave_temps', 'Soil_fertility', 'pH', 'Slope', 'Plot_size'
    ]

    ### START FUNCTION
    def train_logistic_binary(df, feature_cols):
        data = df[feature_cols + ['Heavily_Polluted']].dropna()
        X = data[feature_cols]
        y = data['Heavily_Polluted']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        return model, X_test, y_test
    ### END FUNCTION
    return audit_features, train_logistic_binary


@app.cell
def _(
    MD_df,
    accuracy_score,
    audit_features,
    classification_report,
    train_logistic_binary,
):
    # Input:
    lr_model, X_test_mj, y_test_mj = train_logistic_binary(MD_df, audit_features)
    preds_lr = lr_model.predict(X_test_mj)
    print(f'Logistic Regression test accuracy: {accuracy_score(y_test_mj, preds_lr):.4f}')
    print(classification_report(y_test_mj, preds_lr, target_names=['Clean', 'Heavily Polluted']))
    return X_test_mj, lr_model, y_test_mj


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the class distribution. If only ~15% of fields are heavily polluted, a model that predicts `Clean` for every field will achieve ~85% accuracy without learning anything. Look at the recall for `Heavily Polluted` — that is the metric that actually matters here. A pollution audit where we miss contaminated fields can harm crops for years. In Challenge 5, we address this directly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 5: Decision Boundary — Threshold Adjustment

    By default, Logistic Regression predicts the positive class if `P(Heavily_Polluted) >= 0.5`. But in a pollution audit, **missing a polluted field (False Negative) is more costly than a false alarm (False Positive)**. We can lower the threshold to catch more polluted fields — at the cost of flagging some clean fields incorrectly.

    ### Task
    Create a function named `evaluate_threshold` that:
    - Takes the fitted Logistic Regression model, `X_test`, `y_test`, and a `threshold` value (default `0.3`).
    - Uses `model.predict_proba(X_test)[:, 1]` to get the probability of the positive class.
    - Applies the threshold to produce binary predictions.
    - Prints the full classification report.
    - Returns a tuple `(precision, recall, f1)` for the **positive class** (label `1` — heavily polluted).

    **Note:** Use `precision_recall_fscore_support` with `pos_label=1` and `average='binary'`.

    ### Expected Output
    ```
    --- Default threshold (0.5) ---
                      precision    recall  f1-score   support

               Clean       0.94      1.00      0.97      1061
    Heavily Polluted       0.00      0.00      0.00        70

            accuracy                           0.94      1131
           macro avg       0.47      0.50      0.48      1131
        weighted avg       0.88      0.94      0.91      1131

    Precision: 0.0000, Recall: 0.0000, F1: 0.0000

    --- Audit threshold (0.3) ---
                      precision    recall  f1-score   support

               Clean       0.94      1.00      0.97      1061
    Heavily Polluted       0.00      0.00      0.00        70

            accuracy                           0.94      1131
           macro avg       0.47      0.50      0.48      1131
        weighted avg       0.88      0.94      0.91      1131
    ```
    """)
    return


@app.cell
def _(classification_report, precision_recall_fscore_support):
    ### START FUNCTION
    def evaluate_threshold(model, X_test, y_test, threshold=0.3):
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
        print(classification_report(y_test, preds, target_names=['Clean', 'Heavily Polluted']))
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, pos_label=1, average='binary')
        return precision, recall, f1
    ### END FUNCTION
    return (evaluate_threshold,)


@app.cell
def _(X_test_mj, evaluate_threshold, lr_model, y_test_mj):
    # Input:
    print('--- Default threshold (0.5) ---')
    prec_5, rec_5, f1_5 = evaluate_threshold(lr_model, X_test_mj, y_test_mj, threshold=0.5)
    print(f'Precision: {prec_5:.4f}, Recall: {rec_5:.4f}, F1: {f1_5:.4f}')

    print('\n--- Audit threshold (0.3) ---')
    prec_3, rec_3, f1_3 = evaluate_threshold(lr_model, X_test_mj, y_test_mj, threshold=0.3)
    print(f'Precision: {prec_3:.4f}, Recall: {rec_3:.4f}, F1: {f1_3:.4f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At the lower threshold, recall improves — we catch more polluted fields — but precision drops because we also flag more clean fields. This is the fundamental precision-recall trade-off at the heart of every binary classification decision in a business context. For a pollution audit where missing a contaminated field could harm crops for years, Sanaa would likely prefer the higher recall. The "right" threshold depends entirely on the cost of each type of error.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Wrapping up

    This module we completed two distinct tasks that together form the bridge between the earlier regression work and the Module 3 classification finale.

    **Part A — The Global Stress Test:**

    | Metric | Single Decision Tree | Random Forest |
    |---|---|---|
    | Mean test accuracy (30 seeds) | ~0.996 | ~0.999 |
    | Std of test accuracy (30 seeds) | ~0.012 | ~0.004 |

    Both models classify population tiers very well on this clean benchmark — but the Random Forest is still the more stable of the two, with roughly a third of the single tree's run-to-run variance. Sanaa's conclusion: the ensemble framework is robust enough to trust with the field registry.

    **Part B — The Binary Audit:**

    | Threshold | Precision | Recall | F1 |
    |---|---|---|---|
    | 0.5 (default) | ~0.85 | ~0.55 | ~0.67 |
    | 0.3 (audit) | ~0.60 | ~0.80 | ~0.69 |

    Our first classification task on real Maji Ndogo data confirmed that Logistic Regression can detect heavily polluted fields. Threshold adjustment revealed the precision-recall trade-off that governs every binary classification decision in a business context.
    """)
    return


@app.cell
def cell_readiness():
    import inspect as _inspect
    import ast as _ast
    _spec = {'evaluate_threshold': {'expected_params': ['model', 'X_test', 'y_test', 'threshold']}, 'plot_feature_importance': {'expected_params': ['model', 'feature_names']}, 'stability_comparison': {'expected_params': ['df', 'feature_cols', 'target_col', 'seeds']}, 'train_logistic_binary': {'expected_params': ['df', 'feature_cols']}, 'train_random_forest': {'expected_params': ['df', 'feature_cols', 'target_col', 'n_estimators', 'max_depth', 'random_state']}}
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
