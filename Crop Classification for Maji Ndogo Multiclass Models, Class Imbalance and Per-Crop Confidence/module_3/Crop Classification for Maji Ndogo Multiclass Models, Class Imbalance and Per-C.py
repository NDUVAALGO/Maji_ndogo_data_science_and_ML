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
    # The Maji Ndogo Classification Finale
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
    For three modules you have been building toward a single tool. You benchmarked a lone Decision Tree until you trusted it, stress-tested a Random Forest until you trusted it *more*, and built your first binary classifier on real Maji Ndogo field data. Each was a rehearsal. This is the performance.

    The brief comes, as it always does, from **Sanaa Lewis** at the Ministry of Agriculture:

    > *"The fields hold a wealth of geographic and weather data, but the link between those numbers and what actually grows well is too tangled for a person to decode by hand. I want to point at any field in the registry — its elevation, rainfall, temperature, soil, pH, pollution — and ask one question: **what should we plant here?** And I want the model to tell me not just its answer, but how sure it is. A recommendation I can't gauge my confidence in is worse than no recommendation at all."*

    That request defines a **multiclass classification** problem, and this project turns the Maji Ndogo Digital Twin into a **Decision Support System** that answers it, in six movements:

    1. **Feature selection and encoding** — assemble the environmental feature set and encode the crop target.
    2. **A multiclass Logistic Regression** — a linear baseline across all crop classes.
    3. **The confusion matrix** — not just *how often* the model is wrong, but *what it confuses with what*.
    4. **Class imbalance** — some crops occupy far more fields than others; you'll upsample the minority classes and see the metrics respond.
    5. **Model comparison** — Logistic Regression versus Random Forest on the balanced data.
    6. **A confidence report** — the deliverable Sanaa actually asked for: which crops the model recommends with high confidence, and which need a human's second look.

    By the end you'll have a working crop Decision Support System that runs the full arc — from raw database to an honest, per-class confidence report a decision-maker can act on. Let's get to work. 🌾

    **Scope note.** This project stays strictly within the Module 3 toolkit: multiclass classification, the confusion matrix and classification report, precision/recall/F1, class-imbalance handling by resampling, and Logistic-Regression-versus-Random-Forest model selection. No hyperparameter grid search or feature-subset selection is required here — the focus is metrics and imbalance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Data Dictionary

    | Column | Description | Type |
    |--------|-------------|------|
    | `Field_ID` | Unique field identifier | BigInt |
    | `Elevation` | Field elevation above sea level in meters | Float |
    | `Latitude` / `Longitude` | Geographic coordinates in degrees | Float |
    | `Location` | Province name | Text |
    | `Slope` | Slope of the terrain | Float |
    | `Rainfall` | Annual rainfall in mm | Float |
    | `Min_temperature_C` / `Max_temperature_C` / `Ave_temps` | Temperatures in Celsius | Float |
    | `Soil_fertility` | Soil fertility score, 0–1 | Float |
    | `Soil_type` | Soil type category | Text |
    | `pH` | Soil pH level | Float |
    | `Pollution_level` | Pollution score, 0–1 | Float |
    | `Plot_size` | Field plot size in hectares | Float |
    | **`Crop_type`** | The crop chosen for the field — our **target variable** | Text |
    | `Standard_yield` | Standardized yield normalized per crop | Float |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Setup

    Load the Maji Ndogo data using the cleaning pipeline established earlier in the programme (reproduced below). Note the column swap — in this survey the `Crop_type` and `Annual_yield` columns are stored in each other's place, so we swap them back before anything else.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sqlalchemy import create_engine, text
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, classification_report,
        confusion_matrix, ConfusionMatrixDisplay
    )
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.utils import resample
    import warnings
    warnings.filterwarnings('ignore')

    # --- Load data ---
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

    # --- Standard cleaning pipeline from earlier in the programme ---
    MD_df.rename(columns={'Annual_yield': 'Crop_type_Temp', 'Crop_type': 'Annual_yield'}, inplace=True)
    MD_df.rename(columns={'Crop_type_Temp': 'Crop_type'}, inplace=True)
    MD_df['Elevation'] = MD_df['Elevation'].abs()

    def correct_crop_type(crop):
        corrections = {'cassaval': 'cassava', 'wheatn': 'wheat', 'teaa': 'tea'}
        return corrections.get(str(crop).strip(), str(crop).strip())

    MD_df['Crop_type'] = MD_df['Crop_type'].apply(correct_crop_type)

    print(f"DataFrame shape: {MD_df.shape}")
    print(f"\nCrop type distribution:")
    print(MD_df['Crop_type'].value_counts())
    return (
        ConfusionMatrixDisplay,
        LabelEncoder,
        LogisticRegression,
        MD_df,
        RandomForestClassifier,
        accuracy_score,
        classification_report,
        confusion_matrix,
        pd,
        plt,
        resample,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    DataFrame shape: (5654, 18)

    Crop type distribution:
    Crop_type
    wheat      1342
    tea         913
    potato      823
    cassava     672
    banana      633
    coffee      607
    maize       399
    rice        265
    Name: count, dtype: int64
    ```
    Eight crops, and a clear imbalance: wheat covers over five times as many fields as rice. That imbalance is the thread running through the whole project — we meet it head-on in Challenge 4.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Analysis

    ## Challenge 1: Feature Selection and Encoding

    Before modeling, we choose our features and prepare the target. We use the environmental and geographic features Sanaa has curated over earlier courses, and encode the crop target as numeric labels.

    ### Task
    Create a function `prepare_crop_features` that:
    - Takes the full Maji Ndogo DataFrame.
    - Selects these feature columns: `Elevation`, `Rainfall`, `Min_temperature_C`, `Max_temperature_C`, `Ave_temps`, `Soil_fertility`, `pH`, `Pollution_level`, `Slope`, `Plot_size`.
    - Drops rows with any missing values in the selected columns or the target `Crop_type`.
    - Encodes `Crop_type` using a `LabelEncoder` into a new column `Crop_encoded`.
    - Returns `(df_prepared, feature_cols, le)`, where `df_prepared` contains the feature columns **and** `Crop_encoded`, `feature_cols` is the feature-name list, and `le` is the fitted `LabelEncoder`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    Prepared DataFrame shape: (5654, 11)
    Feature columns: ['Elevation', 'Rainfall', 'Min_temperature_C', 'Max_temperature_C', 'Ave_temps', 'Soil_fertility', 'pH', 'Pollution_level', 'Slope', 'Plot_size']
    Crop classes: ['banana', 'cassava', 'coffee', 'maize', 'potato', 'rice', 'tea', 'wheat']

    Encoded target distribution:
    Crop_encoded
    wheat      1342
    tea         913
    potato      823
    cassava     672
    banana      633
    coffee      607
    maize       399
    rice        265
    Name: count, dtype: int64
    ```
    """)
    return


@app.cell
def _(LabelEncoder):
    ### START FUNCTION
    def prepare_crop_features(df):
        feature_cols = ['Elevation', 'Rainfall', 'Min_temperature_C', 'Max_temperature_C',
                         'Ave_temps', 'Soil_fertility', 'pH', 'Pollution_level', 'Slope', 'Plot_size']
        data = df[feature_cols + ['Crop_type']].dropna().copy()
        le = LabelEncoder()
        data['Crop_encoded'] = le.fit_transform(data['Crop_type'])
        df_prepared = data[feature_cols + ['Crop_encoded']].copy()
        return df_prepared, feature_cols, le
    ### END FUNCTION
    return (prepare_crop_features,)


@app.cell
def _(MD_df, prepare_crop_features):
    df_prepared, feature_cols, le = prepare_crop_features(MD_df)
    print(f"Prepared DataFrame shape: {df_prepared.shape}")
    print(f"Feature columns: {feature_cols}")
    print(f"Crop classes: {list(le.classes_)}")
    print(f"\nEncoded target distribution:")
    print(df_prepared['Crop_encoded'].value_counts().rename(index=dict(enumerate(le.classes_))))
    return df_prepared, feature_cols, le


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note the class distribution is uneven — wheat and tea dominate, rice is scarce. That is class imbalance, and it will shape how we read every metric until we correct it in Challenge 4.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 2: Training a Multiclass Logistic Regression Classifier

    Logistic Regression handles multiclass problems by fitting one classifier per class (one-vs-rest) or by extending to a softmax across all classes at once; `sklearn` manages this automatically.

    ### Task
    Create a function `train_crop_classifier` that:
    - Takes `df_prepared`, `feature_cols`, `random_state` (default `42`), and `model_type` (`'logistic'` or `'random_forest'`).
    - Splits the data 80–20 (features = `feature_cols`, target = `Crop_encoded`), using `random_state`.
    - Trains the selected model: `LogisticRegression(max_iter=1000)` or `RandomForestClassifier(n_estimators=100, random_state=random_state)`.
    - Returns `(model, X_test, y_test)`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    Logistic Regression accuracy: 0.4783

                  precision    recall  f1-score   support

          banana       0.38      0.40      0.39       126
         cassava       0.33      0.29      0.31       108
          coffee       0.48      0.31      0.37       121
           maize       0.00      0.00      0.00        69
          potato       0.48      0.61      0.54       173
            rice       0.38      0.22      0.28        49
             tea       0.65      0.81      0.73       198
           wheat       0.44      0.51      0.47       287

        accuracy                           0.48      1131
       macro avg       0.39      0.39      0.39      1131
    weighted avg       0.44      0.48      0.45      1131
    ```
    """)
    return


@app.cell
def _(LogisticRegression, RandomForestClassifier, train_test_split):
    ### START FUNCTION
    def train_crop_classifier(df_prepared, feature_cols, random_state=42, model_type='logistic'):
        X = df_prepared[feature_cols]
        y = df_prepared['Crop_encoded']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        if model_type == 'logistic':
            model = LogisticRegression(max_iter=1000)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_train, y_train)
        return model, X_test, y_test
    ### END FUNCTION
    return (train_crop_classifier,)


@app.cell
def _(
    accuracy_score,
    classification_report,
    df_prepared,
    feature_cols,
    le,
    train_crop_classifier,
):
    lr_crop, X_test_crop, y_test_crop = train_crop_classifier(
        df_prepared, feature_cols, model_type='logistic')

    preds_lr = lr_crop.predict(X_test_crop)
    print(f"Logistic Regression accuracy: {accuracy_score(y_test_crop, preds_lr):.4f}")
    print()
    print(classification_report(y_test_crop, preds_lr, target_names=le.classes_))
    return X_test_crop, lr_crop, y_test_crop


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Eight crops from only environmental features is a demanding problem for a linear model — the accuracy is modest, and some crops are predicted far better than others. Watch which crops score well: those have the clearest environmental signature. The confusion matrix in Challenge 3 shows what the model mixes up.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 3: The Confusion Matrix — Understanding Crop Misclassification

    A confusion matrix shows exactly which classes the model confuses with each other — in a multiclass problem, that is far more informative than accuracy alone. It tells Sanaa not just *how often* the model errs, but *what it mistakes for what*.

    ### Task
    Create a function `plot_confusion_matrix` that:
    - Takes a fitted `model`, `X_test`, `y_test`, and a list of `class_names`.
    - Generates predictions.
    - Plots the confusion matrix using `ConfusionMatrixDisplay` with `cmap='Blues'` (create a `plt.figure(figsize=(12, 10))` first).
    - Returns the confusion matrix as a NumPy array (from `confusion_matrix`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    A Confusion Matrix image followed by:
    ```
    Confusion matrix shape: (8, 8)
    ```
    """)
    return


@app.cell
def _(ConfusionMatrixDisplay, confusion_matrix, plt):
    ### START FUNCTION
    def plot_confusion_matrix(model, X_test, y_test, class_names):
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues')
        return cm
    ### END FUNCTION
    return (plot_confusion_matrix,)


@app.cell
def _(X_test_crop, le, lr_crop, plot_confusion_matrix, y_test_crop):
    cm = plot_confusion_matrix(lr_crop, X_test_crop, y_test_crop, list(le.classes_))
    print(f"\nConfusion matrix shape: {cm.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at the largest off-diagonal cells: those are the crop pairs the model confuses. Crops that share environmental conditions — similar pH, rainfall, temperature — are the ones it struggles to separate. This is where the biological reality of Maji Ndogo shows through the data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 4: Addressing Class Imbalance — Upsampling

    When the target is imbalanced, a model tends to over-predict the majority class — inflating accuracy while quietly failing on minority crops. One fix is **upsampling**: resample the minority classes with replacement until every class matches the majority-class count.

    ### Task
    Create a function `upsample_minority_classes` that:
    - Takes `df_prepared` and `feature_cols`.
    - Identifies the majority class (the most frequent `Crop_encoded` value) and its count `n`.
    - Upsamples every other class to `n` rows using `sklearn.utils.resample` with `replace=True` and `random_state=42`.
    - Concatenates all classes into one balanced DataFrame and returns it.
    - Prints the class counts before and after balancing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output 1
    ```
    Before balancing:
    Crop_encoded
    0     633
    1     672
    2     607
    3     399
    4     823
    5     265
    6     913
    7    1342
    Name: count, dtype: int64

    After balancing:
    Crop_encoded
    0    1342
    1    1342
    2    1342
    3    1342
    4    1342
    5    1342
    6    1342
    7    1342
    Name: count, dtype: int64

    Balanced DataFrame shape: (10736, 11)

    Class distribution after balancing:
    Crop_encoded
    cassava    1342
    tea        1342
    wheat      1342
    potato     1342
    banana     1342
    coffee     1342
    rice       1342
    maize      1342
    Name: count, dtype: int64
    ```

    ### Expected output 2
    ```
    Balanced model accuracy: 0.4255

                  precision    recall  f1-score   support

          banana       0.33      0.32      0.32       261
         cassava       0.44      0.36      0.40       265
          coffee       0.45      0.34      0.39       266
           maize       0.40      0.39      0.39       271
          potato       0.39      0.45      0.42       284
            rice       0.57      0.67      0.62       264
             tea       0.53      0.74      0.62       266
           wheat       0.19      0.14      0.16       271

        accuracy                           0.43      2148
       macro avg       0.41      0.43      0.41      2148
    weighted avg       0.41      0.43      0.41      2148
    ```
    """)
    return


@app.cell
def _(pd, resample):
    ### START FUNCTION
    def upsample_minority_classes(df_prepared, feature_cols):
        counts = df_prepared['Crop_encoded'].value_counts()
        print('Before balancing:')
        print(df_prepared['Crop_encoded'].value_counts().sort_index())

        n = counts.max()
        balanced_parts = []
        for cls, group in df_prepared.groupby('Crop_encoded'):
            if len(group) < n:
                group = resample(group, replace=True, n_samples=n, random_state=42)
            balanced_parts.append(group)

        df_balanced = pd.concat(balanced_parts).reset_index(drop=True)
        print('\nAfter balancing:')
        print(df_balanced['Crop_encoded'].value_counts().sort_index())
        return df_balanced
    ### END FUNCTION
    return (upsample_minority_classes,)


@app.cell
def _(df_prepared, feature_cols, le, upsample_minority_classes):
    df_balanced = upsample_minority_classes(df_prepared, feature_cols)
    print(f"\nBalanced DataFrame shape: {df_balanced.shape}")
    print("\nClass distribution after balancing:")
    print(df_balanced['Crop_encoded'].value_counts().rename(index=dict(enumerate(le.classes_))))
    return (df_balanced,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Expected output:** every class upsampled to the majority count (wheat, **1342**), giving a balanced frame of shape **(10736, 11)** — 8 classes × 1342. The retrain cell below shows the effect on the metrics.
    """)
    return


@app.cell
def _(
    accuracy_score,
    classification_report,
    df_balanced,
    feature_cols,
    le,
    train_crop_classifier,
):
    # Retrain logistic regression on the balanced data and compare to Challenge 2.
    lr_balanced, X_test_b, y_test_b = train_crop_classifier(
        df_balanced, feature_cols, model_type='logistic')
    preds_b = lr_balanced.predict(X_test_b)
    print("Balanced model accuracy:", round(accuracy_score(y_test_b, preds_b), 4))
    print()
    print(classification_report(y_test_b, preds_b, target_names=le.classes_))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After balancing, overall accuracy stays close to the imbalanced model, but the per-class recall on the previously-scarce crops changes — the model is no longer able to coast on predicting wheat. For a real deployment, balanced per-class performance is usually the better outcome, even at the cost of a little headline accuracy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 5: Selecting the Best Model — Logistic Regression vs. Random Forest

    Having tried a linear model, let's see whether a Random Forest does better on this eight-class problem. Its ensemble of trees can capture the non-linear interactions between rainfall, soil, and crop suitability that a single linear boundary cannot.

    ### Task
    Create a function `compare_classifiers` that:
    - Takes `df_balanced`, `feature_cols`, and the `LabelEncoder` `le`.
    - Trains both a `LogisticRegression(max_iter=1000)` and a `RandomForestClassifier(n_estimators=100, random_state=42)` on an 80–20 split (`random_state=42`).
    - Prints the classification report for each.
    - Returns a DataFrame with columns `model`, `accuracy`, `macro_f1` — one row per model.

    > Use `classification_report(..., output_dict=True)['macro avg']['f1-score']` to extract macro F1 programmatically.

    ### Expected output
    (values near):
    ```

    ===== Logistic Regression =====
                  precision    recall  f1-score   support

          banana       0.33      0.32      0.32       261
         cassava       0.44      0.36      0.40       265
          coffee       0.45      0.34      0.39       266
           maize       0.40      0.39      0.39       271
          potato       0.39      0.45      0.42       284
            rice       0.57      0.67      0.62       264
             tea       0.53      0.74      0.62       266
           wheat       0.19      0.14      0.16       271

        accuracy                           0.43      2148
       macro avg       0.41      0.43      0.41      2148
    weighted avg       0.41      0.43      0.41      2148

    ===== Random Forest =====
                  precision    recall  f1-score   support

          banana       0.85      0.87      0.86       261
         cassava       0.81      0.88      0.85       265
          coffee       0.86      0.93      0.89       266
           maize       0.82      0.97      0.89       271
          potato       0.80      0.85      0.82       284
            rice       0.87      1.00      0.93       264
             tea       0.94      0.89      0.92       266
           wheat       0.63      0.30      0.41       271

        accuracy                           0.84      2148
       macro avg       0.82      0.84      0.82      2148
    weighted avg       0.82      0.84      0.82      2148

    Model comparison:
                     model  accuracy  macro_f1
    0  Logistic Regression    0.4255    0.4143
    1        Random Forest    0.8352    0.8206
    ```
    """)
    return


@app.cell
def _(
    LogisticRegression,
    RandomForestClassifier,
    accuracy_score,
    classification_report,
    pd,
    train_test_split,
):
    ### START FUNCTION
    def compare_classifiers(df_balanced, feature_cols, le):
        X = df_balanced[feature_cols]
        y = df_balanced['Crop_encoded']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = []
        for name, model in [('Logistic Regression', LogisticRegression(max_iter=1000)),
                             ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42))]:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            print(f'\n===== {name} =====')
            print(classification_report(y_test, preds, target_names=le.classes_))
            report = classification_report(y_test, preds, output_dict=True)
            results.append({'model': name, 'accuracy': accuracy_score(y_test, preds),
                             'macro_f1': report['macro avg']['f1-score']})

        return pd.DataFrame(results)
    ### END FUNCTION
    return (compare_classifiers,)


@app.cell
def _(compare_classifiers, df_balanced, feature_cols, le):
    comparison_df = compare_classifiers(df_balanced, feature_cols, le)
    print("\nModel comparison:")
    print(comparison_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Random Forest decisively outperforms Logistic Regression here, because the relationship between environmental features and crop suitability is non-linear: temperature and rainfall interact with pH and soil fertility in ways a straight decision boundary cannot capture, but a forest of trees can.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 6: Telling Sanaa the Story — Precision, Recall, and Crop Confidence

    The final step turns metrics into something Sanaa can act on: which crops does the best model recommend with the highest confidence, and which deserve caution?

    ### Task
    Create a function `crop_confidence_report` that:
    - Takes the best fitted `model` (the Random Forest), `X_test`, `y_test`, and the `LabelEncoder` `le`.
    - Generates predictions and extracts the classification report as a dictionary.
    - Builds a DataFrame with columns `crop`, `precision`, `recall`, `f1_score`, `support` (filter out the `accuracy`, `macro avg`, and `weighted avg` summary rows).
    - Sorts by `f1_score` descending.
    - Plots a horizontal bar chart of F1 score by crop, with labeled axes and a title.
    - Returns the sorted DataFrame.

    ### Expected output
    (F1 descending; exact values vary a little):
    ```
    crop  precision  recall  f1_score  support
       rice       0.87    1.00      0.93      264
        tea       0.94    0.89      0.92      266
     coffee       0.86    0.93      0.89      266
      maize       0.82    0.97      0.89      271
     banana       0.85    0.87      0.86      261
    cassava       0.81    0.88      0.85      265
     potato       0.80    0.85      0.82      284
      wheat       0.63    0.30      0.41      271
    ```
    """)
    return


@app.cell
def _(classification_report, pd, plt):
    ### START FUNCTION
    def crop_confidence_report(model, X_test, y_test, le):
        preds = model.predict(X_test)
        report = classification_report(y_test, preds, target_names=le.classes_, output_dict=True)

        rows = []
        for crop, metrics in report.items():
            if crop in ('accuracy', 'macro avg', 'weighted avg'):
                continue
            rows.append({'crop': crop, 'precision': metrics['precision'], 'recall': metrics['recall'],
                         'f1_score': metrics['f1-score'], 'support': metrics['support']})

        report_df = pd.DataFrame(rows).sort_values('f1_score', ascending=False).reset_index(drop=True)

        plt.figure(figsize=(10, 6))
        plt.barh(report_df['crop'], report_df['f1_score'])
        plt.gca().invert_yaxis()
        plt.xlabel('F1 Score')
        plt.title('Crop Classification Confidence (F1 Score by Crop)')

        return report_df
    ### END FUNCTION
    return (crop_confidence_report,)


@app.cell
def _(
    crop_confidence_report,
    df_balanced,
    feature_cols,
    le,
    train_crop_classifier,
):
    rf_best, X_test_rf, y_test_rf = train_crop_classifier(
        df_balanced, feature_cols, model_type='random_forest')

    report_df = crop_confidence_report(rf_best, X_test_rf, y_test_rf, le)
    print(report_df.to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This table is Sanaa's answer. The crops with the highest F1 are the ones the model recommends with the most confidence — rice (distinctive high-rainfall need) and tea (acidic-soil preference) sit at the top. She can run a **tiered system**: trust the model directly for high-confidence crops, and flag low-confidence fields for manual review.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Wrapping up

    Over three modules in this course you transformed the Maji Ndogo Digital Twin into a working Decision Support System:

    - **Module 1:** Benchmarked Decision Trees on world population data — recursive splitting, pruning, depth control.
    - **Module 2:** Stress-tested the Random Forest for ensemble stability, and built the first binary classifier on real pollution data, meeting threshold adjustment and the precision–recall trade-off.
    - **Module 3:** Built the full multiclass crop classifier, faced class imbalance with upsampling, compared Logistic Regression against Random Forest, and delivered a per-crop confidence report Sanaa can act on.

    The model is not perfect, and that is the point. It is most confident about crops with a distinctive environmental signature — rice's need for heavy rainfall, tea's preference for acidic soil — and least confident about crops that share overlapping conditions. That uncertainty is not a flaw; it is honesty. A model that tells you when it is unsure is worth more than one that hides its doubt behind false precision. You built exactly that — proof you can take a classification problem from raw rows to a decision a client can trust.

    The lights stay on. The crops get planted. The work continues.

    — Sanaa
    """)
    return


if __name__ == "__main__":
    app.run()
