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
    # The Advanced Agricultural Intelligence System

    **This challenge is not graded.** It is here to build your footing before the graded work begins.

    ### Instructions

    - Do not add or remove cells in this notebook.
    - Answer the questions according to the specifications provided.
    - Use the provided **Expected output** blocks and test cells to verify your work before continuing.
    - Use the tools introduced in this course: pandas, NumPy, scikit-learn, TensorFlow/Keras, Matplotlib, and Seaborn — for advanced classification (SVMs, KNN, ensembles, neural networks), hyperparameter tuning, and model selection.
    - The use of StackOverflow, Google, Generative AI tools, and any other online resources is permitted. Use AI to help you understand — not to shortcut the thinking. [Read the honor code here](https://drive.google.com/file/d/1atFOPUQRLz5slb4Q1ASXh8QQfKyXVqrw/preview).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Ministry of Agriculture in Maji Ndogo has stopped asking whether data science works. Over the past two modules you proved it: a Random Forest that reads handwritten survey digits, an SVC tuned with GridSearchCV that separates tangled chemical signatures. The Ministry trusts you now — and trust, as your contact **Sanaa Lewis** likes to remind you, is the most expensive thing a consultant can spend.

    This module she spends it on the question the whole engagement has been building toward:

    > *"A farmer clears a new field. We know its elevation, its soil, its rainfall, its pH — everything except the one thing that matters. What should they plant? I don't want a hunch. I want a recommendation system, and I want to know **which** model is steady enough to stake a season's harvest on. Show me you didn't just pick the first classifier that worked."*

    That last sentence is the real brief. Any model can score well on one lucky train–test split; a *recommendation system people will actually farm by* has to be chosen rigorously. So this project is built in three movements:

    1. **Data Preparation.** Pull the full farm survey from the database, clean the crop labels, encode the categorical soil type, scale the features, and split. The data is real, which means it is messy — part of your job is noticing that.
    2. **A Neural Network Classifier.** Build and train a feed-forward network in TensorFlow/Keras that predicts crop type from field conditions, and read its training curves to see whether it is learning or memorizing.
    3. **Model Selection.** Put the neural network up against five classical classifiers under **5-fold stratified cross-validation** — the professional standard — then defend a single choice on the held-out test set.

    **Scope note.** This is strictly a classification and model-selection project: data preparation, a Keras neural network, five classical classifiers, stratified cross-validation, and evaluation metrics — exactly the Module 3 toolkit. We use only the structured features already in the survey database.

    > **A note for the future:** the survey also references a network of automated weather stations whose readings sit in raw, multilingual text logs. Parsing those into extra features is a natural next step — and is exactly what a later NLP module takes up. For now, we work with the structured features in hand.
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
    import sqlite3
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    import warnings
    warnings.filterwarnings('ignore')
    return accuracy_score, classification_report, np, pd, plt, sqlite3, tf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Dataset

    The Maji Ndogo farm survey lives in `Maji_Ndogo_farm_survey_small.db`, spread across four tables that share a `Field_ID`. We join them into one row per field carrying terrain, soil, weather, and management features, along with the crop each field is recorded as growing.

    | Source column | Meaning |
    |---------------|---------|
    | `Elevation`, `Slope` | Terrain (geographic_features) |
    | `Plot_size`, `Pollution_level` | Management (farm_management_features) |
    | `pH`, `Soil_fertility`, `Soil_type` | Soil chemistry (soil_and_crop_features) |
    | `Rainfall`, `Min_temperature_C`, `Max_temperature_C`, `Ave_temps` | Weather (weather_features) |
    | `Annual_yield` → **`Crop_type`** | The crop grown — our prediction target |

    > **A real-world wrinkle.** In this survey the recorded crop names live in the `Annual_yield` column, so we alias it to `Crop_type` when we load. And like most hand-entered data, the crop labels are messy — you'll find `'cassava '` with a stray space, `'teaa'`, `'wheatn'`, and similar typos. Cleaning those into a consistent set of crop classes is part of Question 1, not an afterthought.
    """)
    return


@app.cell
def _(pd, sqlite3):
    import os
    db_file = 'Maji_Ndogo_farm_survey_small.db'
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"Database file '{db_file}' not found in the current directory: {os.getcwd()}\nDownload it from the course resources and place it in the same folder as this notebook.")
    conn = sqlite3.connect(db_file)
    # Step 1: Check the file exists on disk
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='geographic_features'")
    if cursor.fetchone() is None:
        conn.close()
        raise ValueError(f"Connected to '{db_file}' but the table 'geographic_features' was not found.\nThis usually means sqlite3 created a new empty file at that path instead of opening the real database. Delete the file, re-download Maji_Ndogo_farm_survey_small.db from the course resources, and place it in the same folder as this notebook.")
    query = '\nSELECT\n    g.Field_ID,\n    g.Elevation,\n    g.Slope,\n    f.Plot_size,\n    f.Pollution_level,\n    s.pH,\n    s.Soil_fertility,\n    s.Soil_type,\n    w.Rainfall,\n    w.Min_temperature_C,\n    w.Max_temperature_C,\n    w.Ave_temps,\n    f.Annual_yield AS Crop_type\nFROM geographic_features        AS g\nLEFT JOIN farm_management_features AS f ON g.Field_ID = f.Field_ID\nLEFT JOIN soil_and_crop_features   AS s ON g.Field_ID = s.Field_ID\nLEFT JOIN weather_features         AS w ON g.Field_ID = w.Field_ID\n'
    # Step 2: Connect and verify the expected tables are present
    # sqlite3.connect() silently creates a new empty file if the path resolves but
    # the file is not a valid Maji Ndogo database — we catch that here.
    farm_df = pd.read_sql_query(query, conn)
    conn.close()
    print('Farm survey shape:', farm_df.shape)
    print('\nRaw crop labels (note the messiness):')
    print(farm_df['Crop_type'].value_counts())
    farm_df.head()
    return (farm_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected Output
    ```
    Farm survey shape: (5654, 13)

    Raw crop labels (note the messiness):
    wheat       1316
    tea          895
    potato       823
    cassava      660
    banana       633
    coffee       607
    maize        399
    rice         265
    wheat         13
    wheatn        13
    ...
    ```
    Notice the duplicated-looking labels (`wheat` vs `wheat ` vs `wheatn`). Eight real crops are hiding behind fourteen raw strings. We fix that next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Part 1: Data Preparation

    Before any classifier, we clean the labels, encode the one categorical feature, scale, and split. This is the same disciplined pipeline from earlier modules — now applied to the full survey.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 1: Cleaning, Encoding, and the Train–Test Split

    The raw survey data has a problem Sanaa spotted immediately: the crop labels are a mess. Field workers entered crop names by hand across different seasons and surveys, and the results include `'wheat '` with a trailing space, `'wheatn'` from a miskeyed entry, and `'teaa'` from a double tap. If we encode these dirty labels as-is, the model will try to learn the difference between `'tea'` and `'teaa'` as if they were genuinely different crops — and it will fail, because they are the same crop.

    Cleaning the labels before encoding is not optional. It is the step that makes the rest of the project valid.

    ### Task

    Write a function `prepare_data` that:
    - Takes the farm DataFrame as input and works on a **copy**.
    - Drops rows with any missing values.
    - **Cleans the crop labels** in `Crop_type`: strip surrounding whitespace, lowercase, and map known typos to the correct crop using this mapping: `{'cassaval': 'cassava', 'teaa': 'tea', 'wheatn': 'wheat'}`. Any label that is still not one of the eight valid crops — `banana, cassava, coffee, maize, potato, rice, tea, wheat` — should have its row dropped.
    - Drops the `Field_ID` identifier column (it is not a feature).
    - Encodes the categorical feature column `Soil_type` using `LabelEncoder`.
    - Separates features (`X`) from the target `Crop_type`.
    - Encodes `Crop_type` with a **separate** `LabelEncoder`; keep this encoder so predictions can be decoded later.
    - Scales `X` using `StandardScaler`.
    - Splits into **80% train / 20% test** with `random_state=42`.
    - Returns `(X_train, X_test, y_train, y_test, label_encoder, feature_names)`, where `feature_names` is the list of feature column names (before scaling).

    > **Why clean first?** If you label-encode before cleaning, `'tea'` and `'teaa'` become two different classes and your model is quietly trying to learn a typo. Cleaning collapses them back to eight real crops.

    ### Expected Output
    ```
    Training set: (4523, 11)
    Test set: (1131, 11)
    Number of crop classes: 8
    Crop classes: ['banana', 'cassava', 'coffee', 'maize', 'potato', 'rice', 'tea', 'wheat']
    Feature names: ['Elevation', 'Slope', 'Plot_size', 'Pollution_level', 'pH', 'Soil_fertility', 'Soil_type', 'Rainfall', 'Min_temperature_C', 'Max_temperature_C', 'Ave_temps']
    ```
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def prepare_data(df):
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.model_selection import train_test_split

        data = df.copy()

        # Drop any row with a missing value in any column
        data = data.dropna()

        # Clean the crop labels: strip whitespace, lowercase, fix known typos
        typo_map = {'cassaval': 'cassava', 'teaa': 'tea', 'wheatn': 'wheat'}
        valid_crops = {'banana', 'cassava', 'coffee', 'maize',
                        'potato', 'rice', 'tea', 'wheat'}

        data['Crop_type'] = data['Crop_type'].str.strip().str.lower()
        data['Crop_type'] = data['Crop_type'].replace(typo_map)
        data = data[data['Crop_type'].isin(valid_crops)]

        # Field_ID is an identifier, not a feature
        data = data.drop(columns=['Field_ID'])

        # Encode the one categorical feature
        soil_encoder = LabelEncoder()
        data['Soil_type'] = soil_encoder.fit_transform(data['Soil_type'])

        # Split features from target
        X = data.drop(columns=['Crop_type'])
        y = data['Crop_type']
        feature_names = list(X.columns)

        # Encode the target with its own encoder so we can decode predictions later
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test, label_encoder, feature_names
    ### END FUNCTION
    return (prepare_data,)


@app.cell
def _(farm_df, prepare_data):
    X_train, X_test, y_train, y_test, label_encoder, feature_names = prepare_data(farm_df)

    print("Training set:", X_train.shape)
    print("Test set:", X_test.shape)
    print("Number of crop classes:", len(label_encoder.classes_))
    print("Crop classes:", list(label_encoder.classes_))
    print("Feature names:", feature_names)
    return X_test, X_train, label_encoder, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cleaning collapsed the fourteen raw labels into eight real crops, landing us at 4,523 training and 1,131 test fields across 11 features.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Part 2: Neural Network Classifier

    With the data prepared, we build the first advanced classifier: a **feed-forward neural network** in TensorFlow/Keras. Dense layers separated by dropout layers guard against overfitting, and a Softmax output produces a probability distribution over the eight crops.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 2: Building the Neural Network

    Classical classifiers operate on engineered features using mathematical rules you can inspect directly. A neural network does something different: it learns a hierarchy of representations from the raw features, composing simple patterns into increasingly abstract ones across its layers. Whether that additional complexity pays off on this data is exactly what we will find out in Challenge 7 — but first we need to build the network correctly.

    The architecture here is deliberately modest: two hidden layers with dropout regularisation between them. The dropout layers randomly zero out a fraction of activations during training, which forces the network to learn redundant representations and reduces overfitting on a dataset this size.

    ### Task

    Write a function `build_neural_network` that:
    - Takes `input_dim` (number of input features) and `num_classes` (number of crop types).
    - Builds a **Sequential** model with:
      - Dense layer: 128 units, ReLU activation (set `input_shape=(input_dim,)`)
      - Dropout layer: rate 0.3
      - Dense layer: 64 units, ReLU activation
      - Dropout layer: rate 0.3
      - Output Dense layer: `num_classes` units, Softmax activation
    - Compiles with optimizer `'adam'`, loss `'sparse_categorical_crossentropy'`, metrics `['accuracy']`.
    - Returns the compiled model.

    ### Expected Output
    ```
    Model: "sequential"
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
    ┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
    │ dense (Dense)                   │ (None, 128)            │         1,536 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dropout (Dropout)               │ (None, 128)            │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense_1 (Dense)                 │ (None, 64)             │         8,256 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dropout_1 (Dropout)             │ (None, 64)             │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense_2 (Dense)                 │ (None, 8)              │           520 │
    └─────────────────────────────────┴────────────────────────┴───────────────┘
     Total params: 10,312 (40.28 KB)
     Trainable params: 10,312 (40.28 KB)
     Non-trainable params: 0 (0.00 B)
    ```

    A Keras model summary showing three `Dense` layers (128 → 64 → 8 units) interleaved with two `Dropout` layers, and a positive total parameter count.
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def build_neural_network(input_dim, num_classes):
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax'),
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        return model
    ### END FUNCTION
    return (build_neural_network,)


@app.cell
def _(X_train, build_neural_network, label_encoder):
    input_dim   = X_train.shape[1]
    num_classes = len(label_encoder.classes_)

    nn_model = build_neural_network(input_dim, num_classes)
    nn_model.summary()
    return (nn_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 3: Training the Neural Network

    A compiled model is just an architecture. Training is where the weights are learned — where the network adjusts its internal parameters across 50 passes through the data (epochs) to minimise the cross-entropy loss between its predictions and the true crop labels.

    The validation split is crucial. By holding back 20% of the training data for validation during each epoch, we get a running comparison between training accuracy and out-of-sample accuracy. A widening gap between the two curves is the visual signature of overfitting — the network memorising training examples rather than learning general patterns.

    ### Task

    Write a function `train_neural_network` that:
    - Takes `model`, `X_train`, and `y_train`.
    - Trains for **50 epochs** with **batch size 32**.
    - Uses a **20% validation split**.
    - Sets `verbose=0`.
    - Returns `(model, history)`.

    ### Expected Output
    Training curves for accuracy and loss, then the test accuracy and per-crop classification report below.

    ```
    Neural Network Test Accuracy: 0.4757

                  precision    recall  f1-score   support

          banana       0.41      0.31      0.35       126
         cassava       0.31      0.31      0.31       108
          coffee       0.44      0.53      0.48       121
           maize       0.19      0.06      0.09        69
          potato       0.47      0.66      0.55       173
            rice       0.33      0.22      0.27        49
             tea       0.70      0.81      0.75       198
           wheat       0.44      0.38      0.41       287

        accuracy                           0.48      1131
       macro avg       0.41      0.41      0.40      1131
    weighted avg       0.45      0.48      0.46      1131
    ```

    On this small survey the environmental features only partially separate the eight crops, so accuracy lands in a modest range rather than near-perfect — the point of the project is the *comparison*, which comes next. Watch the gap between the training and validation curves: a widening gap is the visual signature of overfitting that the dropout layers are there to limit.
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def train_neural_network(model, X_train, y_train):
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
        )
        return model, history
    ### END FUNCTION
    return (train_neural_network,)


@app.cell
def _(
    X_test,
    X_train,
    accuracy_score,
    classification_report,
    label_encoder,
    nn_model,
    np,
    plt,
    tf,
    train_neural_network,
    y_test,
    y_train,
):
    tf.random.set_seed(42)
    nn_model_1, history = train_neural_network(nn_model, X_train, y_train)
    print('Training complete.')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Neural Network — Accuracy over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Neural Network — Loss over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    plt.tight_layout()
    plt.show()
    y_pred_probs = nn_model_1.predict(X_test)
    y_pred_nn = np.argmax(y_pred_probs, axis=1)
    nn_accuracy = accuracy_score(y_test, y_pred_nn)
    print(f'Neural Network Test Accuracy: {nn_accuracy:.4f}')
    print()
    print(classification_report(y_test, y_pred_nn, target_names=label_encoder.classes_))
    return (nn_accuracy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Part 3: Classifier Model Selection

    A single result is not a decision. We compare classifiers systematically with **5-fold stratified cross-validation**, which reduces the risk of overfitting to one particular split and respects class imbalance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 4: Building All Classifiers

    The neural network gives us one data point. To make a defensible recommendation to Sanaa, we need a comparison: multiple model families evaluated under the same conditions. Each classifier in this challenge brings a different inductive bias — Logistic Regression assumes a linear boundary, KNN uses proximity, SVM finds a maximal margin, Random Forest aggregates many trees, AdaBoost combines weak learners sequentially. Seeing them all together is how we identify which mathematical assumptions best match the structure of this agricultural data.

    ### Task

    Write a function `get_classifiers` that:
    - Takes no input.
    - Returns a dictionary mapping name strings to **unfitted** sklearn estimators:
      - `'Logistic Regression'`: `LogisticRegression(max_iter=1000, random_state=42)`
      - `'K-Nearest Neighbors'`: `KNeighborsClassifier(n_neighbors=5)`
      - `'Support Vector Machine'`: `SVC(kernel='rbf', random_state=42)`
      - `'Random Forest'`: `RandomForestClassifier(n_estimators=100, random_state=42)`
      - `'AdaBoost'`: `AdaBoostClassifier(n_estimators=100, random_state=42)`

    ### Expected Output
    ```
    Classifiers to evaluate:
      - Logistic Regression
      - K-Nearest Neighbors
      - Support Vector Machine
      - Random Forest
      - AdaBoost
    ```
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def get_classifiers():
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

        return {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Support Vector Machine': SVC(kernel='rbf', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        }
    ### END FUNCTION
    return (get_classifiers,)


@app.cell
def _(get_classifiers):
    classifiers = get_classifiers()
    print("Classifiers to evaluate:")
    for name in classifiers:
        print(f"  - {name}")
    return (classifiers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 5: Cross-Validation Evaluation

    A single train–test split is one opinion. Sanaa needs a verdict that does not depend on how the random seed happened to divide the data. Stratified 5-fold cross-validation gives that: the data is split into five equal folds, the model is trained on four and evaluated on one, and this rotates until every field has been used for testing exactly once. The mean accuracy across the five folds is a far more reliable estimate of real-world performance than any single split.

    The stratified part matters here. With eight crop classes that vary in how frequently they appear, a random split might place all the rare classes in the test fold. Stratification ensures each fold has roughly the same class distribution as the full dataset.

    ### Task

    Write a function `evaluate_classifiers` that:
    - Takes `classifiers` (dict), `X_train`, and `y_train`.
    - Uses **5-fold `StratifiedKFold`** with `shuffle=True` and `random_state=42`.
    - For each classifier, records cross-validation **accuracy** scores via `cross_val_score`.
    - Returns a DataFrame with columns:
      - `Classifier` — the name
      - `Mean CV Accuracy` — mean across folds, rounded to 4 decimals
      - `Std CV Accuracy` — standard deviation across folds, rounded to 4 decimals
    - Sorted by `Mean CV Accuracy` descending, with a reset index.

    > **Note:** 5 classifiers × 5 folds = 25 fits — this may take a minute or two.

    ### Expected Output
    ```
                Classifier  Mean CV Accuracy  Std CV Accuracy
    Support Vector Machine            0.4519           0.0125
             Random Forest            0.4519           0.0076
       Logistic Regression            0.4477           0.0194
                  AdaBoost            0.4287           0.0098
       K-Nearest Neighbors            0.3909           0.0099
    ```
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def evaluate_classifiers(classifiers, X_train, y_train):
        import pandas as pd
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rows = []
        for name, clf in classifiers.items():
            scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy')
            rows.append({
                'Classifier': name,
                'Mean CV Accuracy': round(scores.mean(), 4),
                'Std CV Accuracy': round(scores.std(), 4),
            })

        results_df = pd.DataFrame(rows).sort_values(
            'Mean CV Accuracy', ascending=False
        ).reset_index(drop=True)

        return results_df
    ### END FUNCTION
    return (evaluate_classifiers,)


@app.cell
def _(X_train, classifiers, evaluate_classifiers, y_train):
    cv_results = evaluate_classifiers(classifiers, X_train, y_train)
    print(cv_results.to_string(index=False))
    return (cv_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The accuracies are modest — these eight crops genuinely overlap in environmental conditions, and the survey lacks some discriminating features. What matters for model selection is the **ranking and stability**: tree-based and margin-based methods (Random Forest, SVM) lead, and their low standard deviations say they are consistent across folds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 6: Visualising the Results

    Numbers in a table and a plot tell different stories. The bar chart with error bars makes the ranking immediately readable — and crucially, it lets Sanaa see the neural network's accuracy on the same axis as the classical classifiers. If the network falls below the best classical model, that is a meaningful finding: added complexity did not pay off, and the simpler model is the right recommendation.

    ### Task

    Write a function `plot_classifier_comparison` that:
    - Takes `cv_results` (DataFrame) and `nn_accuracy` (float).
    - Draws a horizontal bar chart of `Mean CV Accuracy` per classifier, with error bars from `Std CV Accuracy`.
    - Adds a vertical dashed line at `nn_accuracy`, labeled for the neural network.
    - Uses a clear title and axis labels.
    - Returns `None`.

    ### Expected Output
    A horizontal bar chart with one bar per classifier (error bars showing fold-to-fold variation) and a dashed vertical line marking the neural network's test accuracy for comparison.
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def plot_classifier_comparison(cv_results, nn_accuracy):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(
            cv_results['Classifier'],
            cv_results['Mean CV Accuracy'],
            xerr=cv_results['Std CV Accuracy'],
            color='steelblue',
            capsize=4,
        )
        ax.axvline(
            nn_accuracy, color='crimson', linestyle='--',
            label=f'Neural Network ({nn_accuracy:.4f})',
        )
        ax.set_xlabel('Mean CV Accuracy')
        ax.set_title('Classifier Comparison: 5-Fold CV Accuracy vs Neural Network')
        ax.invert_yaxis()
        ax.legend()
        plt.tight_layout()
        plt.show()
        return None
    ### END FUNCTION
    return (plot_classifier_comparison,)


@app.cell
def _(cv_results, nn_accuracy, plot_classifier_comparison):
    plot_classifier_comparison(cv_results, nn_accuracy)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 7: Final Model Evaluation

    Cross-validation selects the winner. But the cross-validation was run on training data — we have not yet touched the held-out test set. This is the final step: fit the winning model on the entire training set, run it on the 20% of fields it has never seen, and report the result. This test-set number is the one we quote to Sanaa. It is the most honest estimate of how the deployed system will perform on a new field submitted by a farmer tomorrow.

    ### Task

    Write a function `evaluate_best_classifier` that:
    - Takes `cv_results`, `classifiers`, `X_train`, `y_train`, `X_test`, `y_test`, and `label_encoder`.
    - Identifies the classifier with the **highest mean CV accuracy** (the first row of the sorted `cv_results`).
    - Fits that classifier on the **full training set**.
    - Returns a dictionary with:
      - `'best_classifier_name'` — string
      - `'test_accuracy'` — float, rounded to 4 decimals
      - `'classification_report'` — string, using the crop names as target names

    ### Expected Output
    ```
    Best classifier: Support Vector Machine
    Test accuracy:   0.4695

                  precision    recall  f1-score   support

          banana       0.35      0.41      0.38       126
         cassava       0.34      0.31      0.33       108
          coffee       0.44      0.40      0.42       121
           maize       0.00      0.00      0.00        69
          potato       0.46      0.60      0.52       173
            rice       0.31      0.08      0.13        49
             tea       0.73      0.80      0.76       198
           wheat       0.41      0.46      0.44       287

        accuracy                           0.47      1131
       macro avg       0.38      0.38      0.37      1131
    weighted avg       0.44      0.47      0.45      1131
    ```
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def evaluate_best_classifier(cv_results, classifiers, X_train, y_train,
                                 X_test, y_test, label_encoder):
        from sklearn.metrics import accuracy_score, classification_report

        best_name = cv_results.iloc[0]['Classifier']
        best_model = classifiers[best_name]
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        test_accuracy = round(accuracy_score(y_test, y_pred), 4)
        report = classification_report(
            y_test, y_pred, target_names=label_encoder.classes_
        )

        return {
            'best_classifier_name': best_name,
            'test_accuracy': test_accuracy,
            'classification_report': report,
        }
    ### END FUNCTION
    return (evaluate_best_classifier,)


@app.cell
def _(
    X_test,
    X_train,
    classifiers,
    cv_results,
    evaluate_best_classifier,
    label_encoder,
    y_test,
    y_train,
):
    final_results = evaluate_best_classifier(
        cv_results, classifiers, X_train, y_train, X_test, y_test, label_encoder)

    print(f"Best classifier: {final_results['best_classifier_name']}")
    print(f"Test accuracy:   {final_results['test_accuracy']:.4f}")
    print()
    print(final_results['classification_report'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wrapping up

    You built the **Advanced Agricultural Intelligence System** for Maji Ndogo, end to end:

    1. **Data Preparation.** You loaded the four-table survey, and — crucially — *noticed the data was dirty*. Fourteen raw crop labels collapsed into eight real classes once whitespace and typos were handled. That cleaning step is the difference between a model that learns crops and one that learns typos.
    2. **Neural Network.** You built and trained a feed-forward Keras classifier, and read its training curves for the tell-tale gap between training and validation that signals overfitting.
    3. **Model Selection.** You compared five classical classifiers under 5-fold stratified cross-validation, visualized the results against the neural network, and defended a single choice on held-out data. Cross-validation — not a lucky split — is what lets you answer Sanaa's real question: *which model is steady enough to stake a season on?*

    The honest result here is a modest accuracy, and that is worth sitting with: it tells the Ministry that the structured survey alone does not fully determine the right crop, and that richer features (like the parsed weather-station logs awaiting you in a later module) may be where the next gain comes from. A consultant who reports that clearly is worth more than one who inflates a number.

    **Questions to consider:**
    - How does the neural network compare to the best classical classifier? Does the more complex model actually win here, and what would you conclude if it didn't?
    - The cross-validation standard deviation measures **stability**. Which model is most consistent across folds, and why might stability matter as much as peak accuracy for a system farmers will rely on?
    - The accuracy is modest. Before reaching for a fancier model, what would you try first — more features, more data, or better labels — and how would you justify that order to a client?

    The Ministry now has a working crop recommendation system grounded in real field data, and an honest account of its limits. That is the final deliverable.

    — Sanaa
    """)
    return


@app.cell
def cell_readiness():
    import inspect as _inspect
    import ast as _ast
    _spec = {'build_neural_network': {'expected_params': ['input_dim', 'num_classes']}, 'evaluate_best_classifier': {'expected_params': ['cv_results', 'classifiers', 'X_train', 'y_train', 'X_test', 'y_test', 'label_encoder']}, 'evaluate_classifiers': {'expected_params': ['classifiers', 'X_train', 'y_train']}, 'get_classifiers': {'expected_params': []}, 'plot_classifier_comparison': {'expected_params': ['cv_results', 'nn_accuracy']}, 'prepare_data': {'expected_params': ['df']}, 'train_neural_network': {'expected_params': ['model', 'X_train', 'y_train']}}
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
