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
    # SVM Hyperparameter Tuning
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Multi-dimensional chemical profiling demands high-precision decision boundaries. To accurately classify environmental samples where raw readings span completely different scales, you must isolate the optimal hyperplane configuration. In this project, you build and fine-tune a `Support Vector Classifier` (SVC) on a complex chemical signature benchmark. You will standardize non-linear feature matrices, engineer a rigid custom binary log-loss metric to penalize overconfident algorithmic errors, and execute cross-validated hyperparameter sweeps via `GridSearchCV`. This creates the precise mathematical tuning and explicit engineering handoff needed to map delicate soil profiles before deploying crop suitability models in the field.

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
    - Use the tools introduced in this course: pandas, NumPy, scikit-learn, TensorFlow/Keras, Matplotlib, and Seaborn — for advanced classification (SVMs, KNN, ensembles, neural networks), hyperparameter tuning, and model selection.
    - The use of StackOverflow, Google, Generative AI tools, and any other online resources is permitted. Use AI to help you understand — not to shortcut the thinking. [Read the honor code here](https://drive.google.com/file/d/1atFOPUQRLz5slb4Q1ASXh8QQfKyXVqrw/preview).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Last module, we proved to the Ministry of Agriculture that we can classify handwritten digits accurately. That is one pillar of the digitisation pipeline. But Sanaa, our contact at the ministry, needs to know we can do more than read numbers — she needs to know we can classify complex, multi-dimensional chemical signatures.

    Why? Because the full Maji Ndogo agricultural database — which we will access in Module 3 — contains detailed soil chemistry measurements for each field: pH levels, nitrogen content, phosphorus, potassium, and more. Classifying crop suitability from these chemical profiles is a genuinely hard problem, and before the ministry hands us those files, Sanaa wants to see us solve a structurally identical challenge.

    This module, we will classify the quality of a local fermented grain brew from chemical assay data — a dataset with the same structure as soil chemistry classification. The ministry's agricultural cooperative already runs lab assays on batches of sorghum- and millet-based brew sold in regional markets, and the resulting **Brew Quality Bank** gives us a real, lower-stakes chemistry classification problem to prove ourselves on before the soil data arrives. The variables (acidity, sulphates, pH, alcohol content) are direct proxies for the kind of chemical readings we will encounter when classifying field suitability in Module 3.

    Our tool of choice is a **Support Vector Classifier (SVC)**. We will train a baseline model, evaluate it with a custom log-loss metric, then systematically optimise its hyperparameters using **GridSearchCV**.
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
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    import seaborn as sns

    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    return (
        GridSearchCV,
        SVC,
        accuracy_score,
        make_scorer,
        np,
        pd,
        preprocessing,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Dataset

    We are using the **Maji Ndogo Brew Quality Bank** — chemical assay results for batches of a fermented grain brew produced from two staple crops grown across Maji Ndogo's provinces. Each batch was scored by the cooperative's tasting panel and analysed in the lab, described by the following variables:

    | Feature | Description |
    |---|---|
    | `fixed acidity` | Tartaric-acid-equivalent concentration |
    | `volatile acidity` | Acetic acid concentration |
    | `citric acid` | Citric acid (adds freshness) |
    | `residual sugar` | Remaining sugar after fermentation |
    | `chlorides` | Salt concentration |
    | `free sulfur dioxide` | Free form of SO₂ |
    | `total sulfur dioxide` | Total SO₂ (free + bound) |
    | `density` | Density relative to water |
    | `pH` | Acidity/alkalinity |
    | `sulphates` | Antimicrobial and antioxidant additive |
    | `alcohol` | Alcohol percentage |
    | `type` | Grain base — 0 = sorghum, 1 = millet |
    | `quality` | Panel score between 0 and 10 (our **target**) |

    The data loads directly from the file below — no manual download is required. Notice the structural similarity to soil chemistry: pH, sulphates, and acid concentrations are directly analogous to the soil measurements in the Maji Ndogo database.
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv('brew_quality.csv')
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 1: Data Preprocessing

    Sanaa has been specific about how she wants the quality labels handled. Brew quality scores in this dataset range from 3 to 9, and while that gives us nine possible classes, most of the real-world decisions reduce to a binary question: is this batch acceptable or not? We apply the same framing to Maji Ndogo field data — is this field suitable for irrigation or not. The binary framing keeps the model interpretable and the business decision clear.

    Before the classifier sees any data, we also need to standardise the features. The chemical variables in this dataset have wildly different scales — sulphates are measured in grams per litre while total sulphur dioxide can reach hundreds of parts per million. Without standardisation, the SVC will weight high-magnitude features disproportionately and produce a badly calibrated decision boundary.

    ### Task

    Write a function `data_preprocess` that:
    - Takes a DataFrame as input.
    - Converts `quality` labels:
      - Quality **≤ 4** → label **0** (lower quality)
      - Quality **≥ 5** → label **1** (higher quality)
    - Fills any `NaN` values with zeros.
    - Standardises the features using `sklearn`'s `StandardScaler`.
    - Splits the data into **75% training** and **25% testing**, with `random_state=42`.
    - Returns two tuples: `(X_train, y_train), (X_test, y_test)`.

    ### Expected Output
    ```
    [[ 0.0788577  -0.32538893 -0.17534887 -0.11007128  0.06674735  0.46531515
       1.17818679  2.41030969 -0.70042723  0.84125854 -1.30154194  1.71602681]
     [-1.92454798  1.6299215   1.35374038 -0.57515249 -0.28964441  0.10320922
      -0.70610967 -1.31711707 -0.12747816  0.1400211  -0.95732519  1.71602681]]
    [1 1]
    [[-0.13924929  0.35413355 -1.29463181  1.28553674 -0.51813495 -0.09170206
       1.07569514  0.23211965 -1.40767708 -0.81429035  0.12353384 -0.58274148]
     [ 0.51546331 -1.1330513   0.35623528 -0.42741846  0.00794936  0.28186024
      -1.44797776  0.64443535 -0.96897622  2.73442186 -1.46931266 -0.58274148]]
    [0 1]
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run the cell below to verify your implementation.
    """)
    return


@app.cell
def _(preprocessing, train_test_split):
    ### START FUNCTION
    def data_preprocess(df):
        # Work on a copy so we never mutate the caller's DataFrame.
        df = df.fillna(0).copy()

        # Binarise the target: <=4 is "low quality" (0), >=5 is "high quality" (1).
        df['quality'] = df['quality'].apply(lambda q: 0 if q <= 4 else 1)

        X = df.drop(columns=['quality'])
        y = df['quality'].values

        # Standardise features so no single column dominates the SVC's
        # distance-based decision boundary just because of its raw scale.
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42
        )
        return (X_train, y_train), (X_test, y_test)
    ### END FUNCTION
    return (data_preprocess,)


@app.cell
def _(data_preprocess, df):
    (X_train, y_train), (X_test, y_test) = data_preprocess(df)
    print(X_train[:2])
    print(y_train[:2])
    print(X_test[:2])
    print(y_test[:2])
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 2: Training a Baseline Model

    Before we run any optimisation, we need a reference point. Sanaa will only trust an improved model if she can see what it improved from. A baseline SVC with default parameters gives us that anchor — we measure its log-loss and accuracy, then compare those numbers against the tuned version in Challenges 5 and 6.

    The Support Vector Classifier finds a hyperplane in the feature space that maximally separates the two classes. The `gamma` parameter controls how far the influence of a single training sample reaches. Setting it to `'auto'` scales gamma inversely with the number of features, which is a sensible default when we do not yet know the right scale.

    ### Task

    Write a function `train_SVC_model` that:
    - Takes two numpy arrays `X_train` and `y_train` as input.
    - Returns a fitted `SVC` model with:
      - `random_state = 40`
      - `gamma = 'auto'`

    ### Expected Output
    ```
    array([0, 1], dtype=int64)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run the cell below to verify your implementation.
    """)
    return


@app.cell
def _(SVC):
    ### START FUNCTION
    def train_SVC_model(X_train, y_train):
        model = SVC(random_state=40, gamma='auto')
        model.fit(X_train, y_train)
        return model
    ### END FUNCTION
    return (train_SVC_model,)


@app.cell
def _(X_train, train_SVC_model, y_train):
    svc = train_SVC_model(X_train, y_train)
    svc.classes_
    return (svc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 3: Custom Scoring Function

    Accuracy is a useful headline metric, but Sanaa needs a more sensitive measure of model quality. She has seen classifiers that look good on accuracy but systematically fail on the minority class — and in the context of crop recommendations, a confident wrong prediction can mean an entire growing season is wasted on an unsuitable crop.

    We implement **log-loss** — a metric that penalises confident wrong predictions more severely than uncertain ones. A prediction of 0.99 for the wrong class incurs a much larger penalty than a prediction of 0.51. This asymmetry makes log-loss the right metric when the cost of being confidently wrong is high. This is the same metric Sanaa will use to evaluate crop classifiers in Module 3.

    The formula is:

    $$H(p,q) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

    ### Task

    Write a function `custom_scoring_function` that:
    - Takes two numpy arrays: `y_true` and `y_pred`.
    - Clips predictions using `epsilon = 1e-15` to avoid log(0).
    - Returns a **float64** for the log-loss value, rounded to **7 decimal places**.

    > **Hint:** `np.maximum` and `np.minimum` are useful for clipping values.

    ### Expected Output
    ```
    Log Loss value:  1.020245
    Accuracy:  0.9705
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run the cell below to verify your implementation.
    """)
    return


@app.cell
def _(np):
    ### START FUNCTION
    def custom_scoring_function(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.maximum(epsilon, y_pred)
        y_pred = np.minimum(1 - epsilon, y_pred)
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return np.float64(round(loss, 7))
    ### END FUNCTION
    return (custom_scoring_function,)


@app.cell
def _(X_test, accuracy_score, custom_scoring_function, svc, y_test):
    _y_pred = svc.predict(X_test)
    print('Log Loss value: ', custom_scoring_function(y_test, _y_pred))
    print('Accuracy: ', round(accuracy_score(y_test, _y_pred), 4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hyperparameter Optimisation

    A 96.37% accuracy baseline is already impressive, but log-loss of 1.25 suggests our model is making some confident mistakes. Let's use GridSearchCV to find better hyperparameters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 4: Getting Model Hyperparameters

    Before Sanaa will authorise a grid search, she wants to see exactly which levers we are pulling. Every sklearn estimator exposes its configurable parameters through `get_params()`. We will use this to inspect the full list of SVC hyperparameters — knowing what is tuneable before we decide what to tune is good engineering discipline.

    This function also serves as a reusable diagnostic tool. In Module 3, when we compare multiple classifier types on the Maji Ndogo data, we can call this same function on any sklearn model to inspect its parameter space before designing the search grid.

    ### Task

    Write a function `get_model_hyperparams` that:
    - Takes an sklearn model (estimator) object as input.
    - Returns a **list** of parameter names for the given model.

    ### Expected Output
    ```
    ['C',
     'break_ties',
     'cache_size',
     'class_weight',
     'coef0',
     'decision_function_shape',
     'degree',
     'gamma',
     'kernel',
     'max_iter',
     'probability',
     'random_state',
     'shrinking',
     'tol',
     'verbose']
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run the cell below to verify your implementation.
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def get_model_hyperparams(model):
        return list(model.get_params().keys())
    ### END FUNCTION
    return (get_model_hyperparams,)


@app.cell
def _(get_model_hyperparams, svc):
    get_model_hyperparams(svc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 5: Hyperparameter Search

    Now we run the search. Sanaa has approved a focused grid over two hyperparameters — `C` (the regularisation strength) and `gamma` (the kernel width). These two parameters control the fundamental bias-variance trade-off of the SVC: `C` controls how much we penalise misclassification, and `gamma` controls how tightly the decision boundary wraps around the training data.

    We use our `custom_scoring_function` from Challenge 3 as the scoring metric, wrapped in sklearn's `make_scorer`. Because log-loss is a metric we want to **minimise**, we pass `greater_is_better=False`. GridSearchCV then selects the parameter combination that minimises log-loss across 5 cross-validation folds.

    ### Task

    Write a function `tune_SVC_model` that:
    - Takes `X_train` and `y_train` as input.
    - Defines a parameter grid: `D = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}`.
    - Uses `custom_scoring_function` wrapped in `make_scorer` with `greater_is_better=False`.
    - Returns a **fitted** `GridSearchCV` object with **5-fold cross-validation**.

    > **Hint:** Look at `sklearn.metrics.make_scorer`.

    ### Expected Output
    ```
    Log Loss value:  0.9564801
    Accuracy:  0.9723
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run the cell below to verify your implementation.
    """)
    return


@app.cell
def _(GridSearchCV, SVC, custom_scoring_function, make_scorer):
    ### START FUNCTION
    def tune_SVC_model(X_train, y_train):
        D = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
        scorer = make_scorer(custom_scoring_function, greater_is_better=False)
        grid = GridSearchCV(SVC(), param_grid=D, scoring=scorer, cv=5)
        grid.fit(X_train, y_train)
        return grid
    ### END FUNCTION
    return (tune_SVC_model,)


@app.cell
def _(
    X_test,
    X_train,
    accuracy_score,
    custom_scoring_function,
    tune_SVC_model,
    y_test,
    y_train,
):
    svc_tuned = tune_SVC_model(X_train, y_train)
    _y_pred = svc_tuned.predict(X_test)
    print('Log Loss value: ', custom_scoring_function(y_test, _y_pred))
    print('Accuracy: ', round(accuracy_score(y_test, _y_pred), 4))
    return (svc_tuned,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 6: Extracting the Best Parameters

    The grid search is complete. Sanaa now needs the optimal hyperparameter values in a clean dictionary she can document and hand to the engineering team. When the ministry deploys the classifier to production, the infrastructure team will need these exact parameter values to instantiate the model correctly — not a `GridSearchCV` object, just the parameters.

    This function extracts that dictionary from the fitted search object. It is a simple wrapper, but it is an important one: it makes the handoff from data science to engineering clean and explicit.

    ### Task

    Write a function `get_best_params` that:
    - Takes a fitted `GridSearchCV` object as input.
    - Returns a **dictionary** of the optimal parameters.

    ### Expected Output
    ```
    {'C': 10, 'gamma': 0.01}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run the cell below to verify your implementation.
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def get_best_params(model):
        return model.best_params_
    ### END FUNCTION
    return (get_best_params,)


@app.cell
def _(get_best_params, svc_tuned):
    get_best_params(svc_tuned)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wrapping up

    The tuned SVC reduced log-loss from 1.020 to 0.956 and improved accuracy from 97.05% to 97.23%. A modest but meaningful improvement — and that improvement came entirely from systematically searching the hyperparameter space, not from changing the model architecture.

    This is the principle we will apply in Module 3 when comparing multiple classifier types on the Maji Ndogo farm data. GridSearchCV will help us find the optimal configuration for each model before we select the best overall classifier for crop recommendation.

    The ministry has cleared us for access to the full agricultural database.
    """)
    return


@app.cell
def cell_readiness():
    import inspect as _inspect
    import ast as _ast
    _spec = {'custom_scoring_function': {'expected_params': ['y_true', 'y_pred']}, 'data_preprocess': {'expected_params': ['df']}, 'get_best_params': {'expected_params': ['model']}, 'get_model_hyperparams': {'expected_params': ['model']}, 'train_SVC_model': {'expected_params': ['X_train', 'y_train']}, 'tune_SVC_model': {'expected_params': ['X_train', 'y_train']}}
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
