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
    # Maji Ndogo Energy Shortfall — Regression Predict
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The diagnostic groundwork is complete; now it’s time to deploy. Having isolated the overlapping weather signals that bloated your baseline, the Maji Ndogo Department of Energy needs a regularized model ready for production. In this project, you build a complete end-to-end regression pipeline using scikit-learn. You will prune noisy variables using correlation thresholding, neutralize scaling bias via `StandardScaler`, and implement Ridge and LASSO regularization to penalize unstable coefficients. Finally, you will serialize your architecture using `pickle` to deliver a persistent, automated forecasting tool. It is the exact optimization workflow that transitions a machine learning project from an exploratory notebook into a deployed enterprise solution.
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
    - Use the tools introduced in this course: pandas, NumPy, scikit-learn, statsmodels, and Matplotlib — for simple and multiple linear regression, model evaluation (R², MAE, MSE, RMSE), regularisation, and model persistence.
    - The use of StackOverflow, Google, Generative AI tools, and any other online resources is permitted. Use AI to help you understand — not to shortcut the thinking. [Read the honor code here](https://drive.google.com/file/d/1atFOPUQRLz5slb4Q1ASXh8QQfKyXVqrw/preview).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Three modules ago you proved that linear relationships exist between Maji Ndogo's weather patterns and its energy shortfall. Two modules ago you demonstrated that naively adding more predictors — five provincial temperatures — barely improved the model, because multicollinearity was eating the signal. This module, you fix that.

    The Maji Ndogo Department of Energy has given a clear mandate: build a model they can deploy. That means one that is accurate enough to plan infrastructure investments, robust enough to run on data it has never seen, and persistent enough to run again next month without retraining.

    You are completing the full production pipeline:

    | Step | Technique | Purpose |
    |---|---|---|
    | Feature pruning | Correlation thresholding | Eliminate low-signal, high-noise predictors |
    | Preprocessing | StandardScaler | Put features on equal footing before penalization |
    | Regularization | Ridge (L2) | Shrink unstable multicollinear coefficients |
    | Regularization | LASSO (L1) | Zero out the least useful features entirely |
    | Persistence | pickle | Save the trained model for deployment |
    | Submission | predict on test set | Deliver predictions to the Department |

    The dataset comes from five Maji Ndogo provinces — Sokoto, Kilimani, Hawassa, Amanzi, and Akatsi — recording temperature, humidity, wind speed, pressure, and other weather variables every 3 hours. Your target is `energy_shortfall_3h`: the gap between fossil-fuel and renewable energy generation during each 3-hour window on the national grid.

    Your RMSE submission will appear on the leaderboard. Every unit of improvement represents a real reduction in the Department's forecasting error. Build it well.

    > **AI assist:** Use AI actively this module. Good prompts: *"Explain the difference between Ridge and LASSO regularization in terms of what happens to small coefficients."* *"What is data leakage, and why must I never refit a StandardScaler on test data?"*

    ### Evaluation Metric

    **Root Mean Square Error (RMSE)** — lower is better. Every reduction in RMSE represents a real improvement in the Department's ability to plan energy infrastructure across Maji Ndogo's five provinces.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Data Preparation

    Two columns require preprocessing before any modeling:
    - `Hawassa_wind_deg`: Wind direction encoded as string levels (`level_1` through `level_10`) — extract the numeric value.
    - `Kilimani_pressure`: Atmospheric pressure encoded as strings (`kp1` through `kp25`) — extract the numeric value.
    - `Hawassa_pressure`: Contains missing values (sensor outages at the Hawassa weather station) — impute with the column median (we cannot drop test rows).

    The `prepare_data` helper below handles all three. Run this cell before attempting any challenge.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # headless rendering for test environments
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pickle
    import warnings
    warnings.filterwarnings('ignore')

    train_path = 'df_train.csv'  # Replace with your file path
    test_path  = 'df_test.csv'   # Replace with your file path

    df_train_raw = pd.read_csv(train_path, index_col=0)
    df_test_raw  = pd.read_csv(test_path,  index_col=0)

    def prepare_data(df):
        """Encode categorical columns and impute missing values with column medians."""
        df = df.copy()
        df = df.drop(columns=['time'], errors='ignore')
        df['Hawassa_wind_deg']  = df['Hawassa_wind_deg'].str.extract(r'(\d+)').astype(float)
        df['Kilimani_pressure'] = df['Kilimani_pressure'].str.extract(r'(\d+)').astype(float)
        df = df.fillna(df.median(numeric_only=True))
        return df

    df_train = prepare_data(df_train_raw)
    df_test  = prepare_data(df_test_raw)

    print('Train shape:', df_train.shape)
    print('Test shape: ', df_test.shape)
    return (
        Lasso,
        Ridge,
        StandardScaler,
        df_test_raw,
        df_train,
        mean_squared_error,
        np,
        pd,
        pickle,
        prepare_data,
        r2_score,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 1: Variable Selection Using Correlation

    In Module 2 we found that five provincial temperature columns inflate VIF into the thousands because they measure the same underlying national weather pattern. Rather than feeding all 56 features into a regularized model blindly, we'll prune first: keep only the features that have a meaningful linear relationship with the target.

    Features with low correlation to `energy_shortfall_3h` add noise without improving predictions — and they slow down every model evaluation unnecessarily.

    ### Task
    Create a function named `select_features_by_correlation` that:
    - Takes a DataFrame and a correlation threshold as input.
    - Computes the absolute Pearson correlation of every numeric feature with `energy_shortfall_3h`.
    - Prints the number of selected features.
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def select_features_by_correlation(df, threshold=0.15):
        correlations = df.corr(numeric_only=True)['energy_shortfall_3h'].drop('energy_shortfall_3h')
        selected = correlations[correlations.abs() > threshold].index.tolist()
        print(f'Selected {len(selected)} features out of {len(correlations)}')
        return selected
    ### END FUNCTION
    return (select_features_by_correlation,)


@app.cell
def _(df_train, select_features_by_correlation):
    # Input:
    selected_features = select_features_by_correlation(df_train, threshold=0.15)
    print(f'Selected features: {selected_features}')
    return (selected_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We've reduced from 56 raw features to 22. The selected features are dominated by temperature and pressure across all five provinces — the variables most associated with heating/cooling demand and grid stability. `Sokoto_wind_speed` and `Hawassa_wind_speed` also pass the threshold, reflecting wind energy's contribution to the shortfall. Humidity and cloud cover fall just short of the 0.15 threshold this round. Every feature we trim is one fewer parameter the model has to estimate — and one fewer source of overfitting.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 2: Data Scaling and Ridge Regression

    The 22 selected features live on very different scales: temperatures in Kelvin (~285–305), wind speed in m/s (0–10), pressure in hPa (~995–1030). When regularization penalizes large coefficients, features on bigger scales get penalized more heavily — not because they matter less, but simply because their units are larger. **StandardScaler** removes this bias by transforming each feature to zero mean and unit variance.

    **Ridge regression** (L2 regularization) then applies a penalty proportional to the *square* of each coefficient, shrinking large and unstable coefficients toward zero without eliminating any feature entirely. This directly addresses the multicollinearity we measured in Module 2.

    **Critical rule:** The scaler must be **fit on training data only**, then applied (not refit) to the test data. Fitting on test data would leak future information into the model.

    ### Task
    Create a function named `fit_ridge_model` that:
    - Takes the training DataFrame, the list of selected features, and a regularization parameter `alpha` (default `1.0`).
    - Splits the data 80-20 with `random_state=42`.
    - Scales features using `StandardScaler` — fit on training, transform both.
    - Fits a `Ridge` model on the scaled training set.
    - Evaluates on the scaled test set.
    - Returns `(r2, rmse, scaler, model)` — the scaler and model are returned for use in Challenges 4 and 5.
    """)
    return


@app.cell
def _(Ridge, StandardScaler, mean_squared_error, r2_score, train_test_split):
    ### START FUNCTION
    def fit_ridge_model(df, feature_cols, alpha=1.0):
        X = df[feature_cols]
        y = df['energy_shortfall_3h']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)
        r2 = r2_score(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5

        return r2, rmse, scaler, model
    ### END FUNCTION
    return (fit_ridge_model,)


@app.cell
def _(df_train, fit_ridge_model, selected_features):
    # Input:
    _r2, _rmse, _scaler, _ridge_model = fit_ridge_model(df_train, selected_features)
    print(f'Ridge R-squared: {_r2}')
    print(f'Ridge RMSE: {_rmse}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare this RMSE (~4,581) to Module 2's temperature-only MLR (~4,804). Correlation-based selection and Ridge regularization together cut the error by roughly 220 units — a meaningful improvement. We're building a progressively better model with each technique we add. The Ridge model has lower RMSE because it penalizes the unstable coefficients that were inflating prediction variance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 3: LASSO Regression and Sparsity

    **LASSO regression** (L1 regularization) takes a fundamentally different approach. Rather than *shrinking* coefficients proportionally, LASSO applies a penalty proportional to the *absolute value* of each coefficient — which has the geometric effect of forcing small coefficients all the way to exactly zero. This produces **sparse models**: only the most important features survive.

    LASSO is therefore doing two things simultaneously: regularization *and* automatic feature selection. At high enough `alpha`, it will eliminate features we chose by correlation thresholding, revealing which of our 17 are truly doing the heavy lifting.

    ### Task
    Create a function named `fit_lasso_model` that:
    - Takes the training DataFrame, the list of selected features, and a regularization parameter `alpha` (default `1.0`).
    - Splits 80-20 with `random_state=42`, scales with `StandardScaler`.
    - Fits a `Lasso` model on the scaled training set.
    - Returns `(r2, rmse, n_nonzero)` where `n_nonzero` is the count of non-zero coefficients.

    ### Expected Output
    ```
    LASSO R-squared: 0.12728536730988127
    LASSO RMSE: 4957.636871918052
    Non-zero coefficients: 18 out of 18
    ```
    """)
    return


@app.cell
def _(
    Lasso,
    StandardScaler,
    mean_squared_error,
    np,
    r2_score,
    train_test_split,
):
    ### START FUNCTION
    def fit_lasso_model(df, feature_cols, alpha=1.0):
        X = df[feature_cols]
        y = df['energy_shortfall_3h']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Lasso(alpha=alpha)
        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)
        r2 = r2_score(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        n_nonzero = int(np.sum(model.coef_ != 0))

        return r2, rmse, n_nonzero
    ### END FUNCTION
    return (fit_lasso_model,)


@app.cell
def _(df_train, fit_lasso_model, selected_features):
    # Input:
    r2_lasso, rmse_lasso, n_nonzero = fit_lasso_model(df_train, selected_features)
    print(f'LASSO R-squared: {r2_lasso}')
    print(f'LASSO RMSE: {rmse_lasso}')
    print(f'Non-zero coefficients: {n_nonzero} out of {len(selected_features)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At `alpha=1.0`, LASSO and Ridge perform almost identically — and LASSO retains all 17 features. This tells us the regularization penalty is still mild relative to the signal in the data. Increase `alpha` and LASSO will begin zeroing out coefficients one by one. In a production workflow, you would sweep `alpha` values with cross-validation and choose the setting that minimizes your target metric (RMSE here).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 4: Model Persistence with pickle

    We have a trained model that took seconds to fit — but in a production environment, models may take hours or days. More importantly, the Maji Ndogo Department of Energy's grid operators need to run this model at 3-hour intervals, 24 hours a day, without access to training data or a data scientist.

    **Model persistence** solves this. Python's `pickle` module serializes any Python object — including trained scikit-learn models with all their fitted parameters — to a binary file on disk. Loading is instantaneous and produces an object that is byte-for-byte identical to the original.

    ### Task
    Create a function named `save_and_reload_model` that:
    - Takes a fitted model object and a file path as input.
    - Saves the model to disk using `pickle.dump` (binary write mode: `'wb'`).
    - Reloads it from disk using `pickle.load` (binary read mode: `'rb'`).
    - Returns the reloaded model.

    ### Expected Output
    ```
    Predictions match: True
    ```
    """)
    return


@app.cell
def _(pickle):
    ### START FUNCTION
    def save_and_reload_model(model, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)

        with open(filepath, 'rb') as f:
            reloaded_model = pickle.load(f)

        return reloaded_model
    ### END FUNCTION
    return (save_and_reload_model,)


@app.cell
def _(df_train, fit_ridge_model, np, save_and_reload_model, selected_features):
    # Input:
    _r2, _rmse, _scaler, _ridge_model = fit_ridge_model(df_train, selected_features)
    reloaded_model = save_and_reload_model(_ridge_model, 'maji_ndogo_ridge_model.pkl')
    X_sample = df_train[selected_features].head(5)
    # Verify the reloaded model produces identical predictions
    X_scaled = _scaler.transform(X_sample)
    original_preds = _ridge_model.predict(X_scaled)
    reloaded_preds = reloaded_model.predict(X_scaled)
    print('Original predictions:', original_preds)
    print('Reloaded predictions:', reloaded_preds)
    print('Predictions match:', np.allclose(original_preds, reloaded_preds))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The reloaded model produces bit-for-bit identical predictions. This is exactly what we need for deployment: a pickled model can be shipped to any system with Python and scikit-learn installed, and it will make predictions without access to the original training data or code. Note that we should also pickle the `scaler` separately — any system that loads the model also needs the same scaler to preprocess incoming data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 5: Generating Predictions for Submission

    The final step: applying our trained Ridge model to the unseen test set and producing the submission file for the Maji Ndogo Department of Energy's leaderboard.

    **Critical constraint:** The test set must be preprocessed using the **same scaler** that was fit on the training data — never refit it on the test data. Refitting would contaminate the model with information from the test set (data leakage), invalidating your evaluation.

    ### Task
    Create a function named `generate_submission` that:
    - Takes the raw test DataFrame (`df_test_raw`), the list of selected features, the fitted scaler, and the fitted Ridge model.
    - Prepares the test data: encode categorical columns, impute missing values with medians.
    - Scales the test features using the **already fitted** scaler (`.transform()` only — no `.fit_transform()`).
    - Generates predictions using the Ridge model.
    - Returns a DataFrame with columns `time` and `energy_shortfall_3h`.

    ### Expected Output
    ```
    time  energy_shortfall_3h
    0  2022-01-01 00:00:00          2085.070542
    1  2022-01-01 03:00:00          4587.629794
    2  2022-01-01 06:00:00          4963.214551
    3  2022-01-01 09:00:00          5055.798913
    4  2022-01-01 12:00:00          6843.955753
    Submission shape: (2920, 2)
    Saved to submission.csv
    ```
    """)
    return


@app.cell
def _(pd, prepare_data):
    ### START FUNCTION
    def generate_submission(df_test_raw, feature_cols, scaler, model):
        df_test = prepare_data(df_test_raw)
        X_test = df_test[feature_cols]
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)

        return pd.DataFrame({
            'time': df_test_raw['time'].values,
            'energy_shortfall_3h': predictions,
        })
    ### END FUNCTION
    return (generate_submission,)


@app.cell
def _(
    df_test_raw,
    df_train,
    fit_ridge_model,
    generate_submission,
    selected_features,
):
    # Input:
    _r2, _rmse, _scaler, _ridge_model = fit_ridge_model(df_train, selected_features)
    submission = generate_submission(df_test_raw, selected_features, _scaler, _ridge_model)
    print(submission.head())
    print(f'Submission shape: {submission.shape}')
    submission.to_csv('submission.csv', index=False)
    # Save to CSV for leaderboard submission
    print('Saved to submission.csv')
    return


if __name__ == "__main__":
    app.run()
