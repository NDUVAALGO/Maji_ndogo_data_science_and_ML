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
    # The Global Benchmark Trial — Decision Trees
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Regression tells us *how much*; now it's time to discover *which one*. To shift from numeric forecasting to land and crop classification, you need an architecture designed for categories. In this project, you build your first non-linear model using scikit-learn's `DecisionTreeClassifier` on global population benchmarks. You will trace decision boundaries via tree visualization, evaluate multi-class metrics (precision, recall, and F1-score), and map the bias-variance tradeoff by sweeping `max_depth` to prune away overfit noise. It is the exact interpretability sandbox required to master human-readable logic before deploying algorithms to high-stakes agricultural fields.

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
    The regression work is done. The Maji Ndogo energy shortfall model is built, validated, and saved. Sanaa has filed the report with the Department of Energy.

    Now she's standing at the whiteboard again, this time with a different question.

    "Regression tells us *how much*," she says. "But I need to know *which one*. Which crop grows best in this field? Is this plot suitable for a new irrigation scheme? Is this soil type too degraded to plant on at all? Those are not numbers — they are categories. And for categories, we need a different kind of model."

    She draws a branching tree structure on the whiteboard. "Decision Trees. They ask a series of yes-or-no questions about the data and funnel each observation toward a prediction. Simple to understand, fast to train, and — critically for our farmers — the decision logic is *human-readable*. We can show a field officer exactly why the model said 'plant cassava here'."

    But before we point a Decision Tree at Maji Ndogo's agricultural data — which is noisy, high-stakes, and feeds directly into planting decisions that affect thousands of farmers — Sanaa wants proof of concept on safer ground.

    > *"Take a dataset where the patterns are well-understood and the stakes are low. Prove the model is working. Then we'll trust it with our fields."*

    She has suggested the **World Population dataset**. Population growth patterns are stable and well-documented. If our Decision Tree can correctly classify countries into population tiers from their historical growth data, we know the algorithm is functioning correctly. This is the trial run before the field test.

    By the end of this notebook you will have:
    - Trained your first `DecisionTreeClassifier` in scikit-learn.
    - Visualized the tree to interpret its decision logic.
    - Evaluated it using accuracy, precision, recall, and F1-score.
    - Mapped the bias-variance tradeoff by sweeping `max_depth`.
    - Identified the optimal depth through pruning validation.

    > **AI assist:** Decision tree logic can be abstract. Good prompts: *"Explain the Gini impurity criterion in plain language — what is the model trying to minimize at each split?"* *"What does it mean when a decision tree's training accuracy is 1.0 but test accuracy is 0.6?"*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Data Dictionary

    The World Population dataset contains country-level population and area statistics.

    | Column | Description | Type |
    |---|---|---|
    | `Rank` | Global population rank | Int |
    | `CCA3` | Three-letter country code | String |
    | `Country/Territory` | Full country name | String |
    | `Capital` | Capital city name | String |
    | `Continent` | Continent | String |
    | `2022 Population` | Population in 2022 | Int |
    | `2020 Population` | Population in 2020 | Int |
    | `2015 Population` | Population in 2015 | Int |
    | `2010 Population` | Population in 2010 | Int |
    | `2000 Population` | Population in 2000 | Int |
    | `1990 Population` | Population in 1990 | Int |
    | `1980 Population` | Population in 1980 | Int |
    | `1970 Population` | Population in 1970 | Int |
    | `Area (km²)` | Total land area in square kilometers | Float |
    | `Density (per km²)` | Population density per square kilometer | Float |
    | `Growth Rate` | Annual population growth rate | Float |
    | `World Population Percentage` | Share of global population (%) | Float |

    **Engineered target:** `Pop_class` — classifies each country as `Low`, `Medium`, or `High` based on its 2022 population using tertile thresholds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Setup

    Run the cell below to load the dataset. Replace `file_path` with your local path if running outside the graded environment.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # headless rendering for test environments
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder

    file_path = 'world_population.csv'
    df_raw = pd.read_csv(file_path)

    print(df_raw.shape)
    print(df_raw.head())
    return (
        DecisionTreeClassifier,
        LabelEncoder,
        accuracy_score,
        classification_report,
        df_raw,
        pd,
        plot_tree,
        plt,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Data Preparation

    Before modeling, we engineer the target variable and select our feature set.

    **Target engineering:** The 2022 population is divided into three classes using tertile thresholds:
    - `Low`: Below the 33rd percentile.
    - `Medium`: Between the 33rd and 66th percentiles.
    - `High`: Above the 66th percentile.

    **Features:** Historical population figures (1970–2020), land area, population density, growth rate, and world population share.

    Run the cell below — it creates `df` (the modeling-ready DataFrame) and `le` (the LabelEncoder) which you will use throughout all challenges.
    """)
    return


@app.cell
def _(LabelEncoder, df_raw):
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

    feature_cols = [
        '1970 Population', '1980 Population', '1990 Population',
        '2000 Population', '2010 Population', '2015 Population',
        '2020 Population', 'Area (km²)', 'Density (per km²)',
        'Growth Rate', 'World Population Percentage'
    ]

    df = df_raw[feature_cols + ['Pop_class']].dropna().copy()

    le = LabelEncoder()
    df['Pop_class_encoded'] = le.fit_transform(df['Pop_class'])

    print('Class distribution:')
    print(df['Pop_class'].value_counts())
    print(f'\nDataFrame shape: {df.shape}')
    print(f'Class labels: {list(le.classes_)}')
    return df, feature_cols, le


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 1: Training a Decision Tree Classifier

    We have features and a target. Time to build our first classifier.

    A Decision Tree makes predictions by learning a series of if-else rules from the training data. At each split, it chooses the feature and threshold that best separates the classes — minimizing **Gini impurity** (or entropy). This is the classification analog of minimizing mean squared error in regression trees.

    ### Task
    Create a function named `train_decision_tree` that:
    - Takes a DataFrame, the list of feature columns, the encoded target column name, a `max_depth` parameter (default `3`), and `random_state` (default `42`).
    - Splits the data 80-20 using the given `random_state`.
    - Trains a `DecisionTreeClassifier` with the given `max_depth` on the training set.
    - Returns `(model, X_test, y_test)`.

    **Note:**
    - Use `DecisionTreeClassifier` from `sklearn.tree`.
    - Return format: `(model, X_test, y_test)`.

    ### Expected Output
    ```
    Model max_depth: 3
    Test set size: 47
    Number of leaves: 3
    ```
    """)
    return


@app.cell
def _(DecisionTreeClassifier, train_test_split):
    ### START FUNCTION
    def train_decision_tree(df, feature_cols, target_col, max_depth=3, random_state=42):
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        return model, X_test, y_test
    ### END FUNCTION
    return (train_decision_tree,)


@app.cell
def _(df, feature_cols, train_decision_tree):
    # Input:
    model, X_test, y_test = train_decision_tree(df, feature_cols, 'Pop_class_encoded')

    print(f'Model max_depth: {model.max_depth}')
    print(f'Test set size: {X_test.shape[0]}')
    print(f'Number of leaves: {model.get_n_leaves()}')
    return X_test, model, y_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With `max_depth=3`, the tree can ask at most three questions before reaching a decision — and it uses those three splits to reach up to 8 leaf nodes. This is already capturing meaningful structure. In Challenge 4 we will see what happens when we allow the tree to grow deeper.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 2: Visualizing the Tree

    One of the most powerful features of Decision Trees is their interpretability. Unlike regression coefficients that require domain expertise to interpret, we can literally **draw** a Decision Tree and trace how it reaches any prediction.

    ### Task
    Create a function named `visualize_tree` that:
    - Takes the fitted model, the list of feature column names, and the list of class names.
    - Plots the decision tree using `sklearn.tree.plot_tree` with `filled=True` and `rounded=True`.
    - Uses `plt.figure(figsize=(20, 8))` for readability.
    - Adds a descriptive title.
    - Returns the total number of nodes in the tree (`model.tree_.node_count`).

    **Note:** The number of nodes includes both decision nodes and leaf nodes.

    ### Expected Output
    ```
    Total nodes in tree: 5
    ```
    """)
    return


@app.cell
def _(plot_tree, plt):
    ### START FUNCTION
    def visualize_tree(model, feature_names, class_names):
        plt.figure(figsize=(20, 8))
        plot_tree(model, feature_names=feature_names, class_names=class_names,
                  filled=True, rounded=True)
        plt.title('Decision Tree - Population Class Prediction')
        return model.tree_.node_count
    ### END FUNCTION
    return (visualize_tree,)


@app.cell
def _(feature_cols, le, model, visualize_tree):
    # Input:
    n_nodes = visualize_tree(model, feature_cols, list(le.classes_))
    print(f'Total nodes in tree: {n_nodes}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at the tree diagram. Which feature appears at the root — the very first split? This is the single most informative predictor for separating population classes. In Decision Trees, the root node feature is determined by which split maximally reduces Gini impurity across the full training set. Does the choice make intuitive sense given the data?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 3: Evaluating the Classifier

    A diagram is useful for interpretation, but decisions need numbers. We will use **accuracy** — the proportion of test samples correctly classified — and the **classification report**, which breaks performance down by class with precision, recall, and F1-score.

    ### Task
    Create a function named `evaluate_tree_classifier` that:
    - Takes the fitted model, `X_test`, `y_test`, and the `LabelEncoder` object `le`.
    - Generates predictions on the test set.
    - Uses `le.inverse_transform` to convert encoded predictions and labels back to readable class names.
    - Prints the full classification report.
    - Returns overall accuracy as a float.

    **Note:**
    - Use `accuracy_score` and `classification_report` from `sklearn.metrics`.
    - Return format: `float` (accuracy only).

    ### Expected Output
    ```
                  precision    recall  f1-score   support

            High       1.00      1.00      1.00        14
             Low       1.00      1.00      1.00        19
          Medium       1.00      1.00      1.00        14

        accuracy                           1.00        47
       macro avg       1.00      1.00      1.00        47
    weighted avg       1.00      1.00      1.00        47

    Overall accuracy: 1.0000
    ```
    """)
    return


@app.cell
def _(accuracy_score, classification_report):
    ### START FUNCTION
    def evaluate_tree_classifier(model, X_test, y_test, le):
        y_pred = model.predict(X_test)
        y_pred_labels = le.inverse_transform(y_pred)
        y_test_labels = le.inverse_transform(y_test)
        print(classification_report(y_test_labels, y_pred_labels))
        return accuracy_score(y_test, y_pred)
    ### END FUNCTION
    return (evaluate_tree_classifier,)


@app.cell
def _(X_test, evaluate_tree_classifier, le, model, y_test):
    # Input:
    accuracy = evaluate_tree_classifier(model, X_test, y_test, le)
    print(f'\nOverall accuracy: {accuracy:.4f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Which class does the model struggle with most? Often it is `Medium` — the boundary cases that are neither clearly large nor clearly small populations. This is a common pattern in multi-class classification: the middle class sits at the intersection of two decision boundaries, and a single tree with limited depth may not have enough splits to separate it cleanly. Ensemble methods (Module 2) address exactly this weakness.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 4: The Effect of Tree Depth — Bias-Variance in Classification

    The bias-variance tradeoff applies to classification just as it does to regression. A shallow tree underfits (high bias — it cannot capture the true patterns). A very deep tree overfits (high variance — it memorizes the training data, including noise, and fails to generalize).

    Plotting training and test accuracy against `max_depth` makes this tradeoff visible: training accuracy rises monotonically, while test accuracy peaks and then plateaus or drops.

    ### Task
    Create a function named `depth_vs_accuracy` that:
    - Takes the DataFrame, feature columns, encoded target column, and a list of depths to test.
    - For each depth, trains a `DecisionTreeClassifier` (80-20 split, `random_state=42`) and records both training and test accuracy.
    - Plots both curves on the same figure with a legend, axis labels, and title.
    - Returns a DataFrame with columns `depth`, `train_accuracy`, and `test_accuracy`.

    ### Expected Output
    ```
        depth  train_accuracy  test_accuracy
    0       1        0.684492       0.595745
    1       2        1.000000       1.000000
    2       3        1.000000       1.000000
    3       4        1.000000       1.000000
    4       5        1.000000       1.000000
    5       6        1.000000       1.000000
    6       7        1.000000       1.000000
    7       8        1.000000       1.000000
    8       9        1.000000       1.000000
    9      10        1.000000       1.000000
    10     11        1.000000       1.000000
    11     12        1.000000       1.000000
    12     13        1.000000       1.000000
    13     14        1.000000       1.000000
    14     15        1.000000       1.000000
    ```
    """)
    return


@app.cell
def _(DecisionTreeClassifier, accuracy_score, pd, plt, train_test_split):
    ### START FUNCTION
    def depth_vs_accuracy(df, feature_cols, target_col, depths):
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_accs, test_accs = [], []
        for d in depths:
            m = DecisionTreeClassifier(max_depth=d, random_state=42)
            m.fit(X_train, y_train)
            train_accs.append(accuracy_score(y_train, m.predict(X_train)))
            test_accs.append(accuracy_score(y_test, m.predict(X_test)))

        plt.figure()
        plt.plot(depths, train_accs, label='Train accuracy')
        plt.plot(depths, test_accs, label='Test accuracy')
        plt.xlabel('max_depth')
        plt.ylabel('Accuracy')
        plt.title('Bias-Variance: Accuracy vs Tree Depth')
        plt.legend()

        return pd.DataFrame({'depth': depths, 'train_accuracy': train_accs, 'test_accuracy': test_accs})
    ### END FUNCTION
    return (depth_vs_accuracy,)


@app.cell
def _(depth_vs_accuracy, df, feature_cols):
    # Input:
    depths = list(range(1, 16))
    results_df = depth_vs_accuracy(df, feature_cols, 'Pop_class_encoded', depths)
    print(results_df)
    return (results_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at the plot. At which depth does test accuracy peak before leveling off or declining? That is roughly the sweet spot where the tree captures real patterns without memorizing noise. Notice that training accuracy eventually reaches 1.0 (perfect) — the tree has enough splits to memorize every training example. The divergence between the two curves is the visual signature of overfitting. In Module 2, we will use ensemble methods to push past this limit without increasing the risk of a single-tree overfit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Challenge 5: Pruning Validation — Finding the Optimal `max_depth`

    Challenge 4 showed us the tradeoff visually. Now we formalize it: identify the `max_depth` that maximizes test accuracy and train the final benchmark model at that depth.

    ### Task
    Create a function named `find_optimal_depth` that:
    - Takes the results DataFrame from Challenge 4, the modeling DataFrame, feature columns, target column, and `random_state` (default `42`).
    - Identifies the `max_depth` with the highest `test_accuracy`.
    - If multiple depths tie, selects the shallowest (simpler models are preferred when performance is equal).
    - Trains a final `DecisionTreeClassifier` at the optimal depth on an 80-20 split with the given `random_state`.
    - Returns `(optimal_depth, final_model)`.

    ### Expected Output
    ```
    Optimal max_depth: 2
    Number of leaves in final model: 3
    ```
    """)
    return


@app.cell
def _(DecisionTreeClassifier, train_test_split):
    ### START FUNCTION
    def find_optimal_depth(results_df, df, feature_cols, target_col, random_state=42):
        best_row = results_df.sort_values(
            ['test_accuracy', 'depth'], ascending=[False, True]
        ).iloc[0]
        optimal_depth = int(best_row['depth'])

        X = df[feature_cols]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        final_model = DecisionTreeClassifier(max_depth=optimal_depth, random_state=random_state)
        final_model.fit(X_train, y_train)

        return optimal_depth, final_model
    ### END FUNCTION
    return (find_optimal_depth,)


@app.cell
def _(df, feature_cols, find_optimal_depth, results_df):
    # Input:
    optimal_depth, final_model = find_optimal_depth(results_df, df, feature_cols, 'Pop_class_encoded')
    print(f'Optimal max_depth: {optimal_depth}')
    print(f'Number of leaves in final model: {final_model.get_n_leaves()}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The optimal depth is the result of Sanaa's validation requirement: a model complex enough to be useful, but shallow enough to generalize. In production, this pruned model is what we would deploy — not the depth-15 memorizer, and not the depth-1 guesser.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Wrapping up

    The Global Benchmark Trial is complete. The World Population dataset gave us a clean, low-stakes environment to prove that Decision Trees can learn from multi-feature data and make sensible multi-class predictions.

    **Key findings:**

    | Finding | Detail |
    |---|---|
    | Shallow tree (depth 3) | Reasonable accuracy, fully interpretable — every decision can be traced |
    | Deep trees | Training accuracy → 1.0; test accuracy peaks then levels off — classic overfitting signature |
    | Optimal depth | The pruning sweet spot: complex enough to capture real patterns, simple enough to generalize |

    The Maji Ndogo agricultural data is messier and higher-stakes than world population counts. But we have now proven that the decision tree framework is sound. We know how to train it, visualize it, evaluate it, and tune it.

    **What Sanaa asked for:** proof that the model works before we trust it with the fields.

    **What we delivered:** a complete benchmarking workflow — train, visualize, evaluate, sweep depths, prune — that will transfer directly to the crop classification task waiting in Module 2.
    """)
    return


if __name__ == "__main__":
    app.run()
