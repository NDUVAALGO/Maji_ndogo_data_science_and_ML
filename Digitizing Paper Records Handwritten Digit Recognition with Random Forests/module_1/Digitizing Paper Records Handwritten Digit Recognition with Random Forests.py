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
    # Image Classification with Random Forests
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Paper records lock away historical value; automated ingestion pipelines unlock it. To transition from manual filing cabinets to an integrated digital archive, the Ministry of Agriculture requires an engine capable of parsing physical survey data. In this project, you construct a scalable document digitization component by training a `RandomForestClassifier` on the Maji Ndogo Digit Calibration Bank — a purpose-built digit benchmark assembled by the ministry's digitisation pilot team. You will configure parametric subset extraction, normalize raw pixel intensity matrices to uniform numerical feature scales, and deploy multi-class confusion evaluation metrics to target classification bottlenecks. This establishes the structural pattern-recognition framework necessary to safely ingest legacy regional records prior to unlocking high-stakes national field databases.

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
    The Ministry of Agriculture's Maji Ndogo programme holds a critical problem: decades of farm survey records exist only on paper. Thousands of handwritten field IDs, yield measurements, and crop notes are locked away in filing cabinets across the country, unable to feed into the national agricultural database.

    Before the ministry can trust us with the national farm survey database — which we will work with in Module 3 — they need us to prove something first: that we can build a reliable automated system for reading digits scanned off paper forms. This is the foundation of any digitisation pipeline. If we can accurately classify digits, the ministry can begin scanning and automatically ingesting decades of archived paper surveys.

    Your assignment this module is to build a **Random Forest image classifier** capable of recognising digits. Rather than risk real, irreplaceable field archives on an unproven pipeline, the digitisation pilot team built a calibration set first: the **Maji Ndogo Digit Calibration Bank** — 35,000 digit images rendered from the actual typefaces used on the ministry's historical paper forms, each one distorted with the rotation, skew, and scan noise you'd expect from decades-old documents run through a low-cost scanner. It is deliberately messy, for the same reason a pilot's flight simulator is deliberately turbulent: if the model can handle this, it can handle the real archive. The model you build here is the same type of component that would sit at the front of a real document digitisation system.

    By the end of this project, the ministry will have confidence in your classification skills, and you will have earned access to the full agricultural data in Module 3.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports

    Let's load the libraries we need.
    """)
    return


@app.cell
def _():
    import numpy as np
    import gzip  # Used to extract the compressed digit images
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    return (
        RandomForestClassifier,
        accuracy_score,
        classification_report,
        gzip,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Dataset

    We are using the **Maji Ndogo Digit Calibration Bank** — a purpose-built collection of digit images (0–9) assembled specifically to stress-test this pipeline before it ever touches a real archive. It requires little to no preprocessing, making it the ideal starting point for building our digitisation component.

    The four files are provided alongside this notebook — no download required. The cell below simply confirms they're present in your working directory.
    """)
    return


@app.cell
def _():
    import os

    DATA_FILES = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    missing = [f for f in DATA_FILES if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"Missing data file(s): {missing}. Make sure they are in the same "
            "folder as this notebook."
        )

    print("All four Digit Calibration Bank files are present.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two helper functions below will extract the data for you. **Do not change this code.**
    """)
    return


@app.cell
def _(gzip, np):
    def extract_data(filename, num_images, IMAGE_WIDTH):
        """Extract the images into a 2D array [image index, flattened pixels]."""
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = data.reshape(num_images, IMAGE_WIDTH * IMAGE_WIDTH)
            return data


    def extract_labels(filename, num_images):
        """Extract the labels into a vector of int64 label IDs."""
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels

    return extract_data, extract_labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 1: Extracting the Data

    The Digit Calibration Bank ships as four separate gzip files — two for training and two for testing, one for images and one for labels in each pair. Before we can train anything, we need to unpack them into numpy arrays and normalise the pixel values so every feature lives on the same scale. This is the data ingestion step: the moment the raw bytes on disk become the matrices our model will learn from.

    Sanaa needs this pipeline to be parameterised. The full training bank has 30,000 images and training on all of them takes time. During development we use a smaller subset; once we are confident the model works, we scale up. Your function must accept the subset size as an argument so we can swap it out without rewriting any code.

    ### Task

    Write a function `get_data` that uses the helper functions above to extract a specified number of images and their labels from the gzip files.

    **Function specifications:**
    - Takes two integers as input: `num_train_images` and `num_test_images`.
    - Returns two tuples of the form `(X_train, y_train), (X_test, y_test)`.
    - Normalises image pixel values from the range 0–255 to the range 0–1.
    - The images are **28×28 pixels**.
    - Do **not** shuffle the data — we keep the order fixed so all results are reproducible.

    > **Note:** The filenames expected by your function are:
    > - `'train-images-idx3-ubyte.gz'`
    > - `'train-labels-idx1-ubyte.gz'`
    > - `'t10k-images-idx3-ubyte.gz'`
    > - `'t10k-labels-idx1-ubyte.gz'`

    ### Expected Output
    ```
    (5000,)
    (1000,)
    (5000, 784)
    (1000, 784)
    ```
    """)
    return


@app.cell
def _(extract_data, extract_labels):
    def get_data(num_train_images, num_test_images):
        IMAGE_WIDTH = 28

        X_train = extract_data('train-images-idx3-ubyte.gz', num_train_images, IMAGE_WIDTH)
        y_train = extract_labels('train-labels-idx1-ubyte.gz', num_train_images)
        X_test = extract_data('t10k-images-idx3-ubyte.gz', num_test_images, IMAGE_WIDTH)
        y_test = extract_labels('t10k-labels-idx1-ubyte.gz', num_test_images)

        X_train = X_train / 255.0
        X_test = X_test / 255.0

        return (X_train, y_train), (X_test, y_test)

    return (get_data,)


@app.cell
def _(get_data):
    (X_train, y_train), (X_test, y_test) = get_data(5000, 1000)

    print(y_train.shape)
    print(y_test.shape)
    print(X_train.shape)
    print(X_test.shape)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting the Data

    Let's see what the data looks like. Each image is stored as a flattened 1D array of 784 values. We need to reshape it back to 28×28 pixels to visualise it.
    """)
    return


@app.cell
def _(X_train, plt, y_train):
    image_index = 3  ## Change this to view different images

    print("Label: ", y_train[image_index])
    reshaped_image = X_train[image_index].reshape((28, 28))

    plt.imshow(reshaped_image, cmap='gray')
    plt.title(f'Handwritten digit: {y_train[image_index]}')
    plt.axis('off')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 2: Training the Model

    The data is loaded, normalised, and ready. Now we hand it to a classifier. We are using sklearn's `RandomForestClassifier` — a homogeneous ensemble that grows many decision trees on random bootstrap samples of the training data and lets them vote on the final prediction. A single decision tree is brittle; one bad split near the root and the whole tree goes wrong. The forest averages out that instability across all its trees.

    This is the same ensemble architecture Sanaa wants to use on the Maji Ndogo field registry in Module 3. We are proving the method works on the calibration bank first — a clean, well-understood benchmark — before we trust it with real agricultural decisions.

    ### Task

    Write a function `train_model` that:
    - Takes two numpy arrays as input: `X_train` and `y_train`.
    - Returns a fitted `RandomForestClassifier` with:
      - `n_estimators = 20`
      - `random_state = 42`

    ### Expected Output
    ```
    Model type: RandomForestClassifier
    Number of estimators: 20
    ```
    """)
    return


@app.cell
def _(RandomForestClassifier):
    def train_model(X_train, y_train):
        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        clf.fit(X_train, y_train)
        return clf

    return (train_model,)


@app.cell
def _(X_train, train_model, y_train):
    clf = train_model(X_train, y_train)
    print(f'Model type: {type(clf).__name__}')
    print(f'Number of estimators: {clf.n_estimators}')
    return (clf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 3: Calculating Accuracy

    The model is trained. Before we hand anything to Sanaa, we need to know how well it performs on data it has never seen. Accuracy — the proportion of test images classified correctly — is the first number she will ask for. It gives us a single figure that anchors the conversation: is this model worth discussing further, or do we go back to the drawing board?

    A note of caution that we will return to in Challenge 4: accuracy can be misleading when classes are imbalanced. For this calibration bank the ten digit classes are roughly balanced, so accuracy is a fair summary here. Keep that caveat in mind for the Maji Ndogo work ahead.

    ### Task

    Write a function `calculate_accuracy` that:
    - Takes the fitted model `clf` and two numpy arrays `X_test`, `y_test` as input.
    - Returns a **float** representing the accuracy (between 0 and 1).

    ### Expected Output
    ```
    0.891
    ```
    """)
    return


@app.cell
def _(accuracy_score):
    def calculate_accuracy(clf, X_test, y_test):
        y_pred = clf.predict(X_test)
        return accuracy_score(y_test, y_pred)

    return (calculate_accuracy,)


@app.cell
def _(X_test, calculate_accuracy, clf, y_test):
    print(calculate_accuracy(clf, X_test, y_test))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 4: Classification Report

    Accuracy alone doesn't tell the full story. A classification report shows us **precision**, **recall**, and **F1-score** for each digit class — revealing exactly where the model makes Type I and Type II errors. This is critical information for the ministry: misreading a `1` as a `7` in a field ID would link a survey record to the wrong farm.

    Consider what a precision error means in this context — we flag an image as a particular digit when it is actually something else. And a recall error means we fail to recognise a digit that was genuinely there. Both types of mistake introduce corruption into the digitised survey archive. Sanaa needs to see the full breakdown before she signs off on deploying this component.

    ### Task

    Write a function `get_class_report` that:
    - Takes the fitted model `clf` and two numpy arrays `X_test`, `y_test` as input.
    - Returns a classification report.

    > **Hint:** sklearn has a `classification_report` function.

    ### Expected Output
    ```
                  precision    recall  f1-score   support

               0       0.90      0.98      0.94       105
               1       0.90      0.93      0.92        89
               2       0.93      0.94      0.94       102
               3       0.87      0.93      0.90       103
               4       0.89      0.98      0.93        87
               5       0.91      0.77      0.83       107
               6       0.89      0.78      0.83       109
               7       0.90      0.97      0.93       102
               8       0.79      0.86      0.82        99
               9       0.94      0.79      0.86        97

        accuracy                           0.89      1000
       macro avg       0.89      0.89      0.89      1000
    weighted avg       0.89      0.89      0.89      1000
    ```
    """)
    return


@app.cell
def _(classification_report):
    def get_class_report(clf, X_test, y_test):
        y_pred = clf.predict(X_test)
        return classification_report(y_test, y_pred)

    return (get_class_report,)


@app.cell
def _(X_test, clf, get_class_report, y_test):
    print(get_class_report(clf, X_test, y_test))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting the Results

    Let's visually inspect some predictions to confirm the model is working correctly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected Output
    ```
    Predicted Label:  0
    ```
    """)
    return


@app.cell
def _(X_test, clf, plt):
    preds = clf.predict(X_test)
    image_index_1 = 1
    print('Predicted Label: ', preds[image_index_1])  ## Change this to see other predictions
    plt.imshow(X_test[image_index_1].reshape((28, 28)), cmap='gray')
    plt.title(f'Predicted: {preds[image_index_1]}')
    plt.axis('off')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wrapping up

    89.1% accuracy is a strong result for a Random Forest with only 20 trees and 5,000 training images. The ministry is satisfied.

    But notice something from the classification report: digits `5`, `6`, and `8` have the lowest F1-scores. These are the digits most distorted by the rotation and shear jitter baked into the calibration bank — a reasonable stand-in for the skewed, faded strokes a real decades-old paper form would produce.

    **Try this:** Go back and change `get_data(5000, 1000)` to use more training images. How does accuracy change as you increase from 5,000 to 10,000 to 30,000? This exploration mirrors exactly the kind of model experimentation we will perform in Module 3 when we compare classifiers on the Maji Ndogo farm data.

    The digitisation pipeline component is built. In Module 2, we will prove our ability to classify complex chemical signatures before the ministry grants us full access to the agricultural database.
    """)
    return


if __name__ == "__main__":
    app.run()
