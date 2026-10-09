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
    # The Language of the Land
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Unstructured text requires systematic, rule-based distillation and numerical vectorization before it can feed downstream modeling engines. When raw text logs span multi-lingual variants or informal officer-generated tag clouds, standard relational queries fail. In this project, you construct a dual-purpose linguistic parsing and classification system for the Maji Ndogo Ministry of Agriculture. In Part A, you use declarative **Regular Expressions (Regex)** to isolate multi-lingual patterns, parse invariant ISO timestamps, and map heterogeneous sensor messages into clean, continuous feature vectors. In Part B, you architect an end-to-end **Natural Language Processing (NLP)** pipeline over the extension service's field observation logs—normalizing raw token strings via morphological normalization (lemmatization and stemming), expanding feature context with contiguous $n$-grams, and applying a Term Frequency-Inverse Document Frequency (TF-IDF) transformation to map structural text patterns into predictive field-issue classes via Logistic Regression.

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
    You are the Maji Ndogo Ministry of Agriculture's longest-serving data consultant, and today two of its departments need the same core skill from you: the ability to turn messy, unstructured text into clean, structured data that machines can work with.

    **Part A** comes from the automated monitoring unit. The five weather stations that watch over the national farm network have been quietly logging temperature, rainfall, and air quality readings for over a year — but every reading is buried inside a raw text string. Some messages are in English. Some are in Chinese (the station firmware shipped from the manufacturer that way). There are thirteen distinct message formats, and none of them are in a form your models can use. Your job is to use **regular expressions** to crack open every message and extract the structured numbers hiding inside.

    **Part B** comes from the extension service. Field officers visit farms across the five provinces and log free-text observation tags in the Shamba Ndogo app — labels like *"chewed leaf"*, *"wilting canopy"*, and *"weeding overdue"*. Over 12,000 fields now have observation histories. The Ministry wants to know: can you predict whether a field has a **pest problem** from its observation tags alone, before an agronomist is dispatched? You will build a full **NLP pipeline** — preprocessing, tokenization, bag-of-words, n-grams, and TF-IDF — to answer that question.

    By the end of this project, you will have two portfolio-ready deliverables: a regex-powered data extraction engine and a text classification pipeline — both running on the same national agricultural data platform.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Instructions

    - Do **not** add or remove cells in this notebook.
    - Do **not** edit or remove the `### START FUNCTION` or `### END FUNCTION` comments.
    - Do **not** add any code outside of the functions you are required to edit.
    - Doing any of the above will lead to a mark of 0 for that question.
    - Answer the questions according to the specifications provided.
    - The tests included in this notebook are not exhaustive. The autograder will run additional tests.
    - Ensure your functions return the expected data types and structures.
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
    import re
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    import nltk
    nltk.download('punkt',     quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet',   quiet=True)
    nltk.download('omw-1.4',   quiet=True)

    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    return (
        CountVectorizer,
        LogisticRegression,
        PorterStemmer,
        TfidfVectorizer,
        WordNetLemmatizer,
        accuracy_score,
        classification_report,
        pd,
        re,
        stopwords,
        train_test_split,
        word_tokenize,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading the Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    Weather messages: (1843, 2)
    Field mapping:    (5654, 3)
    Fields:           (15000, 3)
    Field tags:       (119173, 4)

    Sample weather messages:
      Weather Update: As of 2023-11-18 04:45:09, rainfall stands at 1951.07mm.
      Temperature Read at [2022-11-21 17:03:35]: 13.88C.
      Weather Update: As of 2022-11-20 07:12:40, rainfall stands at 1902.71mm.
      Temperature Read at [2023-12-04 04:31:17]: 13.95C.
      Air Quality (2023-07-04 20:59:06): Pollution at 0.33.
      Rain Sensor (2022-12-26 21:20:49): 1013.77 mm precipitation detected.
    ```
    """)
    return


@app.cell
def _(pd):
    weather_df = pd.read_csv('Weather_station_data.csv')
    mapping_df = pd.read_csv('Weather_data_field_mapping.csv')
    fields = pd.read_csv('field_issues.csv')
    tags = pd.read_csv('field_tags.csv')
    print('Weather messages:', weather_df.shape)
    print('Field mapping:   ', mapping_df.shape)
    print('Fields:          ', fields.shape)
    print('Field tags:      ', tags.shape)
    print()
    print('Sample weather messages:')
    for _msg in weather_df['Message'].sample(6, random_state=42):
        print(' ', _msg)
    return fields, tags, weather_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Part A: Advanced String Manipulation — Maji Ndogo Weather Stations

    The weather station network in Maji Ndogo logs three types of readings: **temperature**, **rainfall**, and **air quality/pollution index**. The messages come in thirteen distinct formats across two languages (English and Chinese). Your job is to use regular expressions to extract structured information from this raw text.

    The complete set of message formats is:

    | Language | Type        | Example |
    |----------|-------------|---------|
    | English  | Temperature | `Temp. Reading [2023-05-23 09:41:36]: Current 14.53 C.` |
    | English  | Temperature | `Temperature Read at [2022-01-08 02:54:10]: 12.75C.` |
    | Chinese  | Temperature | `【2022-01-04 21:47:48】温度感应: 现在温度是 12.82C.` |
    | English  | Rainfall    | `Weather Update: As of 2022-08-29 06:44:16, rainfall stands at 1917.49mm.` |
    | English  | Rainfall    | `Rain Sensor (2023-03-12 22:57:18): 854.28 mm precipitation detected.` |
    | Chinese  | Rainfall    | `2022-05-13 01:35:33 雨量警告: 764.08 mm.` |
    | English  | Rainfall    | `2022-01-13 00:30:34, Detected 862.52 mm rainfall in the last 24 hours.` |
    | English  | Pollution   | `Air Quality (2022-09-26 16:17:27): Pollution at 0.06.` |
    | English  | Pollution   | `Pollution Index on 2022-12-08 06:49:18 = 0.26.` |
    | Chinese  | Pollution   | `环境监测报告: 2022-10-25 11:53:44, Air Quality Index = 0.18.` |

    Your goal: write functions that classify, parse, and extract every message into a clean table with one row per station.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 1: Classifying Message Type

    Before extracting numeric values, we need to know what each message is about.

    ### Task
    Write a function `classify_message_type` that identifies the type of reading in a message.

    **Function specifications:**
    - Takes a message string as input.
    - Returns `'temperature'` if the message is a temperature reading (English or Chinese).
    - Returns `'rainfall'` if the message is a rainfall reading (English or Chinese).
    - Returns `'pollution'` if the message is a pollution/air quality reading (English or Chinese).
    - Returns `'unknown'` if none of the above patterns match.

    > **Hint:** Look for distinctive keywords in each message type. Temperature messages contain words like `Temp` or `温度`. Rainfall messages mention `Rain`, `mm`, `雨量`, or `precipitation`. Pollution messages reference `Pollut`, `Quality`, `Index`, or `环境`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    type
    rainfall       625
    temperature    611
    pollution      607
    Name: count, dtype: int64
    ```
    """)
    return


@app.cell
def _(re):
    ### START FUNCTION
    def classify_message_type(msg):
        # your code here
        if re.search(r'Temp|温度', msg):
                return 'temperature'
        if re.search(r'[Rr]ain|mm|雨量|[Pp]recipitation', msg):
                return 'rainfall'
        if re.search(r'[Pp]ollut|[Qq]uality|[Ii]ndex|环境', msg):
                return 'pollution'
        return 'unknown'
    ### END FUNCTION
    return (classify_message_type,)


@app.cell
def _(classify_message_type, weather_df):
    weather_df['type'] = weather_df['Message'].apply(classify_message_type)
    print(weather_df['type'].value_counts())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 2: Extracting the Timestamp

    Every message contains a timestamp in the format `YYYY-MM-DD HH:MM:SS`. It may appear inside `【 】` brackets, inside `[ ]` brackets, or as a plain date prefix — but the underlying format is always the same.

    ### Task
    Write a function `extract_timestamp` that extracts the timestamp string from any message.

    **Function specifications:**
    - Takes a message string as input.
    - Returns the timestamp as a string in the format `'YYYY-MM-DD HH:MM:SS'`, or `None` if no timestamp is found.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    2022-01-04 21:47:48    |  【2022-01-04 21:47:48】温度感应: 现在温度是 12.82C.
    2023-05-23 09:41:36    |  Temp. Reading [2023-05-23 09:41:36]: Current 14.53 C.
    2022-08-29 06:44:16    |  Weather Update: As of 2022-08-29 06:44:16, rainfall st
    2022-10-25 11:53:44    |  环境监测报告: 2022-10-25 11:53:44, Air Quality Index =
    2022-12-08 06:49:18    |  Pollution Index on 2022-12-08 06:49:18 = 0.26.
    ```
    """)
    return


@app.cell
def _(re):
    ### START FUNCTION
    def extract_timestamp(msg):
        # your code here
        match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', msg)
        return match.group(1) if match else None
    ### END FUNCTION
    return (extract_timestamp,)


@app.cell
def _(extract_timestamp):
    _test_messages = ['【2022-01-04 21:47:48】温度感应: 现在温度是 12.82C.', 'Temp. Reading [2023-05-23 09:41:36]: Current 14.53 C.', 'Weather Update: As of 2022-08-29 06:44:16, rainfall stands at 1917.49mm.', '环境监测报告: 2022-10-25 11:53:44, Air Quality Index = 0.18.', 'Pollution Index on 2022-12-08 06:49:18 = 0.26.']
    for _msg in _test_messages:
        print(f'{str(extract_timestamp(_msg)):<22}  |  {_msg[:60]}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 3: Extracting Temperature

    ### Task
    Write a function `extract_temperature` that extracts the numeric temperature from a message.

    **Function specifications:**
    - Takes a message string as input.
    - Returns a **float** representing the temperature in Celsius, or `None` if the message is not a temperature reading.
    - Must handle all temperature message formats (English and Chinese).

    > **Hint:** All temperature messages end with a numeric value immediately followed by `C` or `C.` — look for a float directly before the terminal `C`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    14.53       |  Temp. Reading [2023-05-23 09:41:36]: Current 14.53 C.
    12.75       |  Temperature Read at [2022-01-08 02:54:10]: 12.75C.
    12.82       |  【2022-01-04 21:47:48】温度感应: 现在温度是 12.82C.
    None        |  Weather Update: As of 2022-08-29 06:44:16, rainfall st
    ```
    """)
    return


@app.cell
def _(re):
    ### START FUNCTION
    def extract_temperature(msg):
        # your code here
        match = re.search(r'(-?\d+\.?\d*)\s*C\.?', msg)
        return float(match.group(1)) if match else None
    
    ### END FUNCTION
    return (extract_temperature,)


@app.cell
def _(extract_temperature):
    _test_messages = ['Temp. Reading [2023-05-23 09:41:36]: Current 14.53 C.', 'Temperature Read at [2022-01-08 02:54:10]: 12.75C.', '【2022-01-04 21:47:48】温度感应: 现在温度是 12.82C.', 'Weather Update: As of 2022-08-29 06:44:16, rainfall stands at 1917.49mm.']
    for _msg in _test_messages:
        print(f'{str(extract_temperature(_msg)):<10}  |  {_msg[:60]}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge: Extracting Rainfall

    ### Task
    Write a function `extract_rainfall` that extracts the numeric rainfall measurement from a message.

    **Function specifications:**
    - Takes a message string as input.
    - Returns a **float** representing rainfall in mm, or `None` if the message is not a rainfall reading.
    - Must handle all rainfall message formats (English and Chinese).

    > **Hint:** All rainfall messages contain a numeric value followed by `mm` (with or without a space before it).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    1917.49       |  Weather Update: As of 2022-08-29 06:44:16, rainfall stands at 191
    854.28        |  Rain Sensor (2023-03-12 22:57:18): 854.28 mm precipitation detected
    764.08        |  2022-05-13 01:35:33 雨量警告: 764.08 mm.
    862.52        |  2022-01-13 00:30:34, Detected 862.52 mm rainfall in the last 24 ho
    None          |  Temp. Reading [2023-05-23 09:41:36]: Current 14.53 C.
    ```
    """)
    return


@app.cell
def _(re):
    ### START FUNCTION
    def extract_rainfall(msg):
        # your code here
        match = re.search(r'(\d+\.?\d*)\s*mm', msg)
        return float(match.group(1)) if match else None
    
    ### END FUNCTION
    return (extract_rainfall,)


@app.cell
def _(extract_rainfall):
    _test_messages = ['Weather Update: As of 2022-08-29 06:44:16, rainfall stands at 1917.49mm.', 'Rain Sensor (2023-03-12 22:57:18): 854.28 mm precipitation detected.', '2022-05-13 01:35:33 雨量警告: 764.08 mm.', '2022-01-13 00:30:34, Detected 862.52 mm rainfall in the last 24 hours.', 'Temp. Reading [2023-05-23 09:41:36]: Current 14.53 C.']
    for _msg in _test_messages:
        print(f'{str(extract_rainfall(_msg)):<12}  |  {_msg[:65]}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 5: Extracting Pollution Index

    ### Task
    Write a function `extract_pollution` that extracts the pollution/air quality index from a message.

    **Function specifications:**
    - Takes a message string as input.
    - Returns a **float** representing the pollution index, or `None` if the message is not a pollution reading.
    - Must handle all pollution message formats (English and Chinese), including **negative values**.

    > **Hint:** Pollution messages use three distinct patterns: `Pollution at X.XX`, `Index on DATE = X.XX`, and `Air Quality Index = X.XX`. The value can be negative.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    0.06        |  Air Quality (2022-09-26 16:17:27): Pollution at 0.06.
    0.26        |  Pollution Index on 2022-12-08 06:49:18 = 0.26.
    0.18        |  环境监测报告: 2022-10-25 11:53:44, Air Quality Index = 0.18.
    -0.04       |  Pollution Index on 2023-07-25 03:26:24 = -0.04.
    None        |  Rain Sensor (2023-03-12 22:57:18): 854.28 mm precipitati
    ```
    """)
    return


@app.cell
def _(re):
    ### START FUNCTION
    def extract_pollution(msg):
        # your code here
        match = re.search(r'(?:Pollution at|Index[^=]*=)\s*(-?\d+\.?\d*)', msg)
        return float(match.group(1)) if match else None
    
    ### END FUNCTION
    return (extract_pollution,)


@app.cell
def _(extract_pollution):
    _test_messages = ['Air Quality (2022-09-26 16:17:27): Pollution at 0.06.', 'Pollution Index on 2022-12-08 06:49:18 = 0.26.', '环境监测报告: 2022-10-25 11:53:44, Air Quality Index = 0.18.', 'Pollution Index on 2023-07-25 03:26:24 = -0.04.', 'Rain Sensor (2023-03-12 22:57:18): 854.28 mm precipitation detected.']
    for _msg in _test_messages:
        print(f'{str(extract_pollution(_msg)):<10}  |  {_msg[:65]}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 6: Building the Weather Feature Table

    Now that we can parse individual messages, we aggregate the readings into a summary table: one row per weather station, one column per measurement type.

    ### Task
    Write a function `build_weather_table` that produces this summary.

    **Function specifications:**
    - Takes `weather_df` as input.
    - Applies `extract_temperature`, `extract_rainfall`, and `extract_pollution` to the `Message` column.
    - For each `Weather_station_ID`, computes the **mean** of all available readings for each measurement type.
    - Returns a DataFrame with columns: `Weather_station_ID`, `Temperature`, `Rainfall`, `Pollution`, sorted by `Weather_station_ID`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
       Weather_station_ID  Temperature    Rainfall  Pollution
    0                   0      13.4039  1575.9538     0.3528
    1                   1      12.9989   577.3839     0.2573
    2                   2      13.1797  1690.9553     0.0439
    3                   3      13.2568   905.1914     0.2382
    4                   4      13.1886  1200.1835     0.1283
    ```
    """)
    return


@app.cell
def _(extract_pollution, extract_rainfall, extract_temperature):
    ### START FUNCTION
    def build_weather_table(weather_df):
        # your code here
        df = weather_df.copy()
        df['Temperature'] = df['Message'].apply(extract_temperature)
        df['Rainfall'] = df['Message'].apply(extract_rainfall)
        df['Pollution'] = df['Message'].apply(extract_pollution)
        result = (
            df.groupby('Weather_station_ID')[['Temperature', 'Rainfall', 'Pollution']]
            .mean()
            .reset_index()
            .sort_values('Weather_station_ID')
            .reset_index(drop=True)
            )
        return result
    ### END FUNCTION
    return (build_weather_table,)


@app.cell
def _(build_weather_table, weather_df):
    weather_table = build_weather_table(weather_df)
    print(weather_table.round(4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With this table, we can now join parsed weather features back to the farm survey database — exactly the enrichment that the Maji Ndogo crop classifier from DS-6 was waiting for. One Part A deliverable, five stations, three clean columns. Regex did the heavy lifting.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Part B: Natural Language Processing — Reading the Field Logs

    We turn now to the extension service. Field officers across Maji Ndogo's five provinces log free-text observation tags every time they visit a farm — quick labels like `'wilting canopy'`, `'chewed leaf'`, `'weeding overdue'`, and `'fertilizer needed'`. These tags are a window into what is actually happening on the ground, weeks before formal survey data arrives. They are also a rich NLP playground: noisy, informal, multi-word, and full of redundancy.

    The Ministry's question: **can you predict a field's diagnosed issues from its observation tags alone — no lab results, no survey forms?** Specifically, they want to know whether pest-affected fields leave a distinct textual fingerprint in how officers tag them. If they do, the model can triage which fields get an agronomist visit first.

    Each field in `field_issues.csv` carries a pipe-separated `issues` column from the formal diagnosis registry (e.g. `Pest|Drought`) — that gives us labels. The tags in `field_tags.csv` give us the text. To connect them, we will build a text classification pipeline from scratch: preprocessing, tokenization, bag-of-words, n-grams, TF-IDF, and logistic regression.
    """)
    return


@app.cell
def _(PorterStemmer, WordNetLemmatizer, stopwords):
    ### Initialize NLP tools — run this cell before the questions below
    stop_words = set(stopwords.words('english'))
    stemmer    = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    return lemmatizer, stemmer, stop_words


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 7: Text Preprocessing Pipeline

    Every NLP project begins with the same foundation: get the raw text into a clean, normalized, reduced form that a machine can work with meaningfully.

    ### Task
    Write a function `preprocess_text` that applies a full cleaning pipeline to a string.

    **Function specifications:**
    - Takes `text` (string) and `use_stemming` (bool, default `False`) as input.
    - Converts to lowercase.
    - Removes all non-alphabetic characters (keep only letters and spaces).
    - Tokenizes using `word_tokenize`.
    - Removes stopwords (use the `stop_words` set initialized above) and single-character tokens.
    - If `use_stemming=True`, applies `PorterStemmer` to each token.
    - If `use_stemming=False` (default), applies `WordNetLemmatizer` to each token.
    - Returns the processed tokens joined into a single space-separated string.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    Original:    The fields are experiencing flooding and the crops are dying quickly!
    Lemmatized:  field experiencing flooding crop dying quickly
    Stemmed:     field experienc flood crop die quickly
    ```
    """)
    return


@app.cell
def _(lemmatizer, re, stemmer, stop_words, word_tokenize):
    ### START FUNCTION
    def preprocess_text(text, use_stemming=False):
        # your code here
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        if use_stemming:
            tokens = [stemmer.stem(t) for t in tokens]
        else:
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)
    ### END FUNCTION
    return (preprocess_text,)


@app.cell
def _(preprocess_text):
    sample = 'The fields are experiencing flooding and the crops are dying quickly!'
    print("Original:   ", sample)
    print("Lemmatized: ", preprocess_text(sample, use_stemming=False))
    print("Stemmed:    ", preprocess_text(sample, use_stemming=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the difference: lemmatization preserves recognizable English words (`dying` → `dying`, `experiencing` → `experiencing`), while stemming aggressively chops to roots (`dying` → `die`, `experiencing` → `experienc`) that are faster to compute but less readable. For downstream classifiers, the stemmed version often performs similarly to the lemmatized one — the vocabulary shrinks, which can reduce noise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 8: Building the Tag Corpus

    Before we can vectorize, we need one text document per field — all of that field's observation tags concatenated into a single string.

    ### Task
    Write a function `build_tag_corpus` that aggregates officer tags per field.

    **Function specifications:**
    - Takes the `tags` DataFrame as input.
    - Drops rows with missing `tag` values.
    - Converts all tags to lowercase.
    - Groups by `fieldId`, joining all tags for that field into a single space-separated string.
    - Applies `preprocess_text` to each aggregated string (default `use_stemming=False`).
    - Returns a DataFrame with columns `['fieldId', 'clean_tags']`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    Fields with tag corpus: 12000

    Field 1 — clean tags (first 100 chars):
    termite mound foggy curling aphid damage grasshopper pressure windy browning struggling groundnut ti...
    ```
    """)
    return


@app.cell
def _(preprocess_text):
    ### START FUNCTION
    def build_tag_corpus(tags):
        # your code here
        df = tags.dropna(subset=['tag']).copy()
        df['tag'] = df['tag'].str.lower()
        grouped = (
            df.groupby('fieldId')['tag']
            .apply(lambda x: ' '.join(x))
            .reset_index()
        )
        grouped.columns = ['fieldId', 'clean_tags']
        grouped['clean_tags'] = grouped['clean_tags'].apply(
            lambda t: preprocess_text(t, use_stemming=False)
        )
        return grouped
    ### END FUNCTION
    return (build_tag_corpus,)


@app.cell
def _(build_tag_corpus, tags):
    tag_corpus = build_tag_corpus(tags)

    print(f"Fields with tag corpus: {len(tag_corpus)}")
    print(f"\nField 1 — clean tags (first 100 chars):")
    print(tag_corpus[tag_corpus['fieldId'] == 1]['clean_tags'].values[0][:100])
    return (tag_corpus,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 9: Bag-of-Words Vectorization

    The **bag-of-words** model converts each field's tag document into a vector of word counts. Each dimension of the vector corresponds to one term in the vocabulary; the value is how many times that term appears in that field's tag document.

    ### Task
    Write a function `fit_bow_vectorizer` that fits a `CountVectorizer` on the tag corpus and returns both the sparse matrix and the fitted vectorizer.

    **Function specifications:**
    - Takes `tag_corpus` (DataFrame with `'clean_tags'` column) and `max_features` (int, default `500`) as input.
    - Fits a `CountVectorizer` with `max_features` set to the given value and `stop_words='english'`.
    - Returns a tuple `(bow_matrix, vectorizer)` where `bow_matrix` is the fitted sparse matrix and `vectorizer` is the fitted `CountVectorizer` object.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    Bag-of-Words matrix shape: (12000, 500)

    Top 10 most frequent words:
    leaf         3697
    needed       2520
    spreading    2378
    patch        1844
    sprayed      1668
    damage       1545
    canopy       1530
    topsoil      1403
    applied      1358
    furrow       1323
    dtype: int64
    ```
    """)
    return


@app.cell
def _(CountVectorizer):
    ### START FUNCTION
    def fit_bow_vectorizer(tag_corpus, max_features=500):
        # your code here
        vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english'
            )
        bow_matrix = vectorizer.fit_transform(tag_corpus['clean_tags'])
        return bow_matrix, vectorizer
    ### END FUNCTION
    return (fit_bow_vectorizer,)


@app.cell
def _(fit_bow_vectorizer, pd, tag_corpus):
    bow_matrix, bow_vectorizer = fit_bow_vectorizer(tag_corpus, max_features=500)

    print(f"Bag-of-Words matrix shape: {bow_matrix.shape}")
    word_freq = pd.Series(
        bow_matrix.toarray().sum(axis=0),
        index=bow_vectorizer.get_feature_names_out()
    )
    print("\nTop 10 most frequent words:")
    print(word_freq.sort_values(ascending=False).head(10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each row of the matrix is a field. Each column is a word from the vocabulary. The value in position `(i, j)` is the number of times word `j` appears in field `i`'s observation log. Notice that `leaf`, `needed`, and `spreading` dominate — these are the words extension officers reach for most, across every issue type. They are frequent precisely because they are generic, which is the weakness TF-IDF will address in Challenge 11.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 10: N-Gram Analysis

    Single words (unigrams) lose context. The tag `'dark'` alone is ambiguous. But the bigram `'dark comedy'` is specific and meaningful — it describes a genre, not a mood. N-grams capture combinations of adjacent words and often carry far more discriminative power than individual tokens.

    ### Task
    Write a function `fit_ngram_vectorizer` that fits a **bigram** `CountVectorizer` on the tag corpus.

    **Function specifications:**
    - Takes `tag_corpus` (DataFrame with `'clean_tags'` column) and `max_features` (int, default `200`) as input.
    - Fits a `CountVectorizer` with `ngram_range=(2, 2)`, `max_features` set to the given value, and `stop_words='english'`.
    - Returns a tuple `(bigram_matrix, vectorizer)`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    Bigram matrix shape: (12000, 200)

    Top 10 most frequent bigrams:
    weeding overdue      703
    thistle border       703
    pigweed spreading    699
    midday wilt          699
    herbicide applied    684
    nutsedge problem     683
    wilting canopy       681
    choked row           679
    weed choked          679
    shade needed         675
    dtype: int64
    ```
    """)
    return


@app.cell
def _(CountVectorizer):
    ### START FUNCTION
    def fit_ngram_vectorizer(tag_corpus, max_features=200):
        # your code here
        vectorizer = CountVectorizer(
            ngram_range=(2, 2), 
            max_features=max_features,
            stop_words='english'
            )
        bigram_matrix = vectorizer.fit_transform(tag_corpus['clean_tags'])
        return bigram_matrix, vectorizer
    ### END FUNCTION
    return (fit_ngram_vectorizer,)


@app.cell
def _(fit_ngram_vectorizer, pd, tag_corpus):
    bigram_matrix, bigram_vectorizer = fit_ngram_vectorizer(tag_corpus, max_features=200)

    print(f"Bigram matrix shape: {bigram_matrix.shape}")
    bigram_freq = pd.Series(
        bigram_matrix.toarray().sum(axis=0),
        index=bigram_vectorizer.get_feature_names_out()
    )
    print("\nTop 10 most frequent bigrams:")
    print(bigram_freq.sort_values(ascending=False).head(10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare this list to the top unigrams above. Bigrams like `'based book'` (based on a book) and `'twist ending'` are clearly more informative than their component words `'based'` or `'ending'` in isolation. This is the core advantage of n-grams: they capture context that bag-of-words throws away.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 11: TF-IDF Pest Classifier

    Bag-of-words counts how often a word appears. **TF-IDF (Term Frequency–Inverse Document Frequency)** goes further: it down-weights words that appear in almost every field log (like `'leaf'`) and up-weights words that are distinctive to a small subset of fields (like `'borer'`). This makes TF-IDF generally more powerful than raw counts for classification tasks.

    ### Task
    We will train a **Logistic Regression** classifier to predict whether a field has a diagnosed Pest issue, using only its officers' observation tags as input.

    Write a function `train_tfidf_classifier` that builds this pipeline end to end.

    **Function specifications:**
    - Takes `tag_corpus` (DataFrame with `'fieldId'` and `'clean_tags'`), `fields` (DataFrame with `'fieldId'` and `'issues'`), and `max_features` (int, default `2000`) as input.
    - Merges `tag_corpus` with `fields` on `fieldId`.
    - Creates a binary target column `is_pest`: `1` if `'Pest'` is in the pipe-separated `issues` string, `0` otherwise.
    - Splits into 80/20 train/test using `train_test_split` with `random_state=42` and `stratify` on the target.
    - Fits a `TfidfVectorizer` with `max_features` on the training text only, transforms both sets.
    - Fits a `LogisticRegression(max_iter=1000, random_state=42)` on the training set.
    - Returns `(model, vectorizer, X_test_tfidf, y_test)`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    TF-IDF + Logistic Regression Accuracy: 0.9467

                  precision    recall  f1-score   support

         No Pest       0.93      1.00      0.96      1741
            Pest       1.00      0.81      0.89       659

        accuracy                           0.95      2400
       macro avg       0.97      0.90      0.93      2400
    weighted avg       0.95      0.95      0.94      2400

    Top 5 Pest-predictive words:    ['borer', 'swarm', 'infestation', 'hole', 'pesticide']
    Top 5 No-Pest-predictive words: ['runoff', 'pigweed', 'rot', 'yellowing', 'wilt']
    ```
    """)
    return


@app.cell
def _(LogisticRegression, TfidfVectorizer, pd, train_test_split):
    ### START FUNCTION
    def train_tfidf_classifier(tag_corpus, fields, max_features=2000):
        # your code here
        # Preprocess the text data
        merged_df = pd.merge(fields, tag_corpus, on='fieldId')

        merged_df['is_pest'] = merged_df['issues'].apply(lambda x: 1 if 'Pest' in str(x).split('|') else 0)
        X_train, X_test, y_train, y_test = train_test_split(
            merged_df['clean_tags'], 
            merged_df['is_pest'], 
            test_size=0.2, 
            random_state=42,
            stratify=merged_df['is_pest']
        )
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_tfidf, y_train)

        return model, vectorizer, X_test_tfidf, y_test
    ### END FUNCTION
    return (train_tfidf_classifier,)


@app.cell
def _(
    accuracy_score,
    classification_report,
    fields,
    tag_corpus,
    train_tfidf_classifier,
):
    tfidf_model, tfidf_vectorizer, X_test_tfidf, y_test = train_tfidf_classifier(tag_corpus, fields)

    y_pred = tfidf_model.predict(X_test_tfidf)
    print(f"TF-IDF + Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=['No Pest', 'Pest']))

    ### Most predictive words for Pest
    feature_names = tfidf_vectorizer.get_feature_names_out()
    coefs = tfidf_model.coef_[0]
    top5_pest    = [feature_names[i] for i in coefs.argsort()[-5:][::-1]]
    top5_no_pest = [feature_names[i] for i in coefs.argsort()[:5]]
    print(f"Top 5 Pest-predictive words:    {top5_pest}")
    print(f"Top 5 No-Pest-predictive words: {top5_no_pest}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model achieves ~95% accuracy — impressive for a pipeline that uses nothing but officer-logged observation tags. The most predictive Pest words (`borer`, `swarm`, `infestation`) make intuitive agronomic sense. On the other side, `runoff`, `pigweed`, and `yellowing` are strong signals that a field's problem is something *other* than pests — drainage, weeds, or nutrients. Note the recall of 0.81 on the Pest class: about one pest-affected field in five is missed, which is exactly the kind of honest limitation the Ministry needs to know before using this model for triage.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Exploring Further: Hugging Face Transformers

    The TF-IDF + Logistic Regression pipeline is a strong baseline — fast, interpretable, and respectable on this task. But it has a fundamental limitation: it treats every word independently, with no understanding of context or meaning. The word `'cold'` in `'cold snap'` and `'cold pest'` is treated identically.

    Modern **transformer models** like BERT and DistilBERT are pre-trained on billions of words of text. They encode not just the presence of words but their contextual relationships — `'cold'` in a pest context activates different internal representations than `'cold'` in a comedy context. This is why transformers dominate modern NLP benchmarks.

    We will not train a transformer in this project (it requires GPU resources and significantly more computation), but it is important to understand where TF-IDF sits in the broader landscape. The Hugging Face `transformers` library makes it straightforward to load pre-trained models for sentiment analysis, text classification, named entity recognition, and more — often in just a few lines of code.

    If you want to explore this direction, the `pipeline` API from Hugging Face is an excellent starting point:

    ```python
    from transformers import pipeline

    classifier = pipeline("sentiment-analysis")
    result = classifier("This field was terrifying and brilliant!")
    print(result)
    ```

    This is optional and ungraded — but if you are building a portfolio, showing awareness of both traditional and modern NLP approaches is a strong signal to employers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Wrapping up

    This project delivered two portfolio-ready pieces of work — both on the Ministry of Agriculture's national data platform.

    **Part A** tackled the weather stations. Five automated sensors had been logging temperature, rainfall, and air quality readings for over a year — but every reading was buried in raw text strings scattered across thirteen formats in two languages. Using regular expressions, we classified each message, extracted its timestamp, and pulled out the numeric reading. The result is a clean weather feature table: five rows, one per station, with mean temperature, rainfall, and pollution values ready to be joined back to the farm survey database.

    **Part B** answered the extension service's question: can officer-logged observation tags alone predict whether a field has a pest problem? We built a full NLP pipeline — preprocessing, tokenization, lemmatization, bag-of-words, n-gram analysis, and TF-IDF vectorization — and trained a logistic regression classifier that achieved ~95% accuracy. The most predictive Pest words (`borer`, `swarm`, `infestation`) confirm that the signal is real, not an artifact of class imbalance — and the 0.81 pest recall tells the Ministry exactly how much to trust it for triage.

    Both deliverables share a common thread: unstructured text in, structured insight out. That is the core promise of NLP, and this project demonstrates it end to end — in the language of the land.
    """)
    return


@app.cell
def cell_readiness():
    import inspect as _inspect
    import ast as _ast
    _spec = {'build_tag_corpus': {'expected_params': ['tags']}, 'build_weather_table': {'expected_params': ['weather_df']}, 'classify_message_type': {'expected_params': ['msg']}, 'extract_pollution': {'expected_params': ['msg']}, 'extract_rainfall': {'expected_params': ['msg']}, 'extract_temperature': {'expected_params': ['msg']}, 'extract_timestamp': {'expected_params': ['msg']}, 'fit_bow_vectorizer': {'expected_params': ['tag_corpus', 'max_features']}, 'fit_ngram_vectorizer': {'expected_params': ['tag_corpus', 'max_features']}, 'preprocess_text': {'expected_params': ['text', 'use_stemming']}, 'train_tfidf_classifier': {'expected_params': ['tag_corpus', 'fields', 'max_features']}}
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
