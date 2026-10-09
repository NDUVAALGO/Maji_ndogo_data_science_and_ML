# The Language of the Land: Regex and NLP for Maji Ndogo

A two-part text-processing project for the fictional Maji Ndogo Ministry of Agriculture. **Part A** uses regular expressions to turn raw, bilingual weather-station messages into a clean feature table. **Part B** builds an NLP pipeline that predicts whether a field has a **pest problem** from the free-text observation tags that field officers log. The work is a [marimo](https://marimo.io) notebook.

## The problems

- **Weather stations (Part A):** Five stations log temperature, rainfall and air-quality readings as raw text strings, in English and Chinese, in thirteen different message formats. None of it is usable by a model until the numbers are extracted.
- **Field observations (Part B):** Field officers log short tags such as "chewed leaf" or "weeding overdue" in an app. Can those tags alone tell the Ministry which fields have a diagnosed pest issue, so agronomists can be sent to the right fields first?

## Data

| File | Contents | Size |
|---|---|---|
| `Weather_station_data.csv` | Raw weather messages with a station ID | 1,843 messages |
| `Weather_data_field_mapping.csv` | Mapping between weather stations and fields | 5,654 rows |
| `field_issues.csv` | Fields with a pipe-separated list of diagnosed issues (for example `Pest\|Drought`) | 15,000 fields |
| `field_tags.csv` | Officer observation tags per field | 119,173 tags |

## Part A: Regex extraction

Five functions, each built on regular expressions:

1. `classify_message_type`: labels each message as temperature, rainfall or pollution, in English or Chinese.
2. `extract_timestamp`: pulls out the `YYYY-MM-DD HH:MM:SS` timestamp wherever it sits in the message.
3. `extract_temperature`: extracts the Celsius reading as a float.
4. `extract_rainfall`: extracts the rainfall in mm as a float.
5. `extract_pollution`: extracts the pollution index as a float, including negative values.

`build_weather_table` then applies these to every message and averages the readings per station, giving one row per station:

| Station | Temperature | Rainfall | Pollution |
|---|---|---|---|
| 0 | 13.4039 | 1575.9538 | 0.3528 |
| 1 | 12.9989 | 577.3839 | 0.2573 |
| 2 | 13.1797 | 1690.9553 | 0.0439 |
| 3 | 13.2568 | 905.1914 | 0.2382 |
| 4 | 13.1886 | 1200.1835 | 0.1283 |

The messages split into 625 rainfall, 611 temperature and 607 pollution readings, with none left unclassified.

## Part B: NLP pest classifier

1. **Text preprocessing:** lowercase, strip non-letters, tokenise with NLTK, remove stopwords and single-character tokens, then lemmatise (or optionally stem).
2. **Tag corpus:** combine all of a field's tags into one cleaned document, giving 12,000 fields with tags.
3. **Bag-of-words:** a 500-term `CountVectorizer` matrix.
4. **N-gram analysis:** a 200-term bigram matrix, which surfaces phrases like "weeding overdue" and "wilting canopy".
5. **TF-IDF classifier:** a 2,000-term `TfidfVectorizer` fitted on the training set only, feeding a logistic regression with an 80/20 stratified split. The target `is_pest` is 1 when "Pest" appears in the field's issues.

### Results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Pest | 0.93 | 1.00 | 0.96 | 1,741 |
| Pest | 1.00 | 0.81 | 0.89 | 659 |

Overall test accuracy is **0.9467** on 2,400 fields.

The most pest-predictive words are `borer`, `swarm`, `infestation`, `hole` and `pesticide`. The strongest signals against a pest diagnosis are `runoff`, `pigweed`, `rot`, `yellowing` and `wilt`, which point to drainage, weed or nutrient problems.

**Takeaways**
- The classifier never wrongly flags a field as a pest field (Pest precision 1.00), but it misses about 19% of real pest cases (recall 0.81). For triage, that recall is the number to improve.
- The predictive words match agronomic intuition, which suggests the model learned meaningful patterns rather than noise.
- Transformer models such as BERT are a natural next step, since TF-IDF treats every word independently of its context.

## Tech stack

Python, marimo, pandas, NumPy, Matplotlib, NLTK, scikit-learn, `re` (regular expressions).

## Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install marimo pandas numpy matplotlib nltk scikit-learn
marimo edit notebook.py
```

Put the four CSV files from the table above in the same folder as `notebook.py`. The first run downloads the NLTK resources (punkt, stopwords, wordnet), so it needs an internet connection.

## Project structure

```
.
├── notebook.py                        # marimo notebook, Parts A and B
├── Weather_station_data.csv
├── Weather_data_field_mapping.csv
├── field_issues.csv
├── field_tags.csv
└── README.md
```

## Next steps

- Join the parsed weather table to the farm survey through the field mapping, to enrich crop and yield models with station data.
- Improve pest recall, for example by tuning the decision threshold or using class weights.
- Try a pretrained transformer for the tag classification.

## Credits

The challenge and data come from the Explore AI data science programme.