# shamba-ndogo-recommender

**A hybrid recommendation engine for crop varieties in Maji Ndogo: content-based filtering with TF-IDF, user-based collaborative filtering, and a cold-start comparison.**

Farmers in Maji Ndogo have hundreds of crop varieties to choose from. This project recommends varieties in two complementary ways: by what a variety *is* (its traits, breeder, agronomy notes, and farmer tags), and by what similar farmers *rated highly*. It then shows where each approach breaks down, most importantly for brand-new varieties that nobody has rated yet.

## What's inside

The engine is built as ten small, testable functions in `notebook.py`.

### Part 1: Content-based filtering

| # | Function | What it does |
|---|----------|--------------|
| 1 | `build_content_table` | Merges variety and agronomy data, extracts the release year from the variety name, and cleans placeholder trait values |
| 2 | `aggregate_tags` | Lowercases farmer tags, drops rare ones (`min_count`), and combines the rest into one string per variety |
| 3 | `build_tfidf_matrix` | Builds a text "soup" per variety (traits, breeder, agronomists, keywords, tags) and vectorizes it with TF-IDF |
| 4 | `get_content_recommendations` | Returns the top-N varieties most similar to a given one using cosine similarity |
| 5 | `trait_overlap_score` | Evaluates recommendations with the Jaccard overlap of trait sets between the query and each result |

### Part 2: Collaborative filtering

| # | Function | What it does |
|---|----------|--------------|
| 6 | `build_utility_matrix` | Builds a farmer × variety rating matrix from the most active farmers and most rated varieties |
| 7 | `compute_user_similarity` | Mean-centers each farmer's ratings, then computes cosine similarity between farmers |
| 8 | `predict_rating` | Predicts a rating from the top-k most similar farmers, with fallbacks for unknown farmers or varieties |
| 9 | `evaluate_collaborative_rmse` | Holds out a fraction of known ratings and reports RMSE |

### Part 3: Putting them together

| # | Function | What it does |
|---|----------|--------------|
| 10 | `cold_start_comparison` | For a new variety, shows that content-based filtering still gives recommendations while collaborative filtering cannot |

## Key ideas

- **TF-IDF over "soup" text.** Traits, breeder, agronomists, and tags are combined into one document per variety, so rare, distinctive terms count for more than common ones.
- **Mean-centering.** Some farmers rate generously and others harshly. Subtracting each farmer's average rating makes similarity reflect *taste*, not rating style.
- **The cold-start problem.** Collaborative filtering needs ratings. A new variety has none, so only the content-based approach can recommend it.

## Requirements

- Python 3.9+
- `pandas`
- `numpy`
- `scikit-learn`

```bash
pip install pandas numpy scikit-learn
```

## Usage

```python
from notebook import (
    build_content_table, aggregate_tags, build_tfidf_matrix,
    get_content_recommendations, build_utility_matrix,
    compute_user_similarity, predict_rating,
)

# varieties, agronomy, tags, ratings are pandas DataFrames
content_df = build_content_table(varieties, agronomy)
tag_agg = aggregate_tags(tags, min_count=5)
tfidf_matrix, vectorizer, content_df = build_tfidf_matrix(content_df, tag_agg)

# Similar varieties
get_content_recommendations("Some Variety Name", content_df, tfidf_matrix, top_n=10)

# Predicted rating for a farmer
utility = build_utility_matrix(ratings)
user_sim = compute_user_similarity(utility)
predict_rating(farmer_id, variety_id, utility, user_sim)
```

## Part of the Maji Ndogo series

This project is one of several data science notebooks set in Maji Ndogo.