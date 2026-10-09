"""
Reference implementation — Shamba Ndogo Recommendation Engine

For side-by-side comparison against your own attempt. Signatures match
the notebook's readiness-check spec exactly.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


# ── Challenge 1 ──────────────────────────────────────────────────────────
def build_content_table(varieties, agronomy):
    content_df = varieties.merge(agronomy, on='varietyId', how='left')

    content_df['year'] = pd.to_numeric(
        content_df['variety'].str[-6:].str[1:5], errors='coerce'
    ).fillna(0).astype(int)

    content_df['traits'] = content_df['traits'].replace('(no traits listed)', '')

    for col in ['trial_agronomists', 'breeder', 'trait_keywords']:
        content_df[col] = content_df[col].fillna('')

    return content_df


# ── Challenge 2 ──────────────────────────────────────────────────────────
def aggregate_tags(tags, min_count=5):
    tags = tags.copy()
    tags['tag'] = tags['tag'].str.lower()

    tag_counts = tags['tag'].value_counts()
    valid_tags = tag_counts[tag_counts >= min_count].index
    filtered = tags[tags['tag'].isin(valid_tags)]

    agg = (
        filtered.groupby('varietyId')['tag']
        .apply(lambda s: ' '.join(s))
        .reset_index()
        .rename(columns={'tag': 'agg_tags'})
    )
    return agg


# ── Challenge 3 ──────────────────────────────────────────────────────────
def build_tfidf_matrix(content_df, tag_agg):
    content_df = content_df.merge(tag_agg, on='varietyId', how='left')
    content_df['agg_tags'] = content_df['agg_tags'].fillna('')

    def make_soup(row):
        traits_part = row['traits'].replace('|', ' ')
        breeder_part = row['breeder'].replace(' ', '')
        agronomists = str(row['trial_agronomists']).split('|')[:5]
        agronomists_part = ' '.join(a.replace(' ', '') for a in agronomists)
        keywords_part = row['trait_keywords'].replace('|', ' ')
        tags_part = row['agg_tags']
        return ' '.join([traits_part, breeder_part, agronomists_part,
                          keywords_part, tags_part])

    content_df['soup'] = content_df.apply(make_soup, axis=1)

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(content_df['soup'])

    return tfidf_matrix, vectorizer, content_df


# ── Challenge 4 ──────────────────────────────────────────────────────────
def get_content_recommendations(variety_name, content_df, tfidf_matrix, top_n=10):
    matches = content_df[
        content_df['variety'].str.contains(variety_name, case=False, na=False, regex=False)
    ]
    if matches.empty:
        return pd.DataFrame(columns=['variety', 'traits', 'breeder', 'similarity_score'])

    idx = matches.index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    ranked = sim_scores.argsort()[::-1]
    ranked = ranked[ranked != idx][:top_n]

    result = content_df.loc[ranked, ['variety', 'traits', 'breeder']].copy()
    result['similarity_score'] = sim_scores[ranked]
    result = result.sort_values('similarity_score', ascending=False).reset_index(drop=True)
    return result


# ── Challenge 5 ──────────────────────────────────────────────────────────
def trait_overlap_score(query_title, recommendations, content_df):
    if recommendations is None or recommendations.empty:
        return 0.0

    matches = content_df[
        content_df['variety'].str.contains(query_title, case=False, na=False, regex=False)
    ]
    if matches.empty:
        return 0.0

    query_traits = set(t for t in matches.iloc[0]['traits'].split('|') if t)

    scores = []
    for traits_str in recommendations['traits']:
        rec_traits = set(t for t in str(traits_str).split('|') if t)
        union = query_traits | rec_traits
        intersection = query_traits & rec_traits
        scores.append(len(intersection) / len(union) if union else 0.0)

    return float(np.mean(scores))


# ── Challenge 6 ──────────────────────────────────────────────────────────
def build_utility_matrix(ratings, n_farmers=200, n_varieties=100):
    top_farmers = ratings['farmerId'].value_counts().head(n_farmers).index
    subset = ratings[ratings['farmerId'].isin(top_farmers)]

    top_varieties = subset['varietyId'].value_counts().head(n_varieties).index
    filtered = subset[subset['varietyId'].isin(top_varieties)]

    utility_matrix = filtered.pivot_table(
        index='farmerId', columns='varietyId', values='rating'
    )
    return utility_matrix


# ── Challenge 7 ──────────────────────────────────────────────────────────
def compute_user_similarity(utility_matrix):
    row_means = utility_matrix.mean(axis=1)
    centered = utility_matrix.sub(row_means, axis=0).fillna(0)

    sim_matrix = cosine_similarity(centered)
    return pd.DataFrame(sim_matrix, index=utility_matrix.index, columns=utility_matrix.index)


# ── Challenge 8 ──────────────────────────────────────────────────────────
def predict_rating(farmer_id, variety_id, utility_matrix, user_sim, top_k=10):
    if farmer_id not in utility_matrix.index:
        return 3.0

    user_ratings = utility_matrix.loc[farmer_id]
    if user_ratings.dropna().empty:
        return 3.0

    user_mean = user_ratings.mean()

    if variety_id not in utility_matrix.columns:
        return float(np.clip(user_mean, 0.5, 5.0))

    raters = utility_matrix.index[utility_matrix[variety_id].notna()]
    raters = raters[raters != farmer_id]

    if len(raters) == 0:
        return float(np.clip(user_mean, 0.5, 5.0))

    sims = user_sim.loc[farmer_id, raters]
    top_raters = sims.abs().sort_values(ascending=False).head(top_k).index

    numerator = 0.0
    denominator = 0.0
    for v in top_raters:
        sim_uv = user_sim.loc[farmer_id, v]
        rating_vm = utility_matrix.loc[v, variety_id]
        mean_v = utility_matrix.loc[v].mean()
        numerator += sim_uv * (rating_vm - mean_v)
        denominator += abs(sim_uv)

    if denominator == 0:
        return float(np.clip(user_mean, 0.5, 5.0))

    prediction = user_mean + numerator / denominator
    return float(np.clip(prediction, 0.5, 5.0))


# ── Challenge 9 ──────────────────────────────────────────────────────────
def evaluate_collaborative_rmse(utility_matrix, user_sim, ratings,
                                 test_fraction=0.2, random_state=42):
    rng = np.random.default_rng(random_state)

    actuals, predictions = [], []

    for farmer_id in utility_matrix.index:
        rated = utility_matrix.loc[farmer_id].dropna().index.to_numpy()
        if len(rated) < 2:
            continue

        n_test = max(1, int(len(rated) * test_fraction))
        test_varieties = rng.choice(rated, size=n_test, replace=False)

        for variety_id in test_varieties:
            actuals.append(utility_matrix.loc[farmer_id, variety_id])
            predictions.append(
                predict_rating(farmer_id, variety_id, utility_matrix, user_sim)
            )

    return float(np.sqrt(mean_squared_error(actuals, predictions)))


# ── Challenge 10 ─────────────────────────────────────────────────────────
def cold_start_comparison(content_df, tfidf_matrix, utility_matrix, new_variety_name, top_n=5):
    matches = content_df[
        content_df['variety'].str.contains(new_variety_name, case=False, na=False, regex=False)
    ]

    if matches.empty:
        collab_possible = False
    else:
        variety_id = matches.iloc[0]['varietyId']
        collab_possible = bool(
            variety_id in utility_matrix.columns
            and utility_matrix[variety_id].notna().any()
        )

    content_recs = get_content_recommendations(new_variety_name, content_df, tfidf_matrix, top_n=top_n)

    return {'content_recs': content_recs, 'collab_possible': collab_possible}
