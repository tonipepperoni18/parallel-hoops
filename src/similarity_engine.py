import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


CATEGORY_WEIGHTS = {
    "scoring": 0.175,
    "playmaking": 0.175,
    "individual_defense": 0.175,
    "team_defense": 0.175,
    "impact": 0.20,
    "position": 0.10,
}


FEATURE_GROUPS = {
    "scoring": [
        "pts",
        "ts_pct",
        "efg_pct",
        "fg_pct",
        "2p_pct",
        "3p_pct",
        "ft_pct",
        "three_point_attempt_rate",
        "free_throw_rate",
        "dist",
        "0_3",
        "3_10",
        "10_16",
        "16_3p",
    ],

    "playmaking": [
        "ast",
        "assist_pct",
        "tov",
        "turnover_pct",
        "obpm",
        "ows",
        "ortg",
        "usage_pct",
    ],

    "individual_defense": [
        "stl",
        "blk",
        "steal_pct",
        "block_pct",
        "pf",
    ],

    "team_defense": [
        "drtg",
        "dbpm",
        "dws",
        "drb",
        "def_reb_pct",
    ],

    "impact": [
        "per",
        "ws",
        "ws_per_48",
        "bpm",
        "vorp",
        "on_off",
    ],

    "position": [
        "pg_pct",
        "sg_pct",
        "sf_pct",
        "pf_pct",
        "c_pct",
    ],
}


LOWER_IS_BETTER = [
    "tov",
    "turnover_pct",
    "pf",
    "drtg",
]


def get_available_feature_groups(df):
    """
    Returns available features by category.
    """

    available = {}

    for category, features in FEATURE_GROUPS.items():
        available[category] = [
            feature for feature in features
            if feature in df.columns
        ]

    return available


def build_weighted_feature_matrix(df):
    """
    Builds a normalized and weighted matrix for similarity scoring.
    """

    df = df.copy()

    available_groups = get_available_feature_groups(df)
    weighted_groups = []

    for category, features in available_groups.items():

        if not features:
            continue

        category_df = df[features].copy()

        for col in category_df.columns:
            category_df[col] = pd.to_numeric(category_df[col], errors="coerce")

            if col in LOWER_IS_BETTER:
                category_df[col] = category_df[col] * -1

        category_df = category_df.fillna(category_df.mean())

        std = category_df.std(ddof=0).replace(0, 1)
        z_scores = (category_df - category_df.mean()) / std

        weight = CATEGORY_WEIGHTS[category]
        z_scores = z_scores * weight

        weighted_groups.append(z_scores)

    if not weighted_groups:
        raise ValueError("No usable similarity features found in dataframe.")

    matrix = pd.concat(weighted_groups, axis=1)

    return matrix


def find_player_index(df, player_name):
    """
    Finds the first player matching the search name.
    """

    matches = df[
        df["player"].str.lower().str.contains(player_name.lower(), na=False)
    ]

    if matches.empty:
        raise ValueError(f"No player found matching: {player_name}")

    return matches.index[0]


def find_similar_players(player_name, df, top_n=10):
    """
    Finds the most similar players and includes category similarity scores.
    """

    df = df.copy().reset_index(drop=True)

    target_index = find_player_index(df, player_name)

    matrix = build_weighted_feature_matrix(df)

    overall_scores = cosine_similarity(
        matrix.iloc[[target_index]],
        matrix
    )[0]

    category_scores = calculate_category_similarities(
        player_name,
        df
    )

    category_scores["overall_similarity_score"] = overall_scores

    results = category_scores.drop(index=target_index)

    results = results.sort_values(
        by="overall_similarity_score",
        ascending=False
    )

    return results.head(top_n).reset_index(drop=True)


def calculate_category_similarities(player_name, df):
    """
    Calculates similarity scores by category for one player.
    """

    df = df.copy().reset_index(drop=True)

    target_index = find_player_index(df, player_name)

    results = df[["player", "tm", "pos", "age", "g", "mp"]].copy()

    for category, features in FEATURE_GROUPS.items():
        available_features = [
            feature for feature in features
            if feature in df.columns
        ]

        if not available_features:
            continue

        category_matrix = df[available_features].copy()

        for col in category_matrix.columns:
            category_matrix[col] = pd.to_numeric(
                category_matrix[col],
                errors="coerce"
            )

        category_matrix = category_matrix.fillna(
            category_matrix.mean()
        )

        scores = cosine_similarity(
            category_matrix.iloc[[target_index]],
            category_matrix
        )[0]

        results[f"{category}_similarity"] = scores

    return results


def similarity_feature_report(df):
    """
    Shows which features are being used by each category.
    """

    available = get_available_feature_groups(df)

    report = {}

    for category, features in FEATURE_GROUPS.items():
        report[category] = {
            "weight": CATEGORY_WEIGHTS[category],
            "available": available[category],
            "missing": [
                feature for feature in features
                if feature not in df.columns
            ],
        }

    return report


