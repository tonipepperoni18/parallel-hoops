import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean


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


def find_similar_players(
    player_name,
    df,
    top_n=10,
    same_position=False,
    same_league=False
):
    """
    Finds the most similar players and includes category similarity scores.

    Optional filters:
    - same_position=True only returns players with the same listed position
    - same_league=True only returns players from the same league
    """

    df = df.copy().reset_index(drop=True)

    target_index = find_player_index(df, player_name)
    target_player = df.iloc[target_index]

    target_pos = target_player["pos"] if "pos" in df.columns else None
    target_league = target_player["league"] if "league" in df.columns else None

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

    if same_position and target_pos is not None:
        results = results[
            results["pos"] == target_pos
        ]

    if same_league and target_league is not None:
        results = results[
            results["league"] == target_league
        ]

    results = results.sort_values(
        by="overall_similarity_score",
        ascending=False
    )

    return results.head(top_n).reset_index(drop=True)


def find_league_comps(player_name, df, target_league="NBA", top_n=10):
    results = find_similar_players(
        player_name,
        df,
        top_n=len(df)
    )

    return (
        results[results["league"] == target_league]
        .head(top_n)
        .reset_index(drop=True)
    )
def calculate_category_similarities(player_name, df):
    """Calculates category similarity scores using Euclidean distance.

    This avoids inflated cosine scores in small feature groups.
    """ 
    df = df.copy().reset_index(drop=True)
    
    target_index = find_player_index(df, player_name)
    
    result_cols = [
        col for col in ["player", "tm", "pos", "league", "season", "age", "g", "mp"]
        if col in df.columns
    ]
    
    results = df[result_cols].copy()
    
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
            category_matrix.mean())
        
        target_vector = category_matrix.iloc[target_index]
        
        scores = []
        
        for i in range(len(category_matrix)):
            comparison_vector = category_matrix.iloc[i]
            
            distance = euclidean(
                target_vector, 
                comparison_vector
            )
            
            similarity = 1 / (1 + distance)
            
            scores.append(similarity)
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


