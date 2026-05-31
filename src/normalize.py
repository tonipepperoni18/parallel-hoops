import pandas as pd


LOWER_IS_BETTER = [
    "tov",
    "turnover_pct",
    "pf",
    "drtg",
]


def get_numeric_feature_columns(df, exclude_cols=None):
    """
    Gets numeric columns to normalize.
    """

    if exclude_cols is None:
        exclude_cols = []

    return [
        col for col in df.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def z_score_normalize(df, feature_cols):
    """
    Applies z-score normalization to selected feature columns.
    """

    df = df.copy()

    for col in feature_cols:
        if col in LOWER_IS_BETTER:
            df[col] = df[col] * -1

        mean = df[col].mean()
        std = df[col].std(ddof=0)

        if std == 0 or pd.isna(std):
            df[col] = 0
        else:
            df[col] = (df[col] - mean) / std

    return df


def normalize_by_league(df, league_col="league", exclude_cols=None):
    """
    Normalizes each league separately.

    Example:
    NBA players are normalized against NBA averages.
    WNBA players are normalized against WNBA averages.
    """

    if exclude_cols is None:
        exclude_cols = [
            "player",
            "tm",
            "pos",
            "player_id",
            "league",
            "season",
            "age",
            "g",
            "gs",
            "mp",
        ]

    if league_col not in df.columns:
        raise KeyError(f"Missing league column: {league_col}")

    normalized_parts = []

    for league, league_df in df.groupby(league_col):
        league_df = league_df.copy()

        feature_cols = get_numeric_feature_columns(
            league_df,
            exclude_cols=exclude_cols
        )

        league_normalized = z_score_normalize(
            league_df,
            feature_cols
        )

        normalized_parts.append(league_normalized)

    return pd.concat(normalized_parts, ignore_index=True)