import pandas as pd

def clean_profiles(df):

    df = df.copy()

    columns_to_drop = [
        "rank_perposs",
        "age_perposs",
        "pos_perposs",

        "rank_shooting",
        "age_shooting",
        "pos_shooting",

        "rank_pbp",
        "age_pbp",
        "pos_pbp",

        "awards_perposs",
        "awards_shooting",
        "awards_pbp",

        "player_id_perposs",
        "player_id_shooting",
        "player_id_pbp",
    ]

    existing_cols = [
        col
        for col in columns_to_drop
        if col in df.columns
    ]

    rename_map = {
    "3par": "three_point_attempt_rate",
    "ftr": "free_throw_rate",

    "orb_pct": "off_reb_pct",
    "drb_pct": "def_reb_pct",
    "trb_pct": "total_reb_pct",

    "ast_pct": "assist_pct",
    "stl_pct": "steal_pct",
    "blk_pct": "block_pct",
    "tov_pct": "turnover_pct",

    "usg_pct": "usage_pct",
    }
    df = df.drop(columns=existing_cols)
    df = df.rename(columns=rename_map)
    return df