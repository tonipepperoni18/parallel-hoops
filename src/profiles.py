import pandas as pd


def check_merge_keys(df, df_name, merge_keys):
    """
    Checks that required merge columns exist.
    """

    missing = [col for col in merge_keys if col not in df.columns]

    if missing:
        raise KeyError(
            f"{df_name} is missing merge columns: {missing}\n\n"
            f"Available columns:\n{df.columns.tolist()}"
        )


def drop_suffixed_duplicate_columns(df):
    """
    Drops common duplicate columns created during merges.
    """

    drop_patterns = [
        "_perposs",
        "_shooting",
        "_pbp"
    ]

    keep_if_contains = [
        "pct",
        "per_100",
        "dist",
        "dunk",
        "corner",
        "astd"
    ]

    cols_to_drop = []

    for col in df.columns:
        for pattern in drop_patterns:
            if col.endswith(pattern):
                base_name = col.replace(pattern, "")

                # drop boring duplicates like age_perposs, pos_perposs, rank_perposs
                if base_name in ["rank", "age", "pos", "g", "gs", "mp"]:
                    cols_to_drop.append(col)

    return df.drop(columns=cols_to_drop)


def build_player_profiles(advanced, perposs, shooting, playbyplay):
    """
    Merge cleaned stat tables into one player profile dataset.
    """

    merge_keys = ["player", "tm", "g", "mp"]

    check_merge_keys(advanced, "advanced", merge_keys)
    check_merge_keys(perposs, "perposs", merge_keys)
    check_merge_keys(shooting, "shooting", merge_keys)
    check_merge_keys(playbyplay, "playbyplay", merge_keys)

    profiles = advanced.merge(
        perposs,
        on=merge_keys,
        how="left",
        suffixes=("", "_perposs")
    )

    profiles = profiles.merge(
        shooting,
        on=merge_keys,
        how="left",
        suffixes=("", "_shooting")
    )

    profiles = profiles.merge(
        playbyplay,
        on=merge_keys,
        how="left",
        suffixes=("", "_pbp")
    )

    profiles = drop_suffixed_duplicate_columns(profiles)

    return profiles.reset_index(drop=True)