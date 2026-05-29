import pandas as pd


def fix_bref_multi_header(df):
    """
    Fix Basketball Reference tables where the real column names
    are stored in the first row of data.
    """

    df = df.copy()

    first_row = df.iloc[0].astype(str).tolist()

    if "Player" in first_row:
        df.columns = first_row
        df = df.iloc[1:].reset_index(drop=True)

    return df


def clean_column_names(df):
    """
    Standardize column names.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("▲", "", regex=False)
        .str.replace("%", "_pct", regex=False)
        .str.replace("/", "_per_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace("'", "", regex=False)
    )

    rename_map = {
        "rk": "rank",
        "team": "tm",
        "player_additional": "player_id",
        "#name?": "player_id",
    }

    df = df.rename(columns=rename_map)

    # important: fix duplicate names after renaming/cleaning
    df.columns = make_unique_columns(df.columns)

    return df

def remove_repeated_headers(df):
    """
    Remove repeated header rows inside the table.
    """

    df = df.copy()

    if "player" in df.columns:
        df = df[df["player"].astype(str).str.lower() != "player"]

    return df


def convert_numeric_columns(df):
    """
    Convert columns to numeric when possible.
    """

    df = df.copy()

    text_cols = {"player", "tm", "pos", "awards", "player_id"}

    for col in df.columns:
        if col not in text_cols:
            converted = pd.to_numeric(df[col], errors="coerce")

            if converted.notna().sum() > 0:
                df[col] = converted

    return df


def clean_stat_table(df):
    """
    Main cleaner for Basketball Reference tables.
    """

    df = df.copy()

    df = fix_bref_multi_header(df)
    df = clean_column_names(df)
    df = remove_repeated_headers(df)

    df = df.loc[:, ~df.columns.str.contains("^unnamed", case=False)]

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    df = convert_numeric_columns(df)

    return df.reset_index(drop=True)


def make_unique_columns(columns):
    """
    Makes duplicate column names unique.
    Example:
    fg_pct, fg_pct -> fg_pct, fg_pct_2
    """

    seen = {}
    new_columns = []

    for col in columns:
        if col not in seen:
            seen[col] = 1
            new_columns.append(col)
        else:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")

    return new_columns