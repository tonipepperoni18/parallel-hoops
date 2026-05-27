## File NOTES
"""
loader.py
========================================

Purpose
-------
This file is responsible for automatically loading CSV datasets
for the Parallel Hoops project.

The goal is to centralize all file-loading logic into one place
instead of repeating pd.read_csv() calls throughout notebooks.

This creates:
- cleaner notebooks
- reusable loading logic
- easier debugging
- scalable project structure
- easier transition into production pipelines later


This loader acts as the first stage of the data pipeline.


Core Ideas
----------
1. pathlib
   Used for platform-safe file paths.
   Better than manually typing strings.

2. glob
   Automatically searches folders for matching files.
   This allows automatic loading of every CSV.

3. Dictionaries
   DataFrames are stored inside dictionaries:
   
   Example:
   data["nba"]["25-26"]["advanced"]

"""






from pathlib import Path
import pandas as pd
import glob


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "DATA"


def load_csv(path):
    """
    Safely load a CSV file.
    """
    try:
        df = pd.read_csv(path)

        print(f"Loaded: {Path(path).name}")
        print(f"Shape: {df.shape}")

        return df

    except pd.errors.ParserError as e:
        print(f"\nParser issue in: {path}")
        print(e)
        return None

    except Exception as e:
        print(f"\nError loading: {path}")
        print(e)
        return None


def load_folder(folder_path):
    """
    Automatically load every CSV in a folder.
    Returns dictionary of DataFrames.
    """

    csv_files = glob.glob(str(folder_path / "*.csv"))

    data = {}

    for file in csv_files:

        file_name = Path(file).stem

        df = load_csv(file)

        if df is not None:
            data[file_name] = df

    return data


def load_league_season(league, season):
    """
    Example:
    load_league_season("nba", "25-26")
    """

    folder = DATA_DIR / "raw" / league / season

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    return load_folder(folder)


def load_all_leagues():
    """
    Automatically load everything inside DATA/raw.
    """

    all_data = {}

    raw_path = DATA_DIR / "raw"

    leagues = raw_path.iterdir()

    for league in leagues:

        if league.is_dir():

            league_name = league.name

            all_data[league_name] = {}

            seasons = league.iterdir()

            for season in seasons:

                if season.is_dir():

                    season_name = season.name

                    print(f"\nLoading {league_name} {season_name}")

                    all_data[league_name][season_name] = load_folder(
                        season
                    )

    return all_data