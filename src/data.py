import json
from pathlib import Path
from typing import Tuple

import pandas as pd


# ----------------------------
# Paths
# ----------------------------
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw" / "ml-latest-small"  # change if needed
PROCESSED_DIR = DATA_DIR / "processed"

RATINGS_FILENAME = "ratings.csv"   # change if needed
MOVIES_FILENAME = "movies.csv"     # optional, for later recommendation display

# Directory helpers
# ----------------------------
def ensure_dirs() -> None:
    """Create processed directory if it does not exist."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Loading raw data
# ----------------------------
def load_raw_ratings(filename: str = RATINGS_FILENAME) -> pd.DataFrame:
    """
    Load raw MovieLens ratings data.

    Expected columns: userId, movieId, rating, timestamp
    
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Ratings file not found at {path}. Please ensure the file exists.")
    ratings = pd.read_csv(path)
    return ratings

def load_raw_movies(filename: str = MOVIES_FILENAME) -> pd.DataFrame:
    """
    optinal helper for later when we want movie titles.
    
    Expected columns: movieId, title, genres
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Movies file not found at {path}. Please ensure the file exists.")
    return pd.read_csv(path)

# ----------------------------
# ID mapping
# ----------------------------
def create_id_mappings(ratings: pd.DataFrame) -> Tuple[dict, dict]:
    """
    Create mappings:
        raw userId -> mapped user_idx
        raw movieId -> mapped movie_idx
    """
    unique_user_ids = sorted(ratings["userId"].unique().tolist())
    unique_movie_ids = sorted(ratings["movieId"].unique().tolist())

    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}

    return user_to_idx, movie_to_idx

def apply_id_mappings(ratings: pd.DataFrame, user_to_idx: dict, movie_to_idx: dict) -> pd.DataFrame:
    """ 
    Apply the user and movie ID mappings to the ratings DataFrame.
    """
    ratings["user_idx"] = ratings["userId"].map(user_to_idx)
    ratings["movie_idx"] = ratings["movieId"].map(movie_to_idx)
    return ratings

# ----------------------------
# Splitting
# ----------------------------
def split_user_by_time(ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each user, sort their ratings by timestamp and split into:
        - train: all but last 2 ratings
        - val: second to last rating
        - test: last rating
    """
    train_list = []
    val_list = []
    test_list = []

    for _, user_df in ratings.groupby("user_idx"):
        user_df_sorted = user_df.sort_values("timestamp")
        n = len(user_df_sorted)

        if n < 3:
            train_list.append(user_df_sorted)
        else:
            train_list.append(user_df_sorted.iloc[:-2])
            val_list.append(user_df_sorted.iloc[-2: -1])
            test_list.append(user_df_sorted.iloc[-1:])
    
    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, val_df, test_df

# ----------------------------
# Saving
# ----------------------------
def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Save the train, val, and test DataFrames to CSV files in the processed directory.
    """
    ensure_dirs()
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

def save_id_mappings(user_to_idx: dict, movie_to_idx: dict) -> None:
    """
    Save the user and movie ID mappings to JSON files.
    JSON requires string keys so we sovert keysd to str.
    """
    ensure_dirs()

    user_to_index_json = {str(k): v for k, v in user_to_idx.items()}
    item_to_index_json = {str(k): v for k, v in movie_to_idx.items()}

    index_to_user_json = {str(v): k for k, v in user_to_idx.items()}
    index_to_item_json = {str(v): k for k, v in movie_to_idx.items()}

    with open(PROCESSED_DIR / "user_to_index.json", "w", encoding="utf-8") as f:
        json.dump(user_to_index_json, f, indent=2)

    with open(PROCESSED_DIR / "item_to_index.json", "w", encoding="utf-8") as f:
        json.dump(item_to_index_json, f, indent=2)

    with open(PROCESSED_DIR / "index_to_user.json", "w", encoding="utf-8") as f:
        json.dump(index_to_user_json, f, indent=2)

    with open(PROCESSED_DIR / "index_to_item.json", "w", encoding="utf-8") as f:
        json.dump(index_to_item_json, f, indent=2)

# ----------------------------
# Loading processed artifacts
# ----------------------------
def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    load the train, val, and test DataFrames from the processed directory.
    """
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    return train_df, val_df, test_df

def load_id_mappings() -> Tuple[dict, dict, dict, dict]:
    """
    load saved mappings.
    returns:
        user_to_idx, movie_to_idx, idx_to_user, idx_to_movie
    """
    with open(PROCESSED_DIR / "user_to_index.json", "r", encoding="utf-8") as f:
        user_to_index_raw = json.load(f)

    with open(PROCESSED_DIR / "item_to_index.json", "r", encoding="utf-8") as f:
        item_to_index_raw = json.load(f)

    with open(PROCESSED_DIR / "index_to_user.json", "r", encoding="utf-8") as f:
        index_to_user_raw = json.load(f)

    with open(PROCESSED_DIR / "index_to_item.json", "r", encoding="utf-8") as f:
        index_to_item_raw = json.load(f)

    user_to_index = {int(k): int(v) for k, v in user_to_index_raw.items()}
    item_to_index = {int(k): int(v) for k, v in item_to_index_raw.items()}
    index_to_user = {int(k): int(v) for k, v in index_to_user_raw.items()}
    index_to_item = {int(k): int(v) for k, v in index_to_item_raw.items()}

    return user_to_index, item_to_index, index_to_user, index_to_item

# ----------------------------
# Summary helper
# ----------------------------
def print_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    print("Split summary:")
    print(f"  Train ratings: {len(train_df)}")
    print(f"  Val ratings:   {len(val_df)}")
    print(f"  Test ratings:  {len(test_df)}")
    print(f"  Train users:   {train_df['user_idx'].nunique()}")
    print(f"  Train items:   {train_df['movie_idx'].nunique()}")

# ----------------------------
# Main pipeline
# ----------------------------
def create_and_save_splits():
    """
    Full ingestion pipeline:
    1. Load raw ratings
    2. Create ID mappings
    3. Apply ID mappings to ratings
    4. Split into train/val/test
    5. Save splits and mappings
    6. Print summary"""
    ratings = load_raw_ratings()
    user_to_idx, movie_to_idx = create_id_mappings(ratings)
    ratings_mapped = apply_id_mappings(ratings, user_to_idx, movie_to_idx)
    train_df, val_df, test_df = split_user_by_time(ratings_mapped)
    save_splits(train_df, val_df, test_df)
    save_id_mappings(user_to_idx, movie_to_idx)
    print_split_summary(train_df, val_df, test_df)

def load_movies(path: str) -> pd.DataFrame:
    """Load movies CSV and set movieId as index."""
    return pd.read_csv(path).set_index('movieId')

def load_train_data(path: str) -> pd.DataFrame:
    """Load train data CSV."""
    return pd.read_csv(path)

def get_user_rated_items(train_df: pd.DataFrame, user_id: int) -> set:
    """Get the set of movie_idx that the user has rated in train_df."""
    return set(train_df[train_df['user_idx'] == user_id]['movie_idx'])

if __name__ == "__main__":
    print("Creating data splits...")
    create_and_save_splits()
    print("Done!")





