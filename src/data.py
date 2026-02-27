import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def load_ratings():
    path = "../data/raw/ml-latest-small/ratings.csv"
    ratings = pd.read_csv(path)
    return ratings

def load_movies():
    path = "../data/raw/ml-latest-small/movies.csv"
    movies = pd.read_csv(path)
    return movies


def load_and_preprocess(ratings_dir = "../data/raw/ml-latest-small/ratings.csv", movie_dir = "../data/raw/ml-latest-small/movies.csv"):
    # Load in ratings and movies
    ratings = pd.read_csv(ratings_dir)
    movies = pd.read_csv(movie_dir)
    
    # Remap ids for users and movies from 0 -> n-1
    user_ids = ratings["userId"].unique()
    map_userId = {oldId: newId for newId, oldId in enumerate(user_ids)}

    movie_ids = movies["movieId"].unique()
    map_movieId = {oldId: newId for newId, oldId in enumerate(movie_ids)}

    ratings["userId"] = ratings["userId"].map(map_userId)
    ratings["movieId"] = ratings["movieId"].map(map_movieId)
    movies["movieId"] = movies["movieId"].map(map_movieId)

    # Get count of users and movies for matrices
    n_users = len(map_userId)
    n_movies = len(map_movieId)

    train_list = []
    val_list = []
    test_list = []

    for user_id, user_df in ratings.groupby("userId"):
        n = len(user_df)

        if n < 3:
            train_list.append(user_df)
        else:
            train_list.append(user_df.iloc[:-2])
            val_list.append(user_df.iloc[-2])
            test_list.append(user_df.iloc[-1])
    
    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, val_df, test_df, n_users, n_movies, map_userId, map_movieId

def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def save_dfs(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    ensure_dirs()

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

def load_splits():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    return train, val, test






