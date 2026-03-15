# scripts/run_demo.py

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def find_column(df: pd.DataFrame, candidates: list[str], name: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find a {name} column. Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def load_train_data(train_path: Path) -> tuple[pd.DataFrame, str, str, str]:
    train_df = pd.read_csv(train_path)

    user_col = find_column(train_df, ["user_id", "user_idx", "user", "u"], "user")
    item_col = find_column(train_df, ["movie_id", "movie_idx", "movie", "m"], "movie")
    rating_col = find_column(train_df, ["rating", "r"], "rating")

    return train_df, user_col, item_col, rating_col


def load_movie_titles(movies_path: Path, num_items: int) -> dict[int, str]:
    movies_df = pd.read_csv(movies_path)

    title_col = find_column(movies_df, ["title", "movie_title", "name"], "title")

    # Best case: movie file already uses remapped item indices
    possible_item_cols = ["movie_id", "movie_idx", "movie", "m"]
    item_col = None
    for col in possible_item_cols:
        if col in movies_df.columns:
            item_col = col
            break

    if item_col is not None:
        idx_to_title = {
            int(row[item_col]): str(row[title_col])
            for _, row in movies_df.iterrows()
        }
    else:
        # Fallback: assume rows are already aligned with model item indices 0..num_items-1
        if len(movies_df) < num_items:
            raise ValueError(
                "movies.csv does not have an item index column, and it has fewer rows "
                "than num_items, so I cannot safely map item indices to titles."
            )
        idx_to_title = {
            idx: str(movies_df.iloc[idx][title_col])
            for idx in range(num_items)
        }

    return idx_to_title


def pick_demo_user(
    train_df: pd.DataFrame,
    user_col: str,
    min_ratings: int,
    requested_user: int | None,
) -> int:
    if requested_user is not None:
        if requested_user not in set(train_df[user_col].unique()):
            raise ValueError(f"User {requested_user} not found in training data.")
        return requested_user

    counts = train_df.groupby(user_col).size()
    valid_users = counts[counts >= min_ratings].index.tolist()

    if not valid_users:
        valid_users = counts.index.tolist()

    if not valid_users:
        raise ValueError("No users found in training data.")

    return int(np.random.choice(valid_users))


def get_seen_items(
    train_df: pd.DataFrame,
    user_col: str,
    item_col: str,
    user_id: int,
) -> set[int]:
    user_rows = train_df[train_df[user_col] == user_id]
    return set(user_rows[item_col].astype(int).tolist())


def get_user_top_rated(
    train_df: pd.DataFrame,
    user_col: str,
    item_col: str,
    rating_col: str,
    user_id: int,
    idx_to_title: dict[int, str],
    n: int = 5,
) -> list[tuple[str, float]]:
    user_rows = train_df[train_df[user_col] == user_id].copy()
    user_rows = user_rows.sort_values(rating_col, ascending=False).head(n)

    results: list[tuple[str, float]] = []
    for _, row in user_rows.iterrows():
        item_id = int(row[item_col])
        rating = float(row[rating_col])
        title = idx_to_title.get(item_id, f"<unknown item {item_id}>")
        results.append((title, rating))

    return results


def recommend_top_k(
    U: np.ndarray,
    V: np.ndarray,
    user_id: int,
    seen_items: set[int],
    idx_to_title: dict[int, str],
    k: int = 10,
) -> list[tuple[int, str, float]]:
    scores = U[user_id] @ V.T

    if seen_items:
        scores[list(seen_items)] = -np.inf

    top_k_items = np.argsort(scores)[-k:][::-1]

    results: list[tuple[int, str, float]] = []
    for item_id in top_k_items:
        title = idx_to_title.get(int(item_id), f"<unknown item {item_id}>")
        score = float(scores[item_id])
        results.append((int(item_id), title, score))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a demo for matrix factorization recommendations.")
    parser.add_argument("--u-path", type=str, default="outputs/models/U.npy", help="Path to U.npy")
    parser.add_argument("--v-path", type=str, default="outputs/models/V.npy", help="Path to V.npy")
    parser.add_argument("--train-path", type=str, default="data/processed/train.csv", help="Path to train.csv")
    parser.add_argument("--movies-path", type=str, default="data/raw/ml-latest-small/movies.csv", help="Path to movies.csv")
    parser.add_argument("--user", type=int, default=None, help="User id to demo. If omitted, a valid user is chosen randomly.")
    parser.add_argument("--k", type=int, default=10, help="Number of recommendations to show.")
    parser.add_argument("--history-n", type=int, default=5, help="Number of past liked movies to show.")
    parser.add_argument("--min-ratings", type=int, default=20, help="Minimum ratings for randomly selected demo user.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for user selection.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    u_path = Path(args.u_path)
    v_path = Path(args.v_path)
    train_path = Path(args.train_path)
    movies_path = Path(args.movies_path)

    if not u_path.exists():
        raise FileNotFoundError(f"Could not find U file: {u_path}")
    if not v_path.exists():
        raise FileNotFoundError(f"Could not find V file: {v_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Could not find train file: {train_path}")
    if not movies_path.exists():
        raise FileNotFoundError(f"Could not find movies file: {movies_path}")

    U = np.load(u_path)
    V = np.load(v_path)

    if U.ndim != 2 or V.ndim != 2:
        raise ValueError(f"Expected U and V to be 2D arrays. Got shapes {U.shape} and {V.shape}.")
    if U.shape[1] != V.shape[1]:
        raise ValueError(
            f"Latent dimension mismatch: U has shape {U.shape}, V has shape {V.shape}."
        )

    num_users, k_dim = U.shape
    num_items = V.shape[0]

    train_df, user_col, item_col, rating_col = load_train_data(train_path)
    idx_to_title = load_movie_titles(movies_path, num_items=num_items)

    user_id = pick_demo_user(
        train_df=train_df,
        user_col=user_col,
        min_ratings=args.min_ratings,
        requested_user=args.user,
    )

    if user_id < 0 or user_id >= num_users:
        raise ValueError(
            f"Chosen user_id={user_id} is outside U's range [0, {num_users - 1}]. "
            "This usually means your train.csv user ids do not match the remapped indices used in U."
        )

    seen_items = get_seen_items(train_df, user_col, item_col, user_id)

    top_rated = get_user_top_rated(
        train_df=train_df,
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
        user_id=user_id,
        idx_to_title=idx_to_title,
        n=args.history_n,
    )

    recommendations = recommend_top_k(
        U=U,
        V=V,
        user_id=user_id,
        seen_items=seen_items,
        idx_to_title=idx_to_title,
        k=args.k,
    )

    print("=" * 60)
    print("MATRIX FACTORIZATION RECOMMENDER DEMO")
    print("=" * 60)
    print(f"User ID: {user_id}")
    print(f"Latent dimension: {k_dim}")
    print(f"Seen items in training: {len(seen_items)}")

    print("\nSome movies this user rated highly:")
    if top_rated:
        for idx, (title, rating) in enumerate(top_rated, start=1):
            print(f"{idx:>2}. {title} (rating: {rating:.1f})")
    else:
        print("No prior ratings found for this user.")

    print(f"\nTop {args.k} recommendations:")
    for rank, (item_id, title, score) in enumerate(recommendations, start=1):
        print(f"{rank:>2}. {title} [item {item_id}] (predicted score: {score:.3f})")

    print("\nDone.")


if __name__ == "__main__":
    main()