import pandas as pd
import os
import numpy as np

from baseline import BaseLineBiasModel
from model import MatrixFactorizationSGD


def main():

    # -------------------------
    # Load processed data
    # -------------------------
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    train_data = train_df[["user_idx", "movie_idx", "rating"]].to_numpy()
    val_data = val_df[["user_idx", "movie_idx", "rating"]].to_numpy()
    test_data = test_df[["user_idx", "movie_idx", "rating"]].to_numpy()

    n_users = int(train_df["user_idx"].max()) + 1
    n_items = int(train_df["movie_idx"].max()) + 1

    print("\nUsers:", n_users)
    print("Items:", n_items)
    print("Train samples:", len(train_data))

    # -------------------------
    # BASELINE MODEL
    # -------------------------
    print("\nTraining baseline model...")

    baseline = BaseLineBiasModel(
        n_users=n_users,
        n_items=n_items,
        lr=0.01,
        reg=0.02,
        epochs=20,
    )

    baseline.fit(train_data, val_data=val_data)

    print("\nBaseline Results")
    print("----------------")
    print("Train RMSE:", baseline.rmse(train_data))
    print("Val RMSE:", baseline.rmse(val_data))
    print("Test RMSE:", baseline.rmse(test_data))

    # -------------------------
    # MATRIX FACTORIZATION
    # -------------------------
    print("\nTraining matrix factorization model...")

    mf = MatrixFactorizationSGD(
        n_users=n_users,
        n_items=n_items,
        k=32,
        lr=0.01,
        reg=0.05,
        epochs=30,
    )

    mf.fit(train_data, val_data=val_data)

    print("\nMatrix Factorization Results")
    print("----------------------------")
    print("Train RMSE:", mf.rmse(train_data))
    print("Val RMSE:", mf.rmse(val_data))
    print("Test RMSE:", mf.rmse(test_data))



    os.makedirs("outputs/models", exist_ok=True)

    np.save("outputs/models/U.npy", mf.U)
    np.save("outputs/models/V.npy", mf.V)


if __name__ == "__main__":
    main()