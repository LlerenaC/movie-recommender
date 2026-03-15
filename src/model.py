import numpy as np


class MatrixFactorizationSGD:
    """
    Matrix Factorization with biases, trained using SGD.

    Prediction:
        r_hat(u, i) = mu + b_u + b_i + U[u] · V[i]
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        k: int = 32,
        lr: float = 0.01,
        reg: float = 0.05,
        epochs: int = 20,
        init_mean: float = 0.0,
        init_std: float = 0.1,
        random_state: int = 42,
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.init_mean = init_mean
        self.init_std = init_std
        self.random_state = random_state

        self.mu = 0.0
        self.bu = np.zeros(n_users, dtype=np.float64)
        self.bi = np.zeros(n_items, dtype=np.float64)

        rng = np.random.default_rng(random_state)
        self.U = rng.normal(init_mean, init_std, size=(n_users, k))
        self.V = rng.normal(init_mean, init_std, size=(n_items, k))

        self.is_fitted = False

    def fit(self, train_data, val_data=None, shuffle: bool = True, verbose: bool = True):
        """
        train_data: np.ndarray of shape (n_samples, 3)
                    columns = [user_idx, item_idx, rating]
        val_data: same format, optional
        """
        train_data = np.asarray(train_data, dtype=np.float64)
        self.mu = float(np.mean(train_data[:, 2]))

        best_val_rmse = float("inf")
        best_params = None

        for epoch in range(1, self.epochs + 1):
            if shuffle:
                np.random.shuffle(train_data)

            for u, i, r in train_data:
                u = int(u)
                i = int(i)
                r = float(r)

                pred = self.mu + self.bu[u] + self.bi[i] + np.dot(self.U[u], self.V[i])
                err = r - pred

                # Save old user vector before updating it
                U_old = self.U[u].copy()

                # Bias updates
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                # Latent factor updates
                self.U[u] += self.lr * (err * self.V[i] - self.reg * self.U[u])
                self.V[i] += self.lr * (err * U_old - self.reg * self.V[i])

            train_rmse = self.rmse(train_data)

            if val_data is not None:
                val_rmse = self.rmse(val_data)

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_params = {
                        "bu": self.bu.copy(),
                        "bi": self.bi.copy(),
                        "U": self.U.copy(),
                        "V": self.V.copy(),
                    }

                if verbose:
                    print(
                        f"Epoch {epoch:02d} | "
                        f"train RMSE: {train_rmse:.4f} | "
                        f"val RMSE: {val_rmse:.4f}"
                    )
            else:
                if verbose:
                    print(f"Epoch {epoch:02d} | train RMSE: {train_rmse:.4f}")

        if val_data is not None and best_params is not None:
            self.bu = best_params["bu"]
            self.bi = best_params["bi"]
            self.U = best_params["U"]
            self.V = best_params["V"]

        self.is_fitted = True
        return self

    def predict_one(self, user_idx: int, item_idx: int) -> float:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict_one.")

        pred = (
            self.mu
            + self.bu[user_idx]
            + self.bi[item_idx]
            + np.dot(self.U[user_idx], self.V[item_idx])
        )
        return float(np.clip(pred, 1.0, 5.0))

    def predict(self, data):
        """
        Predict ratings for an array with columns [user_idx, item_idx, ...]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict.")

        data = np.asarray(data)
        users = data[:, 0].astype(int)
        items = data[:, 1].astype(int)

        preds = (
            self.mu
            + self.bu[users]
            + self.bi[items]
            + np.sum(self.U[users] * self.V[items], axis=1)
        )
        preds = np.clip(preds, 1.0, 5.0)
        return preds

    def rmse(self, data) -> float:
        data = np.asarray(data)
        users = data[:, 0].astype(int)
        items = data[:, 1].astype(int)
        targets = data[:, 2].astype(np.float64)

        preds = (
            self.mu
            + self.bu[users]
            + self.bi[items]
            + np.sum(self.U[users] * self.V[items], axis=1)
        )
        preds = np.clip(preds, 1.0, 5.0)

        return float(np.sqrt(np.mean((targets - preds) ** 2)))