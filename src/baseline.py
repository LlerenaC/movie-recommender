import numpy as np
import pandas as pd

class BaseLineBiasModel:
    """
    Baseline recommender:
        r_hat(u, i) = mu + b_u + b_i
        
    Trqained with SGD and L2 regularization.
    """

    def __init__(self, n_users: int, n_items: int, lr: float = 0.01, reg: float = 0.02, epochs: int = 20):
        self.n_users = n_users
        self.n_items = n_items
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

        # Initialize parameters
        self.mu = 0.0
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)

        self.is_fitted = False

    def fit(self, train_data, val_data=None, shuffle: bool = True, verbose: bool = True):
        """
        train_data: np.array of shape (n_samples, 3) with columns [user_idx, item_idx, rating]
        """
        train_data = np.array(train_data)
        self.mu = float(np.mean(train_data[:, 2]))
        
        best_val_rmse = float("inf")
        best_bu = None
        best_bi = None

        for epoch in range(1, self.epochs+1):
            if shuffle:
                np.random.shuffle(train_data)

            for u, i, r in train_data:
                u, i = int(u), int(i)
                r = float(r)

                pred = self.mu + self.b_u[u] + self.b_i[i]
                err = r - pred

                # Update biases
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])
            
            train_rmse = self.rmse(train_data)

            if val_data is not None:
                val_rmse = self.rmse(val_data)
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_bu = self.b_u.copy()
                    best_bi = self.b_i.copy()

                if verbose:
                    print(f"Epoch {epoch:02d} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

            else:
                if verbose:
                    print(f"Epoch {epoch:02d} | Train RMSE: {train_rmse:.4f}")

        if val_data is not None and best_bu is not None:
            self.b_u = best_bu
            self.b_i = best_bi

        self.is_fitted = True
        return self
    
    def predict_one(self, user_idx, item_idx):
        """
        Predict a single rating
        """

        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        
        pred = self.mu + self.b_u[user_idx] + self.b_i[item_idx]

        return pred
    
    def predict(self, data):
        """
        Predict ratings for an arrayt of [user_idx, item_idx, rating] or [user_idx, item_idx].
        returns numpy array of predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        
        data = np.array(data)
        users = data[:, 0].astype(int)
        items = data[:, 1].astype(int)

        preds = self.mu + self.b_u[users] + self.b_i[items]
        preds = np.clip(preds, 1.0, 5.0)  # Clip to valid rating range
        return preds
    
    def rmse(self, data):
        """
        Compute RMSE on given data
        """
        data = np.array(data)
        preds = self.mu + self.b_u[data[:, 0].astype(int)] + self.b_i[data[:, 1].astype(int)]
        preds = np.clip(preds, 1.0, 5.0)
        targets = data[:, 2].astype(float)
        rmse = np.sqrt(np.mean((targets - preds) ** 2))
        return rmse
