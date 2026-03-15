import numpy as np


def get_top_k_recommendations(model, user_id, k=10, exclude_items=None):
    n_items = model.V.shape[0]

    scores = np.array([model.predict(user_id, item_id) for item_id in range(n_items)])

    if exclude_items is not None:
        scores[list(exclude_items)] = -np.inf

    top_k_items = np.argsort(scores)[-k:][::-1]
    return top_k_items