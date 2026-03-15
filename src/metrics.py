import numpy as np

def recommend_top_k(U, V, user_id, seen_items, k=10):
    scores = U[user_id] @ V.T        # predict score for every item

    scores[list(seen_items)] = -np.inf   # remove already seen items

    top_k = np.argsort(scores)[-k:][::-1]

    return top_k