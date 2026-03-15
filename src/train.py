import numpy as np

class BaseLineBiasModel:
    """
    Baseline recommender:
        r_hat(u, i) = mu + b_u + b_i
        
    Trqained with SGD and L2 regularization.
    """