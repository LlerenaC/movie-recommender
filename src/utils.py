import random
import numpy as np
import os


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_rng(seed: int = 42):
    return np.random.default_rng(seed)


def save_model(path: str, **kwargs):
    """
    Save model parameters (U, V, biases, etc.)
    """
    np.savez(path, **kwargs)


def load_model(path: str):
    """
    Load model parameters.
    """
    return np.load(path)