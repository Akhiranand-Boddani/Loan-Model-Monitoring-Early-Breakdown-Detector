import os
import json
import pickle
from typing import Any


# ---------------------------------------------
# JSON SAVE / LOAD
# ---------------------------------------------

def save_json(obj: Any, filepath: str, indent: int = 4) -> None:
    """
    Save a Python object to disk as a JSON file.

    Args:
        obj (Any): The object to save (must be JSON serializable).
        filepath (str): The destination file path.
        indent (int, optional): JSON indent level.
    """
    parent_dir = os.path.dirname(filepath)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    with open(filepath, "w") as f:
        json.dump(obj, f, indent=indent)
    print(f"[persistence] Saved JSON → {filepath}")


def load_json(filepath: str) -> Any:
    """
    Load a JSON from disk.

    Args:
        filepath (str): Path to a JSON file.

    Returns:
        The Python object stored in JSON.
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


# ---------------------------------------------
# MODEL SAVE / LOAD
# ---------------------------------------------

def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model object (scikit/XGBoost) to disk via pickle.

    Args:
        model (Any): Trained model object.
        filepath (str): Where to store it.
    """
    parent_dir = os.path.dirname(filepath)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"[persistence] Saved model → {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load a serialized model from disk.

    Args:
        filepath (str): Path to .pkl file.

    Returns:
        The deserialized model.
    """
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model
