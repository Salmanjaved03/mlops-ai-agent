"""Data preparation script for the ML pipeline."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def prepare_data():
    """Load iris dataset and save to processed directory."""
    logger.info("Preparing data...")
    
    # Ensure directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # Save raw data
    np.save(RAW_DIR / "features.npy", data.data)
    np.save(RAW_DIR / "labels.npy", data.target)
    logger.info(f"Raw data saved to {RAW_DIR}")

    # Save processed splits
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)

    # Save metadata
    metadata = {
        "feature_names": list(data.feature_names),
        "target_names": list(data.target_names),
        "n_samples": len(data.data),
        "n_features": len(data.feature_names),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "split_ratio": 0.2,
        "random_state": 42,
    }
    with open(PROCESSED_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Processed data saved to {PROCESSED_DIR}")
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return metadata


if __name__ == "__main__":
    prepare_data()
