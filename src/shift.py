"""Image-appearance novelty screen based on other acquisition dates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_shift_screen(data: pd.DataFrame, image_columns: list[str]) -> dict:
    projection = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=min(12, len(image_columns), len(data) - 1),
                    random_state=42)),
    ])
    points = projection.fit_transform(data[image_columns]).astype(np.float32)
    date = data.acquisition_date.to_numpy()
    distances = pairwise_distances(points)
    distances[date[:, None] == date[None, :]] = np.inf
    nearest_other_date = distances.min(axis=1)
    if not np.isfinite(nearest_other_date).all():
        raise ValueError("At least two acquisition dates are needed")
    threshold = float(np.quantile(nearest_other_date, 0.95))
    return {"projection": projection, "points": points,
            "threshold": threshold, "training_distances": nearest_other_date}


def shift_distance(screen: dict, image_row: pd.DataFrame,
                   image_columns: list[str]) -> float:
    point = screen["projection"].transform(image_row[image_columns]).astype(np.float32)
    return float(np.linalg.norm(screen["points"] - point, axis=1).min())
