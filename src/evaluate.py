"""Evaluate organ-chip quality models with acquisition-date holdouts."""

from __future__ import annotations

import json
import hashlib
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .shift import fit_shift_screen


ROOT = Path(__file__).resolve().parents[1]
IMAGE_FEATURES = ROOT / "data/handcrafted_features.csv"
SHEET = ROOT / "data/raw/OOC_datasheet.xlsx"
MANIFEST = ROOT / "data/image_manifest.csv"
EMBEDDINGS = ROOT / "data/mobilenet_embeddings.npz"
OUTPUT = ROOT / "reports"
MODEL_DIR = ROOT / "models"


def load_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    features = pd.read_csv(IMAGE_FEATURES)
    selection = ROOT / "data/selection.csv"
    if selection.exists():
        selected = set(pd.read_csv(selection).imageID)
        features = features[features.imageID.isin(selected)]
        if set(features.imageID) != selected:
            raise ValueError("Selected images are missing features; finish download and extraction first")
    sheet = pd.read_excel(SHEET)
    manifest = pd.read_csv(MANIFEST)
    data = (features.merge(sheet, on="imageID", validate="one_to_one")
            .merge(manifest, on="imageID", validate="one_to_one"))
    assert len(data) == len(features)
    assert ((data["Decision 1/2 (good/bad)"] == 1) ==
            (data.quality_label == "good")).all()
    data["good"] = (data.quality_label == "good").astype(int)
    data["acquisition_date"] = data.imageID.str.split("_").str[0]
    data["cell_type"] = data["cell type"].astype(str)
    data["seeding_density"] = pd.to_numeric(
        data["seeding density, cells/ml"].astype(str).str.replace(",", "", regex=False),
        errors="coerce")
    data["flow_ul_min"] = pd.to_numeric(
        data["flow rate"].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
    image_columns = [column for column in features if column != "imageID"]
    embedding_columns = []
    if EMBEDDINGS.exists():
        cache = np.load(EMBEDDINGS)
        embedded_ids = cache["ids"].astype(str)
        if set(data.imageID).issubset(set(embedded_ids)):
            matrix = cache["features"].astype(np.float32)
            index = pd.Index(embedded_ids).get_indexer(data.imageID)
            assert (index >= 0).all()
            embedding_columns = [f"mobilenet_{index:04d}" for index in range(matrix.shape[1])]
            embedding_frame = pd.DataFrame(matrix[index], columns=embedding_columns,
                                           index=data.index)
            data = pd.concat([data, embedding_frame], axis=1)
        else:
            print("MobileNet cache incomplete; evaluating handcrafted features only", flush=True)
    return data, image_columns, embedding_columns


def make_model(name: str, image_columns: list[str],
               embedding_columns: list[str]) -> tuple[Pipeline, list[str]]:
    metadata_numeric = ["day", "time after seeding, h", "seeding_density", "flow_ul_min"]
    if name == "metadata_logistic":
        numerical = metadata_numeric
        classifier = LogisticRegression(max_iter=1000, C=1.0)
    elif name == "image_logistic":
        numerical = image_columns
        classifier = LogisticRegression(max_iter=2000, C=0.1)
    elif name == "image_boosted":
        numerical = image_columns
        classifier = HistGradientBoostingClassifier(
            max_iter=180, max_leaf_nodes=12, learning_rate=0.045,
            min_samples_leaf=20, l2_regularization=1.0, random_state=42)
    elif name == "fusion_boosted":
        numerical = image_columns + metadata_numeric
        classifier = HistGradientBoostingClassifier(
            max_iter=180, max_leaf_nodes=12, learning_rate=0.045,
            min_samples_leaf=20, l2_regularization=1.0, random_state=42)
    elif name == "mobilenet_logistic":
        numerical = embedding_columns
        classifier = LogisticRegression(max_iter=2000, C=0.1)
    else:
        raise ValueError(name)
    categorical = ["cell_type"] if name in {"metadata_logistic", "fusion_boosted"} else []
    transformer = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                               ("scale", StandardScaler())]), numerical),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ], sparse_threshold=0)
    columns = numerical + categorical
    stages = [("transform", transformer)]
    if name == "mobilenet_logistic":
        stages.append(("pca", PCA(n_components=64, svd_solver="randomized", random_state=42)))
    stages.append(("classify", classifier))
    return Pipeline(stages), columns


def score(y: np.ndarray, probability: np.ndarray) -> dict:
    prediction = (probability >= 0.5).astype(int)
    both_labels = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "good_share": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, probability)) if both_labels else None,
        "average_precision": float(average_precision_score(y, probability)) if both_labels else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "brier": float(brier_score_loss(y, probability)),
        "confusion_bad_good": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }


def evaluate_split(data: pd.DataFrame, image_columns: list[str],
                   embedding_columns: list[str], kind: str) -> tuple[dict, dict]:
    y = data.good.to_numpy()
    groups = data.acquisition_date.to_numpy()
    if kind == "date_grouped":
        splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(data, y, groups)
    else:
        splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(data, y)
    split_list = list(splits)
    results = {}
    predictions = {}
    names = ["metadata_logistic", "image_logistic", "image_boosted", "fusion_boosted"]
    if embedding_columns:
        names.append("mobilenet_logistic")
    for name in names:
        model, columns = make_model(name, image_columns, embedding_columns)
        probability = np.full(len(data), np.nan)
        fold_metrics = []
        for fold, (train, valid) in enumerate(split_list):
            if kind == "date_grouped":
                assert not set(groups[train]).intersection(groups[valid])
            model.fit(data.iloc[train][columns], y[train])
            probability[valid] = model.predict_proba(data.iloc[valid][columns])[:, 1]
            fold_metrics.append({"fold": fold, "date_groups": int(len(set(groups[valid]))),
                                 **score(y[valid], probability[valid])})
        assert np.isfinite(probability).all()
        results[name] = {"pooled": score(y, probability), "folds": fold_metrics}
        predictions[name] = probability
        print(kind, name, results[name]["pooled"], flush=True)
    return results, predictions


def date_bootstrap_auc(y: np.ndarray, probability: np.ndarray,
                       dates: np.ndarray, repetitions: int = 500,
                       subtract: np.ndarray | None = None) -> list[float]:
    """Resample acquisition dates, preserving images within each sampled date."""
    rng = np.random.default_rng(42)
    unique_dates = np.unique(dates)
    indices = {date: np.flatnonzero(dates == date) for date in unique_dates}
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(unique_dates, len(unique_dates), replace=True)
        selection = np.concatenate([indices[date] for date in sampled])
        if len(np.unique(y[selection])) == 2:
            value = roc_auc_score(y[selection], probability[selection])
            if subtract is not None:
                value -= roc_auc_score(y[selection], subtract[selection])
            values.append(value)
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def cell_type_holdouts(data: pd.DataFrame, name: str, image_columns: list[str],
                       embedding_columns: list[str]) -> dict:
    results = {}
    for cell_type in sorted(data.cell_type.unique()):
        train = data.cell_type != cell_type
        valid = ~train
        model, columns = make_model(name, image_columns, embedding_columns)
        model.fit(data.loc[train, columns], data.loc[train, "good"])
        probability = model.predict_proba(data.loc[valid, columns])[:, 1]
        results[cell_type] = score(data.loc[valid, "good"].to_numpy(), probability)
    return results


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    data, image_columns, embedding_columns = load_data()
    print(f"Images={len(data)}; acquisition dates={data.acquisition_date.nunique()}; "
          f"good share={data.good.mean():.3f}", flush=True)
    if len(data) < 100:
        raise ValueError("Download at least 100 images before evaluation")
    random_results, random_predictions = evaluate_split(data, image_columns,
                                                        embedding_columns, "image_random")
    grouped_results, grouped_predictions = evaluate_split(data, image_columns,
                                                           embedding_columns, "date_grouped")
    screen = fit_shift_screen(data, image_columns)
    image_candidates = [name for name in grouped_results if name != "metadata_logistic"]
    best_name = min(image_candidates, key=lambda name: (
        -grouped_results[name]["pooled"]["roc_auc"],
        grouped_results[name]["pooled"]["brier"]))
    y = data.good.to_numpy()
    dates = data.acquisition_date.to_numpy()
    selected_probability = grouped_predictions[best_name]
    selected_cell_types = {}
    for cell_type, frame in data.groupby("cell_type"):
        selected_cell_types[cell_type] = score(frame.good.to_numpy(),
                                              selected_probability[frame.index])
    confidence = ((selected_probability <= 0.2) | (selected_probability >= 0.8))
    document = {
        "dataset": {"source": "https://doi.org/10.5281/zenodo.10203721",
                    "n_images": len(data), "acquisition_dates": int(data.acquisition_date.nunique()),
                    "group_proxy": "first six characters of imageID (acquisition date)",
                    "good_images": int(y.sum()), "bad_images": int(len(y) - y.sum()),
                    "image_size_counts": {
                        f"{int(width)}x{int(height)}": int(count) for (width, height), count in
                        data.groupby(["image_width", "image_height"]).size().items()},
                    "metadata_missing_counts": {
                        column: int(data[column].isna().sum()) for column in
                        ("seeding_density", "flow_ul_min", "time after seeding, h")}},
        "reproduction": {
            "python": platform.python_version(),
            "package_versions": {package: version(package) for package in
                                 ("scikit-learn", "numpy", "pandas", "torch", "torchvision")},
            "seed": 42,
            "image_ids_sha256": hashlib.sha256(
                "\n".join(sorted(data.imageID)).encode()).hexdigest(),
        },
        "image_random": random_results,
        "date_grouped": grouped_results,
        "validation_gap": {
            name: {
                "random_minus_grouped_auc": random_results[name]["pooled"]["roc_auc"] -
                                            grouped_results[name]["pooled"]["roc_auc"],
                "paired_date_bootstrap_95pct_interval": date_bootstrap_auc(
                    y, random_predictions[name], dates, subtract=grouped_predictions[name]),
            } for name in grouped_results
        },
        "selected_model": best_name,
        "selected_model_diagnostics": {
            "date_bootstrap_95pct_auc_interval": date_bootstrap_auc(
                y, selected_probability, dates),
            "random_split_date_bootstrap_95pct_auc_interval": date_bootstrap_auc(
                y, random_predictions[best_name], dates),
            "per_cell_type_date_grouped": selected_cell_types,
            "illustrative_20_80_confident_fraction": float(confidence.mean()),
            "illustrative_20_80_confident_balanced_accuracy": float(
                balanced_accuracy_score(y[confidence],
                                        (selected_probability[confidence] >= 0.5).astype(int)))
                if confidence.any() else None,
        },
        "selected_model_cell_type_holdouts": cell_type_holdouts(
            data, best_name, image_columns, embedding_columns),
        "appearance_shift_screen": {
            "method": "12-PC distance to nearest image from a different acquisition date",
            "training_other_date_distance_p95": screen["threshold"],
            "note": "Heuristic novelty flag; not a validated safety detector",
        },
    }
    (OUTPUT / "evaluation.json").write_text(json.dumps(document, indent=2, allow_nan=False))
    prediction_table = data[["imageID", "good", "acquisition_date", "cell_type", "source_split"]].copy()
    for name, values in grouped_predictions.items():
        prediction_table[name + "_prob_good"] = values
    prediction_table[best_name + "_random_oof_prob_good"] = random_predictions[best_name]
    prediction_table.to_csv(OUTPUT / "grouped_oof_predictions.csv", index=False)
    model, columns = make_model(best_name, image_columns, embedding_columns)
    model.fit(data[columns], data.good)
    joblib.dump({"model": model, "columns": columns, "name": best_name,
                 "image_columns": image_columns, "embedding_columns": embedding_columns,
                 "shift_screen": screen, "n_training_images": len(data),
                 "role": "full_data_model", "training_image_ids": data.imageID.tolist()},
                MODEL_DIR / "chipqc.joblib")
    train, valid = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                        .split(data, y, dates))
    demo_model, columns = make_model(best_name, image_columns, embedding_columns)
    demo_model.fit(data.iloc[train][columns], y[train])
    demo_probability = demo_model.predict_proba(data.iloc[valid][columns])[:, 1]
    np.testing.assert_allclose(demo_probability, selected_probability[valid], atol=1e-7)
    joblib.dump({"model": demo_model, "columns": columns, "name": best_name,
                 "image_columns": image_columns, "embedding_columns": embedding_columns,
                 "shift_screen": fit_shift_screen(data.iloc[train], image_columns),
                 "n_training_images": len(train), "role": "date_holdout_demo",
                 "training_image_ids": data.iloc[train].imageID.tolist()},
                MODEL_DIR / "chipqc-demo-heldout.joblib")
    demo_manifest = data.iloc[valid][["imageID", "good", "acquisition_date", "cell_type"]].copy()
    demo_manifest["prob_good"] = demo_probability
    demo_manifest.to_csv(OUTPUT / "demo_holdout_manifest.csv", index=False)
    print("Selected", best_name, "by grouped ROC AUC", flush=True)


if __name__ == "__main__":
    main()
