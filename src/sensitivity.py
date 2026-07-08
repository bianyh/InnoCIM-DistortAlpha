from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LayerSensitivity:
    layer: str
    score: float
    weight: float
    layer_index: int | None = None
    max_accuracy_drop: float | None = None
    mean_accuracy_drop: float | None = None
    max_relative_l2: float | None = None
    max_cosine_drift: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


CIFAR_RESNET20_DEFAULT_LAYERS = [
    "layer3.0.conv1",
    "layer1.0.conv1",
    "layer2.0.conv1",
    "layer2.0.conv2",
    "layer3.1.conv1",
    "layer3.2.conv2",
    "layer1.2.conv1",
    "layer3.0.conv2",
]

VOXFORGE_CRNN_DEFAULT_LAYERS = [
    "cnn.0",
    "cnn.3",
    "classifier",
]


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    std = float(values.std(ddof=0))
    if std < 1e-12:
        return values * 0.0
    return (values - float(values.mean())) / std


def _softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    temperature = max(float(temperature), 1e-6)
    scaled = [v / temperature for v in values]
    peak = max(scaled)
    exps = [math.exp(v - peak) for v in scaled]
    total = sum(exps)
    return [v / max(total, 1e-12) for v in exps]


def default_sensitivity(task: str, model_name: str, top_k: int | None = None) -> list[LayerSensitivity]:
    if task == "cifar10" and model_name == "cifar10_resnet20":
        layers = CIFAR_RESNET20_DEFAULT_LAYERS
    elif task == "voxforge" and model_name == "crnn_ctc":
        layers = VOXFORGE_CRNN_DEFAULT_LAYERS
    else:
        layers = []
    if top_k is not None:
        layers = layers[:top_k]
    if not layers:
        return []
    weight = 1.0 / len(layers)
    return [
        LayerSensitivity(layer=layer, score=1.0, weight=weight, layer_index=idx)
        for idx, layer in enumerate(layers)
    ]


def load_cifar_resnet20_sensitivity(
    sensitivity_csv: str | Path,
    drift_csv: str | Path | None = None,
    top_k: int = 8,
    temperature: float = 0.5,
) -> list[LayerSensitivity]:
    sensitivity_csv = Path(sensitivity_csv)
    if not sensitivity_csv.exists():
        return default_sensitivity("cifar10", "cifar10_resnet20", top_k=top_k)

    sensitivity_df = pd.read_csv(sensitivity_csv)
    if sensitivity_df.empty or "layer" not in sensitivity_df.columns:
        return default_sensitivity("cifar10", "cifar10_resnet20", top_k=top_k)

    grouped = sensitivity_df.groupby("layer", as_index=False).agg(
        layer_index=("layer_index", "min"),
        max_accuracy_drop=("accuracy_drop", "max"),
        mean_accuracy_drop=("accuracy_drop", "mean"),
    )

    if drift_csv is not None and Path(drift_csv).exists():
        drift_df = pd.read_csv(drift_csv)
        if not drift_df.empty and "layer" in drift_df.columns:
            drift_grouped = drift_df.groupby("layer", as_index=False).agg(
                max_relative_l2=("relative_l2", "max"),
                max_cosine_drift=("cosine_drift", "max"),
            )
            grouped = grouped.merge(drift_grouped, on="layer", how="left")

    for column in ["max_relative_l2", "max_cosine_drift"]:
        if column not in grouped.columns:
            grouped[column] = 0.0
        grouped[column] = pd.to_numeric(grouped[column], errors="coerce").fillna(0.0)

    grouped["score"] = (
        _zscore(grouped["max_accuracy_drop"])
        + 0.5 * _zscore(grouped["mean_accuracy_drop"])
        + 0.5 * _zscore(grouped["max_relative_l2"].map(lambda v: math.log1p(max(float(v), 0.0))))
        + 0.25 * _zscore(grouped["max_cosine_drift"])
    )
    grouped = grouped.sort_values(["score", "max_accuracy_drop"], ascending=False).head(top_k).copy()
    weights = _softmax([float(v) for v in grouped["score"].tolist()], temperature=temperature)
    grouped["weight"] = weights

    result: list[LayerSensitivity] = []
    for row in grouped.itertuples(index=False):
        result.append(
            LayerSensitivity(
                layer=str(row.layer),
                score=float(row.score),
                weight=float(row.weight),
                layer_index=int(row.layer_index) if not pd.isna(row.layer_index) else None,
                max_accuracy_drop=float(row.max_accuracy_drop),
                mean_accuracy_drop=float(row.mean_accuracy_drop),
                max_relative_l2=float(row.max_relative_l2),
                max_cosine_drift=float(row.max_cosine_drift),
            )
        )
    return result


def layer_weight_dict(items: list[LayerSensitivity]) -> dict[str, float]:
    return {item.layer: float(item.weight) for item in items}


def save_sensitivity_weights(items: list[LayerSensitivity], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([item.to_dict() for item in items]).to_csv(path, index=False)

