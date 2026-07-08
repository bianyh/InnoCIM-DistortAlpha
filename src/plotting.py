from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_plot_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="colorblind",
        font="DejaVu Sans",
        rc={
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
        },
    )


def savefig(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_nonlinearity_curves(output_path: str | Path) -> None:
    import numpy as np

    set_plot_style()
    alphas = [-1.0, -0.5, 0.0, 0.5, 1.0]
    u = np.linspace(-1, 1, 500)
    plt.figure(figsize=(7.0, 4.6))
    for alpha in alphas:
        y = alpha * (u**3) + (1 - alpha) * u
        plt.plot(u, y, linewidth=2.1, label=f"alpha={alpha:g}")
    plt.plot(u, u, color="black", linewidth=1.0, linestyle="--", alpha=0.45)
    plt.xlabel("Normalized input u")
    plt.ylabel("Distorted response f(u)")
    plt.title("Cubic Input Nonlinearity")
    plt.legend(ncol=3)
    savefig(output_path)


def plot_accuracy_alpha(df: pd.DataFrame, output_path: str | Path) -> None:
    set_plot_style()
    plt.figure(figsize=(7.4, 4.8))
    sns.lineplot(
        data=df,
        x="alpha",
        y="accuracy",
        hue="model",
        marker="o",
        linewidth=2.2,
        markersize=6,
    )
    plt.xlabel("Nonlinearity strength alpha")
    plt.ylabel("Top-1 accuracy")
    plt.title("Inference Accuracy under Input Nonlinearity")
    plt.ylim(max(0.0, df["accuracy"].min() - 0.05), min(1.0, df["accuracy"].max() + 0.03))
    savefig(output_path)


def plot_accuracy_drop(df: pd.DataFrame, output_path: str | Path) -> None:
    set_plot_style()
    work = df.copy()
    clean = work[work["alpha"] == 0].set_index("model")["accuracy"].to_dict()
    work["accuracy_drop"] = work.apply(lambda r: clean.get(r["model"], r["accuracy"]) - r["accuracy"], axis=1)
    plt.figure(figsize=(7.4, 4.8))
    sns.lineplot(
        data=work,
        x="alpha",
        y="accuracy_drop",
        hue="model",
        marker="o",
        linewidth=2.2,
        markersize=6,
    )
    plt.axhline(0, color="black", linewidth=1.0, alpha=0.5)
    plt.xlabel("Nonlinearity strength alpha")
    plt.ylabel("Accuracy drop vs alpha=0")
    plt.title("Accuracy Degradation Trend")
    savefig(output_path)


def plot_layer_heatmap(
    df: pd.DataFrame,
    output_path: str | Path,
    value_col: str,
    title: str,
    max_layers: int | None = None,
) -> None:
    set_plot_style()
    work = df.copy()
    if max_layers is not None:
        layer_order = list(dict.fromkeys(work.sort_values("layer_index")["layer"].tolist()))[:max_layers]
        work = work[work["layer"].isin(layer_order)]
    pivot = work.pivot_table(index="layer", columns="alpha", values=value_col, aggfunc="mean")
    pivot = pivot.loc[list(dict.fromkeys(work.sort_values("layer_index")["layer"].tolist()))]
    height = min(12.0, max(4.2, 0.28 * len(pivot) + 1.5))
    plt.figure(figsize=(8.4, height))
    sns.heatmap(
        pivot,
        cmap="mako_r",
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": value_col.replace("_", " ")},
    )
    plt.xlabel("alpha")
    plt.ylabel("Layer")
    plt.title(title)
    savefig(output_path)


def plot_layer_metric_lines(
    df: pd.DataFrame,
    output_path: str | Path,
    value_col: str,
    title: str,
    alphas: list[float] | None = None,
) -> None:
    set_plot_style()
    work = df.copy()
    if alphas is not None:
        work = work[work["alpha"].isin(alphas)]
    plt.figure(figsize=(8.0, 4.8))
    sns.lineplot(
        data=work,
        x="layer_index",
        y=value_col,
        hue="alpha",
        marker="o",
        linewidth=1.9,
        markersize=4.5,
        palette="viridis",
    )
    plt.xlabel("Layer index")
    plt.ylabel(value_col.replace("_", " "))
    plt.title(title)
    savefig(output_path)


def plot_activation_histograms(df: pd.DataFrame, output_path: str | Path, title: str) -> None:
    set_plot_style()
    plt.figure(figsize=(8.2, 5.0))
    sns.kdeplot(
        data=df,
        x="value",
        hue="run",
        common_norm=False,
        linewidth=2.0,
        fill=True,
        alpha=0.16,
    )
    plt.xlabel("Activation value")
    plt.ylabel("Density")
    plt.title(title)
    savefig(output_path)


def plot_wer_alpha(df: pd.DataFrame, output_path: str | Path) -> None:
    set_plot_style()
    plt.figure(figsize=(7.4, 4.8))
    sns.lineplot(
        data=df,
        x="alpha",
        y="wer",
        hue="model",
        marker="o",
        linewidth=2.2,
        markersize=6,
    )
    plt.xlabel("Nonlinearity strength alpha")
    plt.ylabel("WER")
    plt.title("ASR Word Error Rate under Input Nonlinearity")
    savefig(output_path)
