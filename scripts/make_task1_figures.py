from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.plotting import (
    plot_accuracy_alpha,
    plot_accuracy_drop,
    plot_activation_histograms,
    plot_layer_heatmap,
    plot_layer_metric_lines,
    plot_nonlinearity_curves,
    savefig,
    set_plot_style,
)


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    set_plot_style()
    work = summary.melt(
        id_vars=["model"],
        value_vars=["clean_accuracy", "mean_accuracy", "worst_accuracy"],
        var_name="metric",
        value_name="accuracy",
    )
    plt.figure(figsize=(8.0, 4.8))
    sns.barplot(data=work, x="model", y="accuracy", hue="metric")
    plt.xticks(rotation=15, ha="right")
    plt.xlabel("Model")
    plt.ylabel("Top-1 accuracy")
    plt.title("Clean, Mean Robust, and Worst-case Accuracy")
    savefig(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate task 1 figures from CSV tables")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "task1")
    parser.add_argument("--analysis-model", type=str, default="cifar10_resnet20")
    args = parser.parse_args()

    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_nonlinearity_curves(figures_dir / "nonlinearity_curves.png")

    accuracy_path = tables_dir / "cifar_accuracy_alpha.csv"
    if accuracy_path.exists():
        accuracy_df = pd.read_csv(accuracy_path)
        plot_accuracy_alpha(accuracy_df, figures_dir / "cifar_accuracy_alpha.png")
        plot_accuracy_drop(accuracy_df, figures_dir / "cifar_accuracy_drop.png")

    summary_path = tables_dir / "cifar_accuracy_summary.csv"
    if summary_path.exists():
        plot_summary(pd.read_csv(summary_path), figures_dir / "cifar_accuracy_summary.png")

    layer_path = tables_dir / f"{args.analysis_model}_layer_sensitivity.csv"
    if layer_path.exists():
        sens_df = pd.read_csv(layer_path)
        plot_layer_heatmap(
            sens_df,
            figures_dir / f"{args.analysis_model}_layer_sensitivity_heatmap.png",
            value_col="accuracy_drop",
            title=f"{args.analysis_model}: single-layer nonlinearity sensitivity",
        )

    drift_path = tables_dir / f"{args.analysis_model}_activation_drift.csv"
    if drift_path.exists():
        drift_df = pd.read_csv(drift_path)
        plot_layer_metric_lines(
            drift_df,
            figures_dir / f"{args.analysis_model}_activation_relative_l2.png",
            value_col="relative_l2",
            title=f"{args.analysis_model}: layer-wise accumulated relative L2 drift",
        )
        plot_layer_metric_lines(
            drift_df,
            figures_dir / f"{args.analysis_model}_activation_cosine_drift.png",
            value_col="cosine_drift",
            title=f"{args.analysis_model}: layer-wise accumulated cosine drift",
        )

    hist_path = tables_dir / f"{args.analysis_model}_activation_hist_samples.csv"
    if hist_path.exists():
        hist_df = pd.read_csv(hist_path)
        for (layer, alpha), group in hist_df.groupby(["layer", "alpha"]):
            safe_layer = str(layer).replace(".", "_").replace("/", "_")
            plot_activation_histograms(
                group,
                figures_dir / f"{args.analysis_model}_activation_hist_{safe_layer}_alpha_{alpha:g}.png",
                title=f"{args.analysis_model}: activation distribution at {layer}, alpha={alpha:g}",
            )

    print(f"Figures written to {figures_dir}")


if __name__ == "__main__":
    main()
