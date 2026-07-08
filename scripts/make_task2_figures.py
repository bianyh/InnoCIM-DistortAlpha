from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.plotting import savefig, set_plot_style


METHOD_LABELS = {
    "clean": "Clean",
    "nat_finetune": "NAT Fine-tuning",
    "nat_finetune_robust": "NAT FT Robust-selected",
    "nat_scratch": "NAT From Scratch",
    "nat_curriculum": "Curriculum NAT",
    "fixed_pos": "Fixed alpha=0.2",
    "fixed_neg": "Fixed alpha=-0.2",
}


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def add_method_label(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["method_label"] = work["method"].map(METHOD_LABELS).fillna(work["method"])
    return work


def plot_cifar_train_loss(df: pd.DataFrame, output_path: Path) -> None:
    set_plot_style()
    work = add_method_label(df)
    plt.figure(figsize=(7.6, 4.8))
    sns.lineplot(data=work, x="epoch", y="train_loss", hue="method_label", linewidth=2.1)
    plt.legend(title="Method")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("CIFAR-10 Training Loss")
    savefig(output_path)


def plot_cifar_val_accuracy(df: pd.DataFrame, output_path: Path) -> None:
    set_plot_style()
    work = add_method_label(df)
    plt.figure(figsize=(7.6, 4.8))
    sns.lineplot(data=work, x="epoch", y="val_accuracy", hue="method_label", linewidth=2.1)
    plt.legend(title="Method")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("CIFAR-10 Validation Accuracy")
    plt.ylim(max(0.0, work["val_accuracy"].min() - 0.05), min(1.0, work["val_accuracy"].max() + 0.03))
    savefig(output_path)


def plot_cifar_alpha_accuracy(df: pd.DataFrame, output_path: Path) -> None:
    set_plot_style()
    work = add_method_label(df)
    plt.figure(figsize=(7.6, 4.8))
    sns.lineplot(
        data=work,
        x="alpha",
        y="accuracy",
        hue="method_label",
        marker="o",
        linewidth=2.1,
        markersize=6,
    )
    plt.legend(title="Method")
    plt.xlabel("Nonlinearity strength alpha")
    plt.ylabel("Top-1 accuracy")
    plt.title("CIFAR-10 Accuracy under Inference Nonlinearity")
    plt.ylim(max(0.0, work["accuracy"].min() - 0.05), min(1.0, work["accuracy"].max() + 0.03))
    savefig(output_path)


def plot_cifar_accuracy_drop(df: pd.DataFrame, output_path: Path) -> None:
    set_plot_style()
    work = add_method_label(df)
    clean_by_run = work[work["alpha"].abs() < 1e-12].set_index("run_name")["accuracy"].to_dict()
    work["accuracy_drop"] = work.apply(lambda r: clean_by_run.get(r["run_name"], r["accuracy"]) - r["accuracy"], axis=1)
    plt.figure(figsize=(7.6, 4.8))
    sns.lineplot(
        data=work,
        x="alpha",
        y="accuracy_drop",
        hue="method_label",
        marker="o",
        linewidth=2.1,
        markersize=6,
    )
    plt.legend(title="Method")
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    plt.xlabel("Nonlinearity strength alpha")
    plt.ylabel("Accuracy drop vs alpha=0")
    plt.title("CIFAR-10 Accuracy Drop")
    savefig(output_path)


def plot_cifar_method_summary(df: pd.DataFrame, output_path: Path) -> None:
    set_plot_style()
    work = add_method_label(df)
    long_df = work.melt(
        id_vars=["method_label"],
        value_vars=["clean_accuracy", "avg_robust_accuracy", "worst_accuracy"],
        var_name="metric",
        value_name="accuracy",
    )
    metric_labels = {
        "clean_accuracy": "Clean",
        "avg_robust_accuracy": "Avg robust",
        "worst_accuracy": "Worst",
    }
    long_df["metric"] = long_df["metric"].map(metric_labels)
    plt.figure(figsize=(8.0, 4.8))
    ax = sns.barplot(data=long_df, x="method_label", y="accuracy", hue="metric")
    plt.legend(title="Metric")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    plt.xlabel("Method")
    plt.ylabel("Accuracy")
    plt.title("CIFAR-10 Clean and Robust Accuracy Summary")
    plt.ylim(0.0, min(1.0, long_df["accuracy"].max() + 0.12))
    plt.xticks(rotation=10)
    savefig(output_path)


def plot_voxforge_train_loss(df: pd.DataFrame, output_path: Path) -> None:
    set_plot_style()
    work = add_method_label(df)
    plt.figure(figsize=(7.6, 4.8))
    sns.lineplot(data=work, x="epoch", y="train_loss", hue="method_label", linewidth=2.1)
    plt.legend(title="Method")
    plt.xlabel("Epoch")
    plt.ylabel("CTC training loss")
    plt.title("VoxForge CTC Training Loss")
    savefig(output_path)


def plot_voxforge_val_wer(df: pd.DataFrame, output_path: Path) -> None:
    set_plot_style()
    work = add_method_label(df)
    plt.figure(figsize=(7.6, 4.8))
    sns.lineplot(data=work, x="epoch", y="val_wer", hue="method_label", linewidth=2.1)
    plt.legend(title="Method")
    plt.xlabel("Epoch")
    plt.ylabel("Validation WER")
    plt.title("VoxForge Validation WER")
    savefig(output_path)


def plot_voxforge_alpha_wer(df: pd.DataFrame, output_path: Path) -> None:
    set_plot_style()
    work = add_method_label(df)
    plt.figure(figsize=(7.6, 4.8))
    sns.lineplot(
        data=work,
        x="alpha",
        y="wer",
        hue="method_label",
        marker="o",
        linewidth=2.1,
        markersize=6,
    )
    plt.legend(title="Method")
    plt.xlabel("Nonlinearity strength alpha")
    plt.ylabel("WER")
    plt.title("VoxForge WER under Inference Nonlinearity")
    savefig(output_path)


def plot_voxforge_alpha_cer(df: pd.DataFrame, output_path: Path) -> None:
    set_plot_style()
    work = add_method_label(df)
    plt.figure(figsize=(7.6, 4.8))
    sns.lineplot(
        data=work,
        x="alpha",
        y="cer",
        hue="method_label",
        marker="o",
        linewidth=2.1,
        markersize=6,
    )
    plt.legend(title="Method")
    plt.xlabel("Nonlinearity strength alpha")
    plt.ylabel("CER")
    plt.title("VoxForge CER under Inference Nonlinearity")
    savefig(output_path)


def plot_voxforge_method_summary(df: pd.DataFrame, output_path: Path) -> None:
    set_plot_style()
    work = add_method_label(df)
    long_df = work.melt(
        id_vars=["method_label"],
        value_vars=["clean_wer", "avg_robust_wer", "worst_wer"],
        var_name="metric",
        value_name="wer",
    )
    metric_labels = {
        "clean_wer": "Clean",
        "avg_robust_wer": "Avg robust",
        "worst_wer": "Worst",
    }
    long_df["metric"] = long_df["metric"].map(metric_labels)
    plt.figure(figsize=(8.0, 4.8))
    ax = sns.barplot(data=long_df, x="method_label", y="wer", hue="metric")
    plt.legend(title="Metric")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    plt.xlabel("Method")
    plt.ylabel("WER")
    plt.title("VoxForge Clean and Robust WER Summary")
    plt.xticks(rotation=10)
    savefig(output_path)


def plot_voxforge_cer_summary(df: pd.DataFrame, output_path: Path) -> None:
    set_plot_style()
    work = add_method_label(df)
    long_df = work.melt(
        id_vars=["method_label"],
        value_vars=["clean_cer", "avg_robust_cer", "worst_cer"],
        var_name="metric",
        value_name="cer",
    )
    metric_labels = {
        "clean_cer": "Clean",
        "avg_robust_cer": "Avg robust",
        "worst_cer": "Worst",
    }
    long_df["metric"] = long_df["metric"].map(metric_labels)
    plt.figure(figsize=(8.0, 4.8))
    ax = sns.barplot(data=long_df, x="method_label", y="cer", hue="metric")
    plt.legend(title="Metric")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    plt.xlabel("Method")
    plt.ylabel("CER")
    plt.title("VoxForge Clean and Robust CER Summary")
    plt.xticks(rotation=10)
    savefig(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Task 2 figures from CSV artifacts.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "task2")
    args = parser.parse_args()

    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cifar_logs = read_csv(tables_dir / "cifar_training_logs.csv")
    cifar_alpha = read_csv(tables_dir / "cifar_alpha_eval.csv")
    cifar_summary = read_csv(tables_dir / "cifar_method_summary.csv")

    if cifar_logs is not None:
        plot_cifar_train_loss(cifar_logs, figures_dir / "cifar_train_loss.png")
        plot_cifar_val_accuracy(cifar_logs, figures_dir / "cifar_val_accuracy.png")
    if cifar_alpha is not None:
        plot_cifar_alpha_accuracy(cifar_alpha, figures_dir / "cifar_alpha_accuracy.png")
        plot_cifar_accuracy_drop(cifar_alpha, figures_dir / "cifar_accuracy_drop.png")
    if cifar_summary is not None:
        plot_cifar_method_summary(cifar_summary, figures_dir / "cifar_method_summary.png")

    vox_logs = read_csv(tables_dir / "voxforge_training_logs.csv")
    vox_alpha = read_csv(tables_dir / "voxforge_alpha_eval.csv")
    vox_summary = read_csv(tables_dir / "voxforge_method_summary.csv")

    if vox_logs is not None:
        plot_voxforge_train_loss(vox_logs, figures_dir / "voxforge_train_loss.png")
        if "val_wer" in vox_logs.columns:
            plot_voxforge_val_wer(vox_logs, figures_dir / "voxforge_val_wer.png")
    if vox_alpha is not None and "wer" in vox_alpha.columns:
        plot_voxforge_alpha_wer(vox_alpha, figures_dir / "voxforge_alpha_wer.png")
        if "cer" in vox_alpha.columns:
            plot_voxforge_alpha_cer(vox_alpha, figures_dir / "voxforge_alpha_cer.png")
    if vox_summary is not None:
        plot_voxforge_method_summary(vox_summary, figures_dir / "voxforge_method_summary.png")
        if {"clean_cer", "avg_robust_cer", "worst_cer"}.issubset(vox_summary.columns):
            plot_voxforge_cer_summary(vox_summary, figures_dir / "voxforge_cer_summary.png")

    print(f"Task 2 figures written to {figures_dir}")


if __name__ == "__main__":
    main()
