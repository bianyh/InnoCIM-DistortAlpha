from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "unified_paper" / "figures"


BLUE = "#365F9C"
ORANGE = "#EBA33A"
RED = "#D85B3A"
GREEN = "#2F8C5A"
DARK_RED = "#9E1C21"
DARK = "#394150"
GRID = "#DEE3EA"
GRAY = "#8C939E"
LIGHT_GRAY = "#F2F4F7"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "font.family": "Arial",
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "axes.labelweight": "bold",
            "axes.edgecolor": "#596270",
            "axes.linewidth": 1.25,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "grid.alpha": 1.0,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "xtick.major.width": 1.15,
            "ytick.major.width": 1.15,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#D5DAE1",
            "legend.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def pct(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return x * 100.0


def style_axes(ax: plt.Axes, ygrid: bool = True, xgrid: bool = False) -> None:
    ax.grid(ygrid, axis="y")
    ax.grid(xgrid, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)


def panel_title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, pad=8, fontweight="bold")


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def annotate_value(ax: plt.Axes, x: float, y: float, text: str, color: str, dy: float = 1.0) -> None:
    ax.text(
        x,
        y + dy,
        text,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=color,
    )


def plot_fig01_distortion_and_sensitivity() -> None:
    cifar = read_csv("outputs/task1/tables/cifar_accuracy_alpha.csv")
    vox = read_csv("outputs/task1/tables/voxforge_whisper_wer_alpha.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.3))

    ax = axes[0]
    u = np.linspace(-1, 1, 401)
    alpha_specs = [
        (-1.0, DARK_RED, r"$\alpha=-1$"),
        (-0.5, ORANGE, r"$\alpha=-0.5$"),
        (0.0, BLUE, r"$\alpha=0$"),
        (0.5, GREEN, r"$\alpha=0.5$"),
        (1.0, RED, r"$\alpha=1$"),
    ]
    for alpha, color, label in alpha_specs:
        y = alpha * (u**3) + (1 - alpha) * u
        ax.plot(u, y, lw=2.8, color=color, label=label)
    ax.scatter([-1, 0, 1], [-1, 0, 1], s=72, color="white", edgecolor=DARK_RED, linewidth=2.2, zorder=5)
    ax.axline((0, 0), slope=1, color=GRAY, lw=1.7, ls=(0, (5, 4)))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("Normalized input u")
    ax.set_ylabel(r"$f_\alpha(u)$")
    panel_title(ax, "(a) Nonlinear Transfer")
    ax.legend(loc="lower right", fontsize=9, ncol=1)
    ax.text(
        0.0,
        -1.07,
        "fixed points: -1, 0, +1",
        ha="center",
        color=DARK_RED,
        fontsize=11,
        fontweight="bold",
    )
    style_axes(ax)

    ax = axes[1]
    models = [
        ("cifar10_resnet20", "ResNet20", BLUE),
        ("cifar10_mobilenetv2_x1_0", "MobileNetV2", GREEN),
    ]
    for model, label, color in models:
        sub = cifar[cifar["model"] == model].sort_values("alpha")
        ax.plot(sub["alpha"], pct(sub["accuracy"]), lw=3.0, marker="o", ms=5, color=color, label=label)
    ax.axhline(10, color=GRAY, lw=1.6, ls=(0, (4, 4)))
    ax.set_xlabel(r"Fixed $\alpha$")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_ylim(0, 100)
    panel_title(ax, "(b) CIFAR-10 Accuracy")
    ax.legend(loc="upper center", fontsize=9)
    ax.text(0.38, 13.0, "random guess", color=GRAY, fontsize=10, fontweight="bold")
    style_axes(ax)

    ax = axes[2]
    vox = vox.sort_values("alpha")
    ax.plot(vox["alpha"], vox["wer"], lw=3.0, marker="o", ms=5, color=RED, label="WER")
    ax.plot(vox["alpha"], vox["cer"], lw=3.0, marker="s", ms=5, color=ORANGE, label="CER")
    ax.axhline(float(vox.loc[vox["alpha"] == 0, "wer"].iloc[0]), color=BLUE, lw=1.6, ls=(0, (4, 4)))
    ax.set_xlabel(r"Fixed $\alpha$")
    ax.set_ylabel("Error Rate")
    ax.set_ylim(0, 1.25)
    panel_title(ax, "(c) VoxForge Whisper")
    ax.legend(loc="lower right", fontsize=9)
    style_axes(ax)

    save(fig, "fig01_distortion_and_sensitivity")


def plot_fig02_layer_and_drift() -> None:
    sens = read_csv("outputs/task1/tables/cifar10_resnet20_layer_sensitivity.csv")
    drift = read_csv("outputs/task1/tables/cifar10_resnet20_activation_drift.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5))

    ax = axes[0]
    sub = sens[sens["alpha"] == 1.0].sort_values("accuracy_drop", ascending=False).head(10)
    y = np.arange(len(sub))
    colors = [RED if i < 3 else ORANGE if i < 6 else BLUE for i in range(len(sub))]
    ax.barh(y, pct(sub["accuracy_drop"]), color=colors, edgecolor="#1F2430", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["layer"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy Drop (%)")
    panel_title(ax, "(a) Top Sensitive Layers")
    style_axes(ax, ygrid=False, xgrid=True)
    for yi, val in enumerate(pct(sub["accuracy_drop"])):
        ax.text(val + 1.0, yi, f"{val:.1f}", va="center", fontsize=9, color=DARK, fontweight="bold")
    ax.set_xlim(0, max(pct(sub["accuracy_drop"])) + 12)

    ax = axes[1]
    for alpha, color, label in [(-1.0, DARK_RED, r"$\alpha=-1$"), (0.5, ORANGE, r"$\alpha=0.5$"), (1.0, RED, r"$\alpha=1$")]:
        sub = drift[drift["alpha"] == alpha].sort_values("layer_index")
        ax.plot(sub["layer_index"], sub["cosine_drift"], lw=2.5, color=color, label=label)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Cosine Drift")
    ax.set_ylim(0, 1.08)
    panel_title(ax, "(b) Representation Drift")
    ax.legend(loc="upper left", fontsize=9)
    style_axes(ax)

    save(fig, "fig02_layer_sensitivity_and_drift")


def plot_fig03_nat_baseline() -> None:
    cifar = read_csv("outputs/task2/tables/cifar_method_summary.csv")
    vox = read_csv("outputs/task2/tables/voxforge_method_summary.csv")
    method_map = {
        "clean": "Clean",
        "nat_finetune": "NAT-FT",
        "nat_scratch": "NAT-Scratch",
        "nat_finetune_robust": "RobustSel",
    }
    colors = [BLUE, ORANGE, GREEN, RED]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4))

    ax = axes[0]
    cifar = cifar.copy()
    cifar["label"] = cifar["method"].map(method_map)
    x = np.arange(len(cifar))
    w = 0.34
    ax.bar(x - w / 2, pct(cifar["clean_accuracy"]), width=w, color=BLUE, edgecolor="#1F2430", linewidth=1.1, label="Clean Acc.")
    ax.bar(x + w / 2, pct(cifar["avg_robust_accuracy"]), width=w, color=ORANGE, edgecolor="#1F2430", linewidth=1.1, label="Avg Robust Acc.")
    ax.set_xticks(x)
    ax.set_xticklabels(cifar["label"], rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    panel_title(ax, "(a) CIFAR-10 NAT Baseline")
    ax.legend(loc="upper right", fontsize=9)
    style_axes(ax)
    for xi, val in zip(x, pct(cifar["avg_robust_accuracy"])):
        ax.text(xi + w / 2, val + 1.2, f"{val:.1f}", ha="center", fontsize=9, color=DARK, fontweight="bold")

    ax = axes[1]
    vox = vox.copy()
    vox["label"] = vox["method"].map(method_map)
    x = np.arange(len(vox))
    ax.bar(x - w / 2, pct(vox["clean_cer"]), width=w, color=BLUE, edgecolor="#1F2430", linewidth=1.1, label="Clean CER")
    ax.bar(x + w / 2, pct(vox["avg_robust_cer"]), width=w, color=ORANGE, edgecolor="#1F2430", linewidth=1.1, label="Avg Robust CER")
    ax.set_xticks(x)
    ax.set_xticklabels(vox["label"], rotation=15, ha="right")
    ax.set_ylim(65, 95)
    ax.set_ylabel("CER (%) lower is better")
    panel_title(ax, "(b) VoxForge NAT Baseline")
    ax.legend(loc="upper left", fontsize=9)
    style_axes(ax)
    for xi, val in zip(x, pct(vox["avg_robust_cer"])):
        ax.text(xi + w / 2, val + 0.7, f"{val:.1f}", ha="center", fontsize=9, color=DARK, fontweight="bold")

    save(fig, "fig03_nat_baseline")


def plot_fig04_fpc_main() -> None:
    fpc = read_csv("outputs/paper_fpc_summary/tables/cifar_model_comparison.csv")
    keep = ["Clean", "NAT-scratch grid", "Per-occ NAT last"]
    fpc = fpc[fpc["run_name"].isin(keep)].copy()

    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    x = np.arange(len(fpc))
    w = 0.18
    series = [
        ("clean_acc", "Clean", BLUE),
        ("random_mean_acc", "Random", GRAY),
        ("fpc_b5_acc", "FPC B=5", ORANGE),
        ("fpc_b6_acc", "FPC B=6", RED),
    ]
    for i, (col, label, color) in enumerate(series):
        vals = pct(fpc[col])
        ax.bar(x + (i - 1.5) * w, vals, width=w, color=color, edgecolor="#1F2430", linewidth=1.1, label=label)
        for xi, val in zip(x + (i - 1.5) * w, vals):
            ax.text(xi, val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(fpc["run_name"], rotation=0)
    ax.set_ylim(0, 102)
    ax.set_ylabel("Top-1 Accuracy (%)")
    panel_title(ax, "FPC Restores Random Nonlinear Inference")
    ax.legend(loc="upper center", ncol=4, fontsize=10)
    style_axes(ax)
    ax.annotate(
        "random nonlinear collapse",
        xy=(0 - 0.5 * w, pct(fpc["random_mean_acc"].iloc[0])),
        xytext=(0.12, 38),
        arrowprops={"arrowstyle": "->", "color": DARK_RED, "lw": 2},
        color=DARK_RED,
        fontsize=11,
        fontweight="bold",
    )
    save(fig, "fig04_fpc_main_result")


def plot_fig05_nbit_ablation() -> None:
    nbit = read_csv("outputs/paper_fpc_summary/tables/cifar_nbit_long.csv")
    keep = [
        ("Clean", BLUE),
        ("NAT-scratch grid", GREEN),
        ("Per-occ NAT last", RED),
    ]

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    for run, color in keep:
        sub = nbit[(nbit["run_name"] == run) & (nbit["method"] == "fpc")].sort_values("bits")
        ax.plot(sub["bits"], pct(sub["accuracy_mean"]), lw=3.2, marker="o", ms=6, color=color, label=run)
    rand = nbit[(nbit["run_name"] == "Clean") & (nbit["method"] == "random_single")]["accuracy_mean"].iloc[0]
    clean = nbit[(nbit["run_name"] == "Clean") & (nbit["method"] == "clean")]["accuracy_mean"].iloc[0]
    ax.axhline(pct(rand), color=GRAY, lw=2.0, ls=(0, (5, 5)), label="Random baseline")
    ax.axhline(pct(clean), color=DARK, lw=2.0, ls=(0, (2, 4)), label="Clean reference")
    ax.set_xlim(0.75, 8.25)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("FPC Bit Width B")
    ax.set_ylabel("Top-1 Accuracy (%)")
    panel_title(ax, "Accuracy Saturates after 5-6 Bits")
    ax.legend(loc="lower right", fontsize=9)
    style_axes(ax)
    ax.annotate(
        "default: B=5",
        xy=(5, 88.8),
        xytext=(4.0, 74),
        arrowprops={"arrowstyle": "->", "color": DARK_RED, "lw": 2},
        color=DARK_RED,
        fontsize=12,
        fontweight="bold",
    )
    ax.annotate(
        "near-clean",
        xy=(6, 89.3),
        xytext=(6.4, 95),
        arrowprops={"arrowstyle": "->", "color": DARK, "lw": 2},
        color=DARK,
        fontsize=12,
        fontweight="bold",
    )
    save(fig, "fig05_nbit_ablation")


def plot_fig06_fp32_counterexample() -> None:
    fp32 = read_csv("outputs/paper_fp32_ieee_summary/tables/cifar_fp32_ieee_model_comparison.csv")
    fpc = read_csv("outputs/paper_fpc_summary/tables/cifar_model_comparison.csv")

    runs = ["Clean", "NAT-scratch grid", "Per-occ NAT last"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.5))

    ax = axes[0]
    sub = fp32[fp32["run_name"].isin(runs)].copy()
    x = np.arange(len(sub))
    w = 0.22
    vals = [
        ("random_single_acc", "Random", GRAY),
        ("fp32_ieee_random_hw_acc", "FP32 IEEE", RED),
        ("fp32_ieee_no_hw_acc", "No-HW check", BLUE),
    ]
    for i, (col, label, color) in enumerate(vals):
        ax.bar(x + (i - 1) * w, pct(sub[col]), width=w, color=color, edgecolor="#1F2430", linewidth=1.1, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["run_name"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Top-1 Accuracy (%)")
    panel_title(ax, "(a) FP32 Value-Plane Test")
    ax.legend(loc="upper center", ncol=3, fontsize=9)
    style_axes(ax)

    ax = axes[1]
    merge = fp32[fp32["run_name"].isin(runs)][["run_name", "fp32_recovery_ratio"]].merge(
        fpc[fpc["run_name"].isin(runs)][["run_name", "b5_recovery_ratio", "b6_recovery_ratio"]],
        on="run_name",
    )
    x = np.arange(len(merge))
    vals = [
        ("fp32_recovery_ratio", "FP32 IEEE", RED),
        ("b5_recovery_ratio", "FPC B=5", ORANGE),
        ("b6_recovery_ratio", "FPC B=6", BLUE),
    ]
    for i, (col, label, color) in enumerate(vals):
        ax.bar(x + (i - 1) * w, pct(merge[col]), width=w, color=color, edgecolor="#1F2430", linewidth=1.1, label=label)
    ax.axhline(100, color=GRAY, lw=1.7, ls=(0, (5, 4)))
    ax.set_xticks(x)
    ax.set_xticklabels(merge["run_name"])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Recovery Ratio (%)")
    panel_title(ax, "(b) Recovery Ratio")
    ax.legend(loc="lower right", fontsize=9)
    style_axes(ax)

    save(fig, "fig06_fp32_counterexample")


def plot_fig07_efficiency_tradeoff() -> None:
    eff = read_csv("outputs/paper_efficiency_cifar/tables/cifar_efficiency_summary.csv")
    fpc = eff[eff["method"] == "fpc"].sort_values("bits").copy()
    fp32 = eff[eff["method"] == "fp32_ieee"].iloc[0]
    clean = eff[eff["method"] == "clean"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(fpc["bits"], pct(fpc["mean_accuracy"]), lw=3.2, marker="o", ms=6, color=BLUE, label="Accuracy")
    ax2.bar(
        fpc["bits"],
        pct(fpc["activation_stream_vs_fp32"]),
        width=0.55,
        color=ORANGE,
        alpha=0.75,
        edgecolor="#1F2430",
        linewidth=1.0,
        label="Activation traffic",
    )
    ax.axhline(pct(clean["mean_accuracy"]), color=GRAY, lw=1.8, ls=(0, (5, 4)))
    ax.set_xlabel("FPC Bit Width B")
    ax.set_ylabel("Accuracy (%)", color=BLUE)
    ax2.set_ylabel("Activation Traffic vs FP32 (%)", color=ORANGE)
    ax.set_ylim(0, 100)
    ax2.set_ylim(0, 30)
    ax.set_xticks(range(1, 9))
    panel_title(ax, "(a) Accuracy-Bandwidth Trade-off")
    style_axes(ax)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    handles = [
        Line2D([0], [0], color=BLUE, lw=3, marker="o", label="Accuracy"),
        Patch(facecolor=ORANGE, edgecolor="#1F2430", label="Activation traffic"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    ax = axes[1]
    methods = ["Clean", "FPC B=5", "FPC B=6", "FPC B=8", "FP32 IEEE"]
    passes = [
        clean["matrix_passes_per_operator"],
        float(fpc.loc[fpc["bits"] == 5, "matrix_passes_per_operator"].iloc[0]),
        float(fpc.loc[fpc["bits"] == 6, "matrix_passes_per_operator"].iloc[0]),
        float(fpc.loc[fpc["bits"] == 8, "matrix_passes_per_operator"].iloc[0]),
        fp32["matrix_passes_per_operator"],
    ]
    acc = [
        pct(clean["mean_accuracy"]),
        pct(float(fpc.loc[fpc["bits"] == 5, "mean_accuracy"].iloc[0])),
        pct(float(fpc.loc[fpc["bits"] == 6, "mean_accuracy"].iloc[0])),
        pct(float(fpc.loc[fpc["bits"] == 8, "mean_accuracy"].iloc[0])),
        pct(fp32["mean_accuracy"]),
    ]
    colors = [BLUE, ORANGE, RED, GREEN, DARK_RED]
    x = np.arange(len(methods))
    ax.bar(x, passes, color=colors, edgecolor="#1F2430", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=16, ha="right")
    ax.set_ylabel("Matrix Passes / Operator")
    panel_title(ax, "(b) Serial Compute Cost")
    style_axes(ax)
    for xi, p, a in zip(x, passes, acc):
        ax.text(xi, p + 0.55, f"{int(p)}x\n{a:.1f}%", ha="center", va="bottom", fontsize=9, color=DARK, fontweight="bold")
    ax.set_ylim(0, 28)

    save(fig, "fig07_efficiency_tradeoff")


def plot_fig08_voxforge_fpc() -> None:
    vox = read_csv("outputs/paper_fpc_summary/tables/voxforge_model_comparison.csv")
    runs = ["Clean", "NAT-FT grid", "NAT-scratch grid"]
    vox = vox[vox["run_name"].isin(runs)].copy()

    fig, ax = plt.subplots(figsize=(9.8, 4.7))
    x = np.arange(len(vox))
    w = 0.22
    series = [
        ("clean_cer", "Clean CER", BLUE),
        ("random_mean_cer", "Random CER", GRAY),
        ("fpc_b3_cer", "FPC B=3 CER", ORANGE),
        ("fpc_b4_cer", "FPC B=4 CER", RED),
    ]
    for i, (col, label, color) in enumerate(series):
        vals = pct(vox[col])
        ax.bar(x + (i - 1.5) * w, vals, width=w, color=color, edgecolor="#1F2430", linewidth=1.0, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(vox["run_name"])
    ax.set_ylim(70, 100)
    ax.set_ylabel("CER (%) lower is better")
    panel_title(ax, "VoxForge Cross-Task Validation")
    ax.legend(loc="upper center", ncol=4, fontsize=9)
    style_axes(ax)
    save(fig, "fig08_voxforge_fpc")


def main() -> None:
    configure_style()
    plot_fig01_distortion_and_sensitivity()
    plot_fig02_layer_and_drift()
    plot_fig03_nat_baseline()
    plot_fig04_fpc_main()
    plot_fig05_nbit_ablation()
    plot_fig06_fp32_counterexample()
    plot_fig07_efficiency_tradeoff()
    plot_fig08_voxforge_fpc()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
