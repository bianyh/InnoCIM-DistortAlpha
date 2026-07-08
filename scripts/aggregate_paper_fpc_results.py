from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


CIFAR_RUNS = {
    "Clean": ROOT / "outputs" / "paper_fpc_cifar_clean_full",
    "NAT-FT grid": ROOT / "outputs" / "paper_fpc_cifar_nat_ft_full",
    "NAT-scratch grid": ROOT / "outputs" / "paper_fpc_cifar_nat_scratch_full",
    "RobustSel grid": ROOT / "outputs" / "paper_fpc_cifar_nat_robustsel_full",
    "Per-occ NAT best": ROOT / "outputs" / "paper_fpc_cifar_per_occ_scratch_full_best",
    "Per-occ NAT last": ROOT / "outputs" / "paper_fpc_cifar_per_occ_scratch_full_last",
}


VOXFORGE_RUNS = {
    "Clean": ROOT / "outputs" / "paper_fpc_voxforge_clean",
    "NAT-FT grid": ROOT / "outputs" / "paper_fpc_voxforge_nat_ft",
    "NAT-scratch grid": ROOT / "outputs" / "paper_fpc_voxforge_nat_scratch",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_cifar_runs() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_name, run_dir in CIFAR_RUNS.items():
        csv_path = run_dir / "tables" / "cifar_fpc_summary.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        frame.insert(0, "run_name", run_name)
        frame.insert(1, "source_dir", str(run_dir))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_voxforge_runs() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_name, run_dir in VOXFORGE_RUNS.items():
        csv_path = run_dir / "tables" / "voxforge_fpc_summary.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        frame.insert(0, "run_name", run_name)
        frame.insert(1, "source_dir", str(run_dir))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_cifar(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for run_name, group in long_df.groupby("run_name", sort=False):
        clean = group[group["method"] == "clean"].iloc[0]
        random_row = group[group["method"] == "random_single"].iloc[0]
        fpc = group[group["method"] == "fpc"].copy()
        best = fpc.loc[fpc["accuracy_mean"].idxmax()]
        row: dict[str, float | int | str] = {
            "run_name": run_name,
            "clean_acc": float(clean["accuracy_mean"]),
            "random_mean_acc": float(random_row["accuracy_mean"]),
            "random_worst_acc": float(random_row["accuracy_worst"]),
            "random_std_acc": float(random_row["accuracy_std"]),
            "best_fpc_bit": int(best["bits"]),
            "best_fpc_acc": float(best["accuracy_mean"]),
        }
        for bit in [1, 2, 3, 4, 5, 6, 7, 8]:
            bit_rows = fpc[fpc["bits"] == bit]
            row[f"fpc_b{bit}_acc"] = float(bit_rows["accuracy_mean"].iloc[0]) if not bit_rows.empty else float("nan")
        denom = max(float(clean["accuracy_mean"]) - float(random_row["accuracy_mean"]), 1e-12)
        for bit in [4, 5, 6, 8]:
            bit_acc = float(row[f"fpc_b{bit}_acc"])
            row[f"b{bit}_clean_gap"] = float(clean["accuracy_mean"]) - bit_acc
            row[f"b{bit}_recovery_ratio"] = (bit_acc - float(random_row["accuracy_mean"])) / denom
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_voxforge(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for run_name, group in long_df.groupby("run_name", sort=False):
        clean = group[group["method"] == "clean"].iloc[0]
        random_row = group[group["method"] == "random_single"].iloc[0]
        fpc = group[group["method"] == "fpc"].copy()
        best = fpc.loc[fpc["cer_mean"].idxmin()]
        row: dict[str, float | int | str] = {
            "run_name": run_name,
            "clean_cer": float(clean["cer_mean"]),
            "random_mean_cer": float(random_row["cer_mean"]),
            "random_worst_cer": float(random_row["cer_worst"]),
            "random_std_cer": float(random_row["cer_std"]),
            "best_fpc_bit": int(best["bits"]),
            "best_fpc_cer": float(best["cer_mean"]),
        }
        for bit in [1, 2, 3, 4, 5, 6, 7, 8]:
            bit_rows = fpc[fpc["bits"] == bit]
            row[f"fpc_b{bit}_cer"] = float(bit_rows["cer_mean"].iloc[0]) if not bit_rows.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def plot_cifar_model_comparison(summary: pd.DataFrame, figure_dir: Path) -> None:
    if summary.empty:
        return
    labels = summary["run_name"].tolist()
    x = range(len(labels))
    width = 0.18
    series = [
        ("Clean", "clean_acc", "#4c78a8", -1.5),
        ("Random field", "random_mean_acc", "#e45756", -0.5),
        ("FPC B=4", "fpc_b4_acc", "#f2cf5b", 0.5),
        ("FPC B=6", "fpc_b6_acc", "#54a24b", 1.5),
    ]
    plt.figure(figsize=(11.0, 5.0))
    for name, column, color, offset in series:
        plt.bar([i + offset * width for i in x], summary[column], width=width, label=name, color=color)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("CIFAR-10 Full-Test Robustness Under Per-Operator Random Alpha")
    plt.xticks(list(x), labels, rotation=18, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    plt.tight_layout()
    plt.savefig(figure_dir / "cifar_model_comparison.png", dpi=240)
    plt.close()


def plot_cifar_nbit(long_df: pd.DataFrame, figure_dir: Path) -> None:
    fpc = long_df[long_df["method"] == "fpc"].copy()
    if fpc.empty:
        return
    plt.figure(figsize=(9.2, 5.2))
    for run_name, group in fpc.groupby("run_name", sort=False):
        group = group.sort_values("bits")
        plt.plot(group["bits"], group["accuracy_mean"], marker="o", linewidth=1.8, label=run_name)
    plt.xlabel("FPC bit precision B")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("CIFAR-10 n-bit Ablation for Fixed-Point Bit-Serial Coding")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_dir / "cifar_nbit_ablation.png", dpi=240)
    plt.close()


def plot_training_curve(output_dir: Path, figure_dir: Path) -> None:
    csv_path = output_dir / "tables" / "cifar_training_logs.csv"
    if not csv_path.exists():
        return
    logs = pd.read_csv(csv_path)
    if logs.empty:
        return
    plt.figure(figsize=(8.8, 4.8))
    plt.plot(logs["epoch"], logs["val_clean_accuracy"], label="Val clean", color="#4c78a8")
    plt.plot(logs["epoch"], logs["val_random_accuracy"], label="Val random field", color="#e45756")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("Per-Occurrence NAT From-Scratch Training Curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "cifar_per_occ_nat_scratch_training_curve.png", dpi=240)
    plt.close()


def plot_voxforge_nbit(long_df: pd.DataFrame, figure_dir: Path) -> None:
    fpc = long_df[long_df["method"] == "fpc"].copy()
    if fpc.empty:
        return
    plt.figure(figsize=(8.8, 4.8))
    for run_name, group in fpc.groupby("run_name", sort=False):
        group = group.sort_values("bits")
        plt.plot(group["bits"], group["cer_mean"], marker="o", linewidth=1.8, label=run_name)
    plt.xlabel("FPC bit precision B")
    plt.ylabel("CER")
    plt.title("VoxForge n-bit Ablation for Fixed-Point Bit-Serial Coding")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_dir / "voxforge_nbit_ablation.png", dpi=240)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paper-level FPC experiment results.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "paper_fpc_summary")
    args = parser.parse_args()

    table_dir = ensure_dir(args.output_dir / "tables")
    figure_dir = ensure_dir(args.output_dir / "figures")

    cifar_long = read_cifar_runs()
    if not cifar_long.empty:
        cifar_long.to_csv(table_dir / "cifar_nbit_long.csv", index=False)
        cifar_summary = summarize_cifar(cifar_long)
        cifar_summary.to_csv(table_dir / "cifar_model_comparison.csv", index=False)
        plot_cifar_model_comparison(cifar_summary, figure_dir)
        plot_cifar_nbit(cifar_long, figure_dir)
        plot_training_curve(ROOT / "outputs" / "paper_cifar_per_occ_nat_scratch_full_e80", figure_dir)
        print(cifar_summary.to_string(index=False))

    vox_long = read_voxforge_runs()
    if not vox_long.empty:
        vox_long.to_csv(table_dir / "voxforge_nbit_long.csv", index=False)
        vox_summary = summarize_voxforge(vox_long)
        vox_summary.to_csv(table_dir / "voxforge_model_comparison.csv", index=False)
        plot_voxforge_nbit(vox_long, figure_dir)
        print(vox_summary.to_string(index=False))


if __name__ == "__main__":
    main()
