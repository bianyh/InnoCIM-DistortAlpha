from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


RUNS = {
    "Clean": ROOT / "outputs" / "paper_fp32_ieee_cifar_clean",
    "NAT-FT grid": ROOT / "outputs" / "paper_fp32_ieee_cifar_nat_ft_grid",
    "NAT-scratch grid": ROOT / "outputs" / "paper_fp32_ieee_cifar_nat_scratch_grid",
    "RobustSel grid": ROOT / "outputs" / "paper_fp32_ieee_cifar_robustsel_grid",
    "Per-occ NAT best": ROOT / "outputs" / "paper_fp32_ieee_cifar_per_occ_best",
    "Per-occ NAT last": ROOT / "outputs" / "paper_fp32_ieee_cifar_per_occ_last",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_runs() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_name, run_dir in RUNS.items():
        csv_path = run_dir / "tables" / "cifar_fp32_ieee_summary.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        frame.insert(0, "run_name", run_name)
        frame.insert(1, "source_dir", str(run_dir))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for run_name, group in long_df.groupby("run_name", sort=False):
        row: dict[str, float | int | str] = {"run_name": run_name}
        for method in ["clean", "random_single", "fp32_ieee_no_hw", "fp32_ieee_random_hw"]:
            item = group[group["method"] == method]
            if item.empty:
                continue
            row[f"{method}_acc"] = float(item["accuracy_mean"].iloc[0])
            row[f"{method}_worst_acc"] = float(item["accuracy_worst"].iloc[0])
            row[f"{method}_std_acc"] = float(item["accuracy_std"].iloc[0])
        if "clean_acc" in row and "random_single_acc" in row and "fp32_ieee_random_hw_acc" in row:
            denom = max(float(row["clean_acc"]) - float(row["random_single_acc"]), 1e-12)
            row["fp32_recovery_ratio"] = (float(row["fp32_ieee_random_hw_acc"]) - float(row["random_single_acc"])) / denom
            row["fp32_clean_gap"] = float(row["clean_acc"]) - float(row["fp32_ieee_random_hw_acc"])
        rows.append(row)
    return pd.DataFrame(rows)


def make_figures(summary: pd.DataFrame, output_dir: Path) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    if summary.empty:
        return
    labels = summary["run_name"].tolist()
    x = range(len(labels))
    width = 0.2
    series = [
        ("Clean", "clean_acc", "#4c78a8", -1.5),
        ("Random field", "random_single_acc", "#e45756", -0.5),
        ("IEEE no-HW", "fp32_ieee_no_hw_acc", "#72b7b2", 0.5),
        ("IEEE random-HW", "fp32_ieee_random_hw_acc", "#f58518", 1.5),
    ]
    plt.figure(figsize=(11.2, 5.0))
    for name, column, color, offset in series:
        if column not in summary:
            continue
        plt.bar([i + offset * width for i in x], summary[column], width=width, color=color, label=name)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("CIFAR-10 IEEE754 FP32 Value-Plane Coding Under Random Nonlinearity")
    plt.xticks(list(x), labels, rotation=18, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    plt.tight_layout()
    plt.savefig(figure_dir / "cifar_fp32_ieee_model_comparison.png", dpi=240)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate FP32 IEEE754 bit-string experiments.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "paper_fp32_ieee_summary")
    args = parser.parse_args()

    table_dir = ensure_dir(args.output_dir / "tables")
    ensure_dir(args.output_dir / "figures")
    long_df = read_runs()
    if long_df.empty:
        raise SystemExit("No FP32 IEEE result CSVs found.")
    long_df.to_csv(table_dir / "cifar_fp32_ieee_long.csv", index=False)
    summary = summarize(long_df)
    summary.to_csv(table_dir / "cifar_fp32_ieee_model_comparison.csv", index=False)
    make_figures(summary, args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
