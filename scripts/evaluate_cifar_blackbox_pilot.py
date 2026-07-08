from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.cifar_models import build_cifar10_loaders, build_cifar_model
from src.metrics import count_parameters, top1_accuracy
from src.nonlinear import (
    BlackBoxDistortionConfig,
    BlackBoxRandomDistortionInjector,
    BlindPilotInverseInjector,
    list_target_layers,
)
from src.training import ensure_dir, load_checkpoint, save_json, set_seed


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    mode: str,
    family: str,
    anchors: int,
    repeats: int,
) -> dict[str, float]:
    accs: list[float] = []
    losses: list[float] = []
    for _ in range(max(1, repeats)):
        correct = 0
        total = 0
        loss_sum = 0.0
        for images, labels in tqdm(loader, desc=f"CIFAR BPIC eval {mode} {family} K={anchors}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if mode == "clean":
                logits = model(images)
            else:
                cfg = BlackBoxDistortionConfig(family=family, anchors=anchors)
                if mode == "random_blackbox":
                    with BlackBoxRandomDistortionInjector(model, cfg):
                        logits = model(images)
                elif mode == "bpic":
                    with BlindPilotInverseInjector(model, cfg):
                        logits = model(images)
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
            batch_correct, batch_total = top1_accuracy(logits, labels)
            correct += batch_correct
            total += batch_total
            loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
        accs.append(correct / max(total, 1))
        losses.append(loss_sum / max(total, 1))
    return {
        "accuracy_mean": sum(accs) / len(accs),
        "accuracy_worst": min(accs),
        "accuracy_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
        "loss_mean": sum(losses) / len(losses),
    }


def make_figures(summary: pd.DataFrame, output_dir: Path) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    non_clean = summary[summary["method"] != "clean"].copy()
    if non_clean.empty:
        return

    best_rows = []
    for family, group in non_clean.groupby("family", sort=False):
        random_rows = group[group["method"] == "random_blackbox"]
        if not random_rows.empty:
            best_rows.append(random_rows.iloc[0].to_dict())
        bpic_rows = group[group["method"] == "bpic"]
        if not bpic_rows.empty:
            best_rows.append(bpic_rows.loc[bpic_rows["accuracy_mean"].idxmax()].to_dict())
    best = pd.DataFrame(best_rows)
    if not best.empty:
        plt.figure(figsize=(9.2, 4.8))
        families = list(dict.fromkeys(best["family"].tolist()))
        x = range(len(families))
        width = 0.32
        random_vals = []
        bpic_vals = []
        for family in families:
            random_row = best[(best["family"] == family) & (best["method"] == "random_blackbox")]
            bpic_row = best[(best["family"] == family) & (best["method"] == "bpic")]
            random_vals.append(float(random_row["accuracy_mean"].iloc[0]) if not random_row.empty else float("nan"))
            bpic_vals.append(float(bpic_row["accuracy_mean"].iloc[0]) if not bpic_row.empty else float("nan"))
        plt.bar([i - width / 2 for i in x], random_vals, width=width, color="#e45756", label="unknown random distortion")
        plt.bar([i + width / 2 for i in x], bpic_vals, width=width, color="#54a24b", label="BPIC best anchor count")
        plt.xticks(list(x), families, rotation=15, ha="right")
        plt.ylim(0.0, 1.0)
        plt.ylabel("Accuracy")
        plt.title("Formula-Agnostic Blind Pilot Inversion Calibration")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / "cifar_bpic_family_comparison.png", dpi=240)
        plt.close()

    bpic = non_clean[non_clean["method"] == "bpic"].copy()
    if not bpic.empty:
        plt.figure(figsize=(8.6, 4.8))
        for family, group in bpic.groupby("family", sort=False):
            group = group.sort_values("anchors")
            plt.plot(group["anchors"], group["accuracy_mean"], marker="o", linewidth=1.8, label=family)
        plt.xlabel("Pilot anchors per operator call")
        plt.ylabel("Accuracy")
        plt.ylim(0.0, 1.0)
        plt.title("BPIC Anchor Count Ablation")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / "cifar_bpic_anchor_ablation.png", dpi=240)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate formula-agnostic blind pilot inverse calibration on CIFAR-10.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "paper_task4_bpic_cifar")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "task2" / "checkpoints" / "cifar" / "cifar_resnet20_clean_best.pt")
    parser.add_argument("--model", type=str, default="cifar10_resnet20")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--test-subset", type=int, default=0, help="0 means full CIFAR-10 test set.")
    parser.add_argument("--families", type=str, default="contest_cubic,gamma,tanh,sinusoid,mixed")
    parser.add_argument("--anchors", type=str, default="9,17,33,65")
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--bpic-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    table_dir = ensure_dir(output_dir / "tables")
    ensure_dir(output_dir / "figures")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = build_cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        val_size=args.val_size,
        seed=args.seed,
        test_subset=None if args.test_subset <= 0 else args.test_subset,
    )
    model = build_cifar_model(args.model, pretrained=False).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    families = parse_str_list(args.families)
    anchors = parse_int_list(args.anchors)
    metadata = {
        "dataset": "cifar10",
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "method": "BPIC_Blind_Pilot_Inversion_Calibration",
        "params": count_parameters(model),
        "target_layers": len(list_target_layers(model)),
        "families": families,
        "anchors": anchors,
        "test_subset": None if args.test_subset <= 0 else args.test_subset,
        "formula_use": "method uses only pilot input-output pairs; formulas are used only to simulate black-box hardware distortions",
    }
    save_json(metadata, table_dir / "cifar_bpic_metadata.json")

    rows: list[dict[str, float | int | str]] = []
    clean = evaluate(model, test_loader, device, "clean", family="none", anchors=0, repeats=1)
    rows.append({"dataset": "cifar10", "model": args.model, "family": "none", "method": "clean", "anchors": 0, "repeats": 1, **clean})
    for family in families:
        random_result = evaluate(model, test_loader, device, "random_blackbox", family=family, anchors=0, repeats=args.random_repeats)
        rows.append(
            {
                "dataset": "cifar10",
                "model": args.model,
                "family": family,
                "method": "random_blackbox",
                "anchors": 0,
                "repeats": int(args.random_repeats),
                **random_result,
            }
        )
        for anchor_count in anchors:
            result = evaluate(model, test_loader, device, "bpic", family=family, anchors=anchor_count, repeats=args.bpic_repeats)
            rows.append(
                {
                    "dataset": "cifar10",
                    "model": args.model,
                    "family": family,
                    "method": "bpic",
                    "anchors": int(anchor_count),
                    "repeats": int(args.bpic_repeats),
                    **result,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(table_dir / "cifar_bpic_summary.csv", index=False)
    make_figures(summary, output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
