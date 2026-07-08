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
from src.nonlinear import BitSerialFixedPointConfig, BitSerialFixedPointInjector, RandomAlphaNonlinearityInjector, list_target_layers
from src.training import ensure_dir, load_checkpoint, save_json, set_seed


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader, device: torch.device, mode: str, bits: int = 0, repeats: int = 1) -> dict[str, float]:
    accs: list[float] = []
    losses: list[float] = []
    for _ in range(max(1, repeats)):
        correct = 0
        total = 0
        loss_sum = 0.0
        for images, labels in tqdm(loader, desc=f"CIFAR FPC eval {mode} B={bits}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if mode == "clean":
                logits = model(images)
            elif mode == "random_single":
                with RandomAlphaNonlinearityInjector(model, alpha_low=-1.0, alpha_high=1.0):
                    logits = model(images)
            elif mode == "fpc":
                cfg = BitSerialFixedPointConfig(bits=bits, simulate_hardware_nonlinearity=True, endpoint_alpha=False)
                with BitSerialFixedPointInjector(model, cfg):
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


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def make_figures(summary: pd.DataFrame, output_dir: Path) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    labels = []
    values = []
    colors = []
    for _, row in summary.iterrows():
        method = str(row["method"])
        if method == "clean":
            labels.append("Clean")
            colors.append("#4c78a8")
        elif method == "random_single":
            labels.append("Random Field")
            colors.append("#e45756")
        else:
            labels.append(f"FPC B={int(row['bits'])}")
            colors.append("#54a24b")
        values.append(float(row["accuracy_mean"]))
    plt.figure(figsize=(9.0, 4.8))
    plt.bar(labels, values, color=colors)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("CIFAR-10 Distribution-Free Fixed-Point Bit-Serial Coding")
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(figure_dir / "cifar_fpc_accuracy_summary.png", dpi=220)
    plt.close()

    fpc = summary[summary["method"] == "fpc"].sort_values("bits")
    plt.figure(figsize=(7.4, 4.4))
    plt.plot(fpc["bits"], fpc["accuracy_mean"], marker="o", color="#54a24b", label="FPC")
    clean = summary[summary["method"] == "clean"]
    random_single = summary[summary["method"] == "random_single"]
    if not clean.empty:
        plt.axhline(float(clean["accuracy_mean"].iloc[0]), color="#4c78a8", linestyle="--", linewidth=1.2, label="clean")
    if not random_single.empty:
        plt.axhline(float(random_single["accuracy_mean"].iloc[0]), color="#e45756", linestyle=":", linewidth=1.2, label="random field")
    plt.xlabel("Bit-serial fixed-point precision B")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("FPC Accuracy vs Bit Precision")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "cifar_fpc_bit_curve.png", dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed-point bit-serial coding on CIFAR-10.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "task3_fpc_cifar")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "task2" / "checkpoints" / "cifar" / "cifar_resnet20_clean_best.pt")
    parser.add_argument("--model", type=str, default="cifar10_resnet20")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--test-subset", type=int, default=0, help="0 means the full CIFAR-10 test set.")
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--bits", type=str, default="1,2,3,4,5,6,8")
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

    metadata = {
        "dataset": "cifar10",
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "method": "FPC",
        "params": count_parameters(model),
        "target_layers": len(list_target_layers(model)),
        "bits": parse_int_list(args.bits),
        "test_subset": None if args.test_subset <= 0 else args.test_subset,
        "protocol": "per_operator_per_call_alpha_unknown_in_minus1_1",
        "alpha_assumption": "no distribution assumption; each bit-plane is invariant for any alpha",
        "fpc_hardware_alpha_sampling": "random_uniform_minus1_1_per_bit_plane",
    }
    save_json(metadata, table_dir / "cifar_fpc_metadata.json")

    rows: list[dict[str, float | int | str]] = []
    for method, bits, repeats in [("clean", 0, 1), ("random_single", 0, args.random_repeats)]:
        result = evaluate(model, test_loader, device, method, bits=bits, repeats=repeats)
        rows.append({"dataset": "cifar10", "model": args.model, "method": method, "bits": bits, "repeats": repeats, **result})
    for bits in parse_int_list(args.bits):
        result = evaluate(model, test_loader, device, "fpc", bits=bits, repeats=1)
        rows.append({"dataset": "cifar10", "model": args.model, "method": "fpc", "bits": bits, "repeats": 1, **result})
    summary = pd.DataFrame(rows)
    summary.to_csv(table_dir / "cifar_fpc_summary.csv", index=False)
    make_figures(summary, output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
