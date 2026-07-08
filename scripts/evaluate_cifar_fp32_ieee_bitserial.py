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
    FP32IEEEBitSerialConfig,
    FP32IEEEBitSerialInjector,
    RandomAlphaNonlinearityInjector,
    list_target_layers,
)
from src.training import ensure_dir, load_checkpoint, save_json, set_seed


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    mode: str,
    mantissa_bits: int,
    repeats: int = 1,
) -> dict[str, float]:
    accs: list[float] = []
    losses: list[float] = []
    for _ in range(max(1, repeats)):
        correct = 0
        total = 0
        loss_sum = 0.0
        for images, labels in tqdm(loader, desc=f"CIFAR FP32 IEEE eval {mode}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if mode == "clean":
                logits = model(images)
            elif mode == "random_single":
                with RandomAlphaNonlinearityInjector(model, alpha_low=-1.0, alpha_high=1.0):
                    logits = model(images)
            elif mode == "fp32_ieee_no_hw":
                cfg = FP32IEEEBitSerialConfig(
                    mantissa_bits=mantissa_bits,
                    include_implicit_bit=True,
                    simulate_hardware_nonlinearity=False,
                )
                with FP32IEEEBitSerialInjector(model, cfg):
                    logits = model(images)
            elif mode == "fp32_ieee_random_hw":
                cfg = FP32IEEEBitSerialConfig(
                    mantissa_bits=mantissa_bits,
                    include_implicit_bit=True,
                    simulate_hardware_nonlinearity=True,
                    alpha_low=-1.0,
                    alpha_high=1.0,
                )
                with FP32IEEEBitSerialInjector(model, cfg):
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
    label_map = {
        "clean": "Clean",
        "random_single": "Random field",
        "fp32_ieee_no_hw": "IEEE planes no-HW",
        "fp32_ieee_random_hw": "IEEE planes random-HW",
    }
    color_map = {
        "clean": "#4c78a8",
        "random_single": "#e45756",
        "fp32_ieee_no_hw": "#72b7b2",
        "fp32_ieee_random_hw": "#f58518",
    }
    labels = [label_map.get(str(m), str(m)) for m in summary["method"]]
    colors = [color_map.get(str(m), "#999999") for m in summary["method"]]
    plt.figure(figsize=(8.6, 4.8))
    plt.bar(labels, summary["accuracy_mean"], color=colors)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("CIFAR-10 IEEE754 FP32 Bit-String Value-Plane Experiment")
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(figure_dir / "cifar_fp32_ieee_accuracy.png", dpi=240)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IEEE754 FP32 bit-string value-plane coding on CIFAR-10.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "paper_fp32_ieee_cifar")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "task2" / "checkpoints" / "cifar" / "cifar_resnet20_clean_best.pt")
    parser.add_argument("--model", type=str, default="cifar10_resnet20")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--test-subset", type=int, default=0, help="0 means full CIFAR-10 test set.")
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--fp32-repeats", type=int, default=3)
    parser.add_argument("--mantissa-bits", type=int, default=23)
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

    plane_count = int(args.mantissa_bits) + 1
    metadata = {
        "dataset": "cifar10",
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "method": "IEEE754_FP32_value_plane_bit_serial",
        "params": count_parameters(model),
        "target_layers": len(list_target_layers(model)),
        "mantissa_bits": int(args.mantissa_bits),
        "plane_count": plane_count,
        "test_subset": None if args.test_subset <= 0 else args.test_subset,
        "protocol": "per_operator_per_call_alpha_unknown_in_minus1_1",
        "hardware_alpha_sampling": "random_uniform_minus1_1_per_fp32_value_plane",
        "note": "Raw IEEE sign/exponent storage bits are not additive; experiment uses IEEE754-decoded additive value planes.",
    }
    save_json(metadata, table_dir / "cifar_fp32_ieee_metadata.json")

    rows: list[dict[str, float | int | str]] = []
    schedule = [
        ("clean", 1),
        ("random_single", args.random_repeats),
        ("fp32_ieee_no_hw", 1),
        ("fp32_ieee_random_hw", args.fp32_repeats),
    ]
    for method, repeats in schedule:
        result = evaluate(model, test_loader, device, method, mantissa_bits=args.mantissa_bits, repeats=repeats)
        rows.append(
            {
                "dataset": "cifar10",
                "model": args.model,
                "method": method,
                "mantissa_bits": int(args.mantissa_bits),
                "plane_count": plane_count,
                "repeats": int(repeats),
                **result,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(table_dir / "cifar_fp32_ieee_summary.csv", index=False)
    make_figures(summary, output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
