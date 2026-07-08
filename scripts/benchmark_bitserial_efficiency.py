from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.cifar_models import build_cifar10_loaders, build_cifar_model
from src.metrics import top1_accuracy
from src.nonlinear import (
    BitSerialFixedPointConfig,
    BitSerialFixedPointInjector,
    FP32IEEEBitSerialConfig,
    FP32IEEEBitSerialInjector,
    RandomAlphaNonlinearityInjector,
)
from src.training import ensure_dir, load_checkpoint, save_json, set_seed


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def timed_eval(model: torch.nn.Module, loader, device: torch.device, mode: str, bits: int, mantissa_bits: int) -> dict[str, float]:
    sync(device)
    start = time.perf_counter()
    correct = 0
    total = 0
    for images, labels in loader:
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
        elif mode == "fp32_ieee":
            cfg = FP32IEEEBitSerialConfig(mantissa_bits=mantissa_bits, simulate_hardware_nonlinearity=True)
            with FP32IEEEBitSerialInjector(model, cfg):
                logits = model(images)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        batch_correct, batch_total = top1_accuracy(logits, labels)
        correct += batch_correct
        total += batch_total
    sync(device)
    elapsed = time.perf_counter() - start
    return {
        "elapsed_s": elapsed,
        "accuracy": correct / max(total, 1),
        "images": total,
        "images_per_s": total / max(elapsed, 1e-12),
    }


def hardware_cost_row(method: str, bits: int, mantissa_bits: int) -> dict[str, float | int | str]:
    if method == "fpc":
        matrix_passes = max(1, int(bits))
        activation_bits = max(1, int(bits))
        coding = f"FPC-{bits}"
    elif method == "fp32_ieee":
        matrix_passes = int(mantissa_bits) + 1
        activation_bits = 32
        coding = "IEEE754 value planes"
    else:
        matrix_passes = 1
        activation_bits = 32
        coding = method
    return {
        "coding": coding,
        "matrix_passes_per_operator": matrix_passes,
        "serial_cycles_per_operator": matrix_passes,
        "activation_stream_bits": activation_bits,
        "activation_stream_vs_fp32": activation_bits / 32.0,
        "ideal_sequential_throughput_vs_clean": 1.0 / matrix_passes,
    }


def make_figures(summary: pd.DataFrame, output_dir: Path) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    fpc = summary[summary["method"] == "fpc"].sort_values("bits")
    if not fpc.empty:
        plt.figure(figsize=(8.0, 4.6))
        plt.plot(fpc["bits"], fpc["mean_elapsed_s"], marker="o", color="#4c78a8", label="measured software time")
        clean = summary[summary["method"] == "clean"]
        if not clean.empty:
            clean_time = float(clean["mean_elapsed_s"].iloc[0])
            plt.plot(fpc["bits"], clean_time * fpc["matrix_passes_per_operator"], linestyle="--", color="#e45756", label="ideal linear Bx trend")
        plt.xlabel("FPC bit precision B")
        plt.ylabel("Elapsed seconds")
        plt.title("Software Simulation Time vs FPC Bit Precision")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / "cifar_efficiency_time_vs_bits.png", dpi=240)
        plt.close()

        plt.figure(figsize=(8.0, 4.6))
        plt.plot(fpc["bits"], fpc["activation_stream_vs_fp32"], marker="o", color="#54a24b")
        plt.xlabel("FPC bit precision B")
        plt.ylabel("Activation stream bits / FP32 bits")
        plt.title("Activation Bit-Traffic Ratio vs FPC Bit Precision")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(figure_dir / "cifar_efficiency_activation_bits.png", dpi=240)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark bit-serial simulation and analytical efficiency.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "paper_efficiency_cifar")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "task2" / "checkpoints" / "cifar" / "cifar_resnet20_clean_best.pt")
    parser.add_argument("--model", type=str, default="cifar10_resnet20")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--test-subset", type=int, default=2048)
    parser.add_argument("--bits", type=str, default="1,2,3,4,5,6,7,8")
    parser.add_argument("--mantissa-bits", type=int, default=23)
    parser.add_argument("--timing-repeats", type=int, default=3)
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
        val_size=5000,
        seed=args.seed,
        test_subset=None if args.test_subset <= 0 else args.test_subset,
    )
    model = build_cifar_model(args.model, pretrained=False).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    save_json(
        {
            "dataset": "cifar10",
            "model": args.model,
            "checkpoint": str(args.checkpoint),
            "test_subset": None if args.test_subset <= 0 else args.test_subset,
            "timing_repeats": args.timing_repeats,
            "note": "Software timing includes Python hook overhead; hardware cost columns are analytical.",
        },
        table_dir / "cifar_efficiency_metadata.json",
    )

    modes: list[tuple[str, int]] = [("clean", 0), ("random_single", 0)]
    modes.extend(("fpc", bit) for bit in parse_int_list(args.bits))
    modes.append(("fp32_ieee", 32))

    rows: list[dict[str, float | int | str]] = []
    for method, bits in modes:
        samples = [timed_eval(model, test_loader, device, method, bits, args.mantissa_bits) for _ in range(max(1, args.timing_repeats))]
        elapsed = [float(item["elapsed_s"]) for item in samples]
        images_per_s = [float(item["images_per_s"]) for item in samples]
        acc = [float(item["accuracy"]) for item in samples]
        cost = hardware_cost_row(method, bits, args.mantissa_bits)
        rows.append(
            {
                "dataset": "cifar10",
                "model": args.model,
                "method": method,
                "bits": bits,
                "timing_repeats": int(args.timing_repeats),
                "mean_elapsed_s": sum(elapsed) / len(elapsed),
                "std_elapsed_s": statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0,
                "mean_images_per_s": sum(images_per_s) / len(images_per_s),
                "mean_accuracy": sum(acc) / len(acc),
                **cost,
            }
        )
    summary = pd.DataFrame(rows)
    clean_time = float(summary[summary["method"] == "clean"]["mean_elapsed_s"].iloc[0])
    summary["software_time_vs_clean"] = summary["mean_elapsed_s"] / max(clean_time, 1e-12)
    summary.to_csv(table_dir / "cifar_efficiency_summary.csv", index=False)
    make_figures(summary, output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
