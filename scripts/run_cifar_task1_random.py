from __future__ import annotations

import argparse
import math
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
from src.metrics import compare_tensors, count_parameters, top1_accuracy
from src.nonlinear import ActivationRecorder, RandomAlphaNonlinearityInjector, list_target_layers
from src.training import AverageMeter, ensure_dir, load_checkpoint, save_json, set_seed


@torch.inference_mode()
def evaluate_clean(model: torch.nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_meter = AverageMeter()
    for images, labels in tqdm(loader, desc="task1 random clean eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        batch_correct, batch_total = top1_accuracy(logits, labels)
        correct += batch_correct
        total += batch_total
        loss_meter.update(float(loss.item()) / max(batch_total, 1), batch_total)
    return {"accuracy": correct / max(total, 1), "loss": loss_meter.avg}


@torch.inference_mode()
def evaluate_random(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    alpha_low: float,
    alpha_high: float,
    scope: str,
    enabled_layers: set[str] | None = None,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_meter = AverageMeter()
    for images, labels in tqdm(loader, desc="task1 random-field eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with RandomAlphaNonlinearityInjector(
            model,
            alpha_low=alpha_low,
            alpha_high=alpha_high,
            scope=scope,
            enabled_layers=enabled_layers,
        ):
            logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        batch_correct, batch_total = top1_accuracy(logits, labels)
        correct += batch_correct
        total += batch_total
        loss_meter.update(float(loss.item()) / max(batch_total, 1), batch_total)
    return {"accuracy": correct / max(total, 1), "loss": loss_meter.avg}


@torch.inference_mode()
def evaluate_random_mc(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    alpha_low: float,
    alpha_high: float,
    scope: str,
    mc: int,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_meter = AverageMeter()
    mc = max(1, int(mc))
    for images, labels in tqdm(loader, desc=f"task1 random-field MC-{mc}", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        probs = []
        for _ in range(mc):
            with RandomAlphaNonlinearityInjector(model, alpha_low=alpha_low, alpha_high=alpha_high, scope=scope):
                probs.append(F.softmax(model(images), dim=-1))
        mean_prob = torch.stack(probs, dim=0).mean(dim=0).clamp_min(1e-12)
        log_prob = torch.log(mean_prob)
        loss = F.nll_loss(log_prob, labels, reduction="sum")
        correct += int((mean_prob.argmax(dim=1) == labels).sum().item())
        total += labels.numel()
        loss_meter.update(float(loss.item()) / max(labels.numel(), 1), labels.numel())
    return {"accuracy": correct / max(total, 1), "loss": loss_meter.avg}


@torch.inference_mode()
def activation_drift_random(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    layers: list[str],
    alpha_low: float,
    alpha_high: float,
    scope: str,
    repeats: int,
    max_batches: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    model.eval()
    for repeat in range(repeats):
        for step, (images, _) in enumerate(tqdm(loader, desc=f"task1 activation drift repeat={repeat + 1}", leave=False)):
            if step >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            with ActivationRecorder(model, layers, keep_batches=1, detach_cpu=True) as clean_rec:
                model(images)
            with RandomAlphaNonlinearityInjector(model, alpha_low=alpha_low, alpha_high=alpha_high, scope=scope):
                with ActivationRecorder(model, layers, keep_batches=1, detach_cpu=True) as random_rec:
                    model(images)
            clean_outputs = clean_rec.stacked()
            random_outputs = random_rec.stacked()
            for layer in layers:
                clean = clean_outputs.get(layer)
                random = random_outputs.get(layer)
                if clean is None or random is None:
                    continue
                stats = compare_tensors(clean, random)
                rows.append(
                    {
                        "dataset": "cifar10",
                        "model": "cifar10_resnet20",
                        "protocol": "per_operator_random_alpha",
                        "repeat": repeat + 1,
                        "batch_index": step,
                        "layer": layer,
                        **stats,
                    }
                )
    return pd.DataFrame(rows)


def make_figures(output_dir: Path) -> None:
    table_dir = output_dir / "tables"
    figure_dir = ensure_dir(output_dir / "figures")
    summary_path = table_dir / "cifar_random_field_summary.csv"
    layer_path = table_dir / "cifar_layer_random_sensitivity.csv"
    drift_path = table_dir / "cifar_activation_drift_random.csv"

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        work = summary[summary["eval_type"].isin(["clean", "random_single", "random_mc"])].copy()
        plt.figure(figsize=(7.2, 4.4))
        labels = []
        values = []
        for _, row in work.iterrows():
            if row["eval_type"] == "clean":
                label = "Clean"
            elif row["eval_type"] == "random_single":
                label = "Random Field"
            else:
                label = f"MC-{int(row['mc'])}"
            labels.append(label)
            values.append(float(row["accuracy"]))
        plt.bar(labels, values, color=["#4c78a8"] + ["#f58518"] * (len(values) - 1))
        plt.ylim(0.0, 1.0)
        plt.ylabel("Accuracy")
        plt.title("CIFAR-10 Per-Operator Random Alpha Sensitivity")
        plt.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(figure_dir / "cifar_random_field_summary.png", dpi=220)
        plt.close()

    if layer_path.exists():
        layer = pd.read_csv(layer_path)
        grouped = layer.groupby("layer", as_index=False).agg(mean_drop=("accuracy_drop", "mean"))
        grouped = grouped.sort_values("mean_drop", ascending=False).head(20)
        plt.figure(figsize=(8.8, 6.0))
        plt.barh(grouped["layer"], grouped["mean_drop"], color="#b279a2")
        plt.gca().invert_yaxis()
        plt.xlabel("Accuracy drop vs clean")
        plt.title("Layer-Only Random Alpha Sensitivity")
        plt.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.savefig(figure_dir / "cifar_layer_random_sensitivity.png", dpi=220)
        plt.close()

    if drift_path.exists():
        drift = pd.read_csv(drift_path)
        grouped = (
            drift.groupby("layer", as_index=False)
            .agg(relative_l2=("relative_l2", "mean"), cosine_drift=("cosine_drift", "mean"))
            .sort_values("relative_l2", ascending=False)
            .head(20)
        )
        plt.figure(figsize=(8.8, 6.0))
        plt.barh(grouped["layer"], grouped["relative_l2"], color="#54a24b")
        plt.gca().invert_yaxis()
        plt.xlabel("Mean relative L2 drift")
        plt.title("Random-Field Activation Drift")
        plt.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.savefig(figure_dir / "cifar_random_activation_drift.png", dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1 CIFAR sensitivity under per-operator random alpha.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "task1_random")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "task2" / "checkpoints" / "cifar" / "cifar_resnet20_clean_best.pt")
    parser.add_argument("--model", type=str, default="cifar10_resnet20")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--test-subset", type=int, default=2048)
    parser.add_argument("--scope", choices=["per_tensor", "per_sample", "per_channel"], default="per_tensor")
    parser.add_argument("--alpha-low", type=float, default=-1.0)
    parser.add_argument("--alpha-high", type=float, default=1.0)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--layer-repeats", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=0, help="0 means all target layers.")
    parser.add_argument("--drift-layers", type=str, default="conv1,layer1.0.conv1,layer2.0.conv1,layer3.0.conv1,layer3.2.conv2,fc")
    parser.add_argument("--drift-repeats", type=int, default=2)
    parser.add_argument("--drift-batches", type=int, default=2)
    parser.add_argument("--mc-values", type=str, default="1,3,5,9")
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
        test_subset=args.test_subset,
    )
    model = build_cifar_model(args.model, pretrained=False).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    target_layers = [name for name, _ in list_target_layers(model)]
    if args.max_layers > 0:
        target_layers = target_layers[: args.max_layers]

    metadata = {
        "dataset": "cifar10",
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "params": count_parameters(model),
        "protocol": "per_operator_random_alpha",
        "alpha_low": args.alpha_low,
        "alpha_high": args.alpha_high,
        "target_layers": target_layers,
        "test_subset": args.test_subset,
    }
    save_json(metadata, table_dir / "cifar_random_task1_metadata.json")

    clean = evaluate_clean(model, test_loader, device)
    summary_rows: list[dict[str, float | str | int]] = [
        {
            "dataset": "cifar10",
            "model": args.model,
            "eval_type": "clean",
            "accuracy": clean["accuracy"],
            "loss": clean["loss"],
            "mc": 0,
            "repeat_count": 1,
            "accuracy_std": 0.0,
            "accuracy_worst": clean["accuracy"],
        }
    ]

    random_repeat_rows = []
    repeat_accs = []
    for repeat in range(args.random_repeats):
        result = evaluate_random(
            model,
            test_loader,
            device,
            alpha_low=args.alpha_low,
            alpha_high=args.alpha_high,
            scope=args.scope,
        )
        repeat_accs.append(float(result["accuracy"]))
        random_repeat_rows.append(
            {
                "dataset": "cifar10",
                "model": args.model,
                "protocol": "per_operator_random_alpha",
                "repeat": repeat + 1,
                "accuracy": result["accuracy"],
                "loss": result["loss"],
            }
        )
    pd.DataFrame(random_repeat_rows).to_csv(table_dir / "cifar_random_field_repeats.csv", index=False)
    summary_rows.append(
        {
            "dataset": "cifar10",
            "model": args.model,
            "eval_type": "random_single",
            "accuracy": sum(repeat_accs) / max(len(repeat_accs), 1),
            "loss": math.nan,
            "mc": 1,
            "repeat_count": len(repeat_accs),
            "accuracy_std": statistics.pstdev(repeat_accs) if len(repeat_accs) > 1 else 0.0,
            "accuracy_worst": min(repeat_accs) if repeat_accs else math.nan,
        }
    )

    for mc in [int(item.strip()) for item in args.mc_values.split(",") if item.strip()]:
        result = evaluate_random_mc(
            model,
            test_loader,
            device,
            alpha_low=args.alpha_low,
            alpha_high=args.alpha_high,
            scope=args.scope,
            mc=mc,
        )
        summary_rows.append(
            {
                "dataset": "cifar10",
                "model": args.model,
                "eval_type": "random_mc",
                "accuracy": result["accuracy"],
                "loss": result["loss"],
                "mc": mc,
                "repeat_count": 1,
                "accuracy_std": 0.0,
                "accuracy_worst": result["accuracy"],
            }
        )

    layer_rows = []
    for layer in target_layers:
        layer_accs = []
        for repeat in range(args.layer_repeats):
            result = evaluate_random(
                model,
                test_loader,
                device,
                alpha_low=args.alpha_low,
                alpha_high=args.alpha_high,
                scope=args.scope,
                enabled_layers={layer},
            )
            layer_accs.append(float(result["accuracy"]))
            layer_rows.append(
                {
                    "dataset": "cifar10",
                    "model": args.model,
                    "protocol": "single_layer_per_occurrence_random_alpha",
                    "layer": layer,
                    "repeat": repeat + 1,
                    "accuracy": result["accuracy"],
                    "loss": result["loss"],
                    "clean_accuracy": clean["accuracy"],
                    "accuracy_drop": clean["accuracy"] - result["accuracy"],
                }
            )
        if layer_accs:
            layer_rows.append(
                {
                    "dataset": "cifar10",
                    "model": args.model,
                    "protocol": "single_layer_per_occurrence_random_alpha_summary",
                    "layer": layer,
                    "repeat": 0,
                    "accuracy": sum(layer_accs) / len(layer_accs),
                    "loss": math.nan,
                    "clean_accuracy": clean["accuracy"],
                    "accuracy_drop": clean["accuracy"] - sum(layer_accs) / len(layer_accs),
                }
            )
    pd.DataFrame(layer_rows).to_csv(table_dir / "cifar_layer_random_sensitivity.csv", index=False)

    drift_layers = [item.strip() for item in args.drift_layers.split(",") if item.strip()]
    module_names = {name for name, _ in model.named_modules()}
    drift_layers = [layer for layer in drift_layers if layer in module_names]
    if drift_layers:
        drift_df = activation_drift_random(
            model=model,
            loader=test_loader,
            device=device,
            layers=drift_layers,
            alpha_low=args.alpha_low,
            alpha_high=args.alpha_high,
            scope=args.scope,
            repeats=args.drift_repeats,
            max_batches=args.drift_batches,
        )
        drift_df.to_csv(table_dir / "cifar_activation_drift_random.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(table_dir / "cifar_random_field_summary.csv", index=False)
    make_figures(output_dir)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Tables: {table_dir}")
    print(f"Figures: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
