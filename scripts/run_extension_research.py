from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.cifar_models import build_cifar10_loaders, build_cifar_model
from src.metrics import compare_tensors, count_parameters, top1_accuracy
from src.nonlinear import (
    BitSerialFixedPointConfig,
    BitSerialFixedPointInjector,
    GaussianNoiseConfig,
    GaussianNoiseInjector,
    NonlinearityConfig,
    NonlinearityInjector,
    RandomAlphaNonlinearityInjector,
    UniformQuantizationConfig,
    UniformQuantizationInjector,
    list_target_layers,
)
from src.training import ensure_dir, load_checkpoint, save_json, set_seed


ContextFactory = Callable[[], object]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def model_family(model_name: str) -> str:
    lowered = model_name.lower()
    if "resnet" in lowered:
        return "resnet"
    if "mobilenet" in lowered:
        return "mobilenetv2"
    if "vgg" in lowered:
        return "vgg"
    if "shufflenet" in lowered:
        return "shufflenetv2"
    if "repvgg" in lowered:
        return "repvgg"
    if "vit" in lowered:
        return "vit"
    return "other"


def summarize_accuracy_sweep(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    if df.empty:
        return pd.DataFrame(rows)
    for model_name, group in df.groupby("model", sort=False):
        group = group.copy()
        group["alpha"] = group["alpha"].astype(float)
        clean_rows = group[group["alpha"] == 0.0]
        clean_acc = float(clean_rows["accuracy"].iloc[0]) if not clean_rows.empty else float(group["accuracy"].max())
        mild = group[(group["alpha"].abs() <= 0.2) & (group["alpha"] != 0.0)]
        strong = group[group["alpha"].abs() >= 0.5]
        rows.append(
            {
                "model": model_name,
                "family": model_family(str(model_name)),
                "clean_accuracy": clean_acc,
                "mean_accuracy": float(group["accuracy"].mean()),
                "mild_mean_accuracy": float(mild["accuracy"].mean()) if not mild.empty else math.nan,
                "strong_mean_accuracy": float(strong["accuracy"].mean()) if not strong.empty else math.nan,
                "worst_accuracy": float(group["accuracy"].min()),
                "max_accuracy_drop": float(clean_acc - group["accuracy"].min()),
                "params": int(group["params"].iloc[0]) if "params" in group else 0,
                "params_m": float(group["params"].iloc[0]) / 1_000_000.0 if "params" in group else 0.0,
                "target_layers": int(group["target_layers"].iloc[0]) if "target_layers" in group else 0,
                "target_layers_per_mparam": (
                    float(group["target_layers"].iloc[0]) / max(float(group["params"].iloc[0]) / 1_000_000.0, 1e-12)
                    if "params" in group and "target_layers" in group
                    else 0.0
                ),
                "sample_count": int(group["total"].iloc[0]) if "total" in group else 0,
            }
        )
    return pd.DataFrame(rows)


def save_correlations(summary: pd.DataFrame, path: Path) -> None:
    cols = ["params", "target_layers", "target_layers_per_mparam", "mean_accuracy", "worst_accuracy", "max_accuracy_drop"]
    usable = summary[[col for col in cols if col in summary.columns]].dropna()
    if len(usable) < 3:
        corr = pd.DataFrame()
    else:
        corr = usable.corr(method="spearman")
    corr.to_csv(path, index_label="metric")


@torch.inference_mode()
def evaluate_with_context(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    context_factory: ContextFactory,
    desc: str,
    repeats: int = 1,
) -> dict[str, float]:
    accs: list[float] = []
    losses: list[float] = []
    repeats = max(1, int(repeats))
    for repeat in range(repeats):
        model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        with context_factory():
            for images, labels in tqdm(loader, desc=f"{desc} r={repeat + 1}", leave=False):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                batch_correct, batch_total = top1_accuracy(logits, labels)
                correct += batch_correct
                total += batch_total
                loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
        accs.append(correct / max(total, 1))
        losses.append(loss_sum / max(total, 1))
    return {
        "accuracy_mean": float(sum(accs) / len(accs)),
        "accuracy_worst": float(min(accs)),
        "accuracy_std": float(statistics.pstdev(accs)) if len(accs) > 1 else 0.0,
        "loss_mean": float(sum(losses) / len(losses)),
        "repeats": repeats,
    }


@torch.inference_mode()
def logit_drift_with_context(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    context_factory: ContextFactory,
    desc: str,
    max_batches: int,
) -> dict[str, float]:
    clean_chunks: list[torch.Tensor] = []
    pert_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    max_batches = int(max_batches)
    for step, (images, labels) in enumerate(tqdm(loader, desc=desc, leave=False)):
        if max_batches > 0 and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clean_logits = model(images)
        with context_factory():
            pert_logits = model(images)
        clean_chunks.append(clean_logits.detach().cpu())
        pert_chunks.append(pert_logits.detach().cpu())
        label_chunks.append(labels.detach().cpu())

    if not clean_chunks:
        return {}
    clean = torch.cat(clean_chunks, dim=0)
    pert = torch.cat(pert_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)
    stats = compare_tensors(clean, pert)
    clean_prob = F.softmax(clean, dim=1).clamp_min(1e-12)
    pert_prob = F.softmax(pert, dim=1).clamp_min(1e-12)
    kl = (clean_prob * (clean_prob.log() - pert_prob.log())).sum(dim=1).mean()
    clean_pred = clean.argmax(dim=1)
    pert_pred = pert.argmax(dim=1)
    clean_correct = clean_pred == labels
    pert_correct = pert_pred == labels
    stats.update(
        {
            "kl_clean_to_pert": float(kl.item()),
            "argmax_flip_ratio": float((clean_pred != pert_pred).float().mean().item()),
            "clean_accuracy_subset": float(clean_correct.float().mean().item()),
            "pert_accuracy_subset": float(pert_correct.float().mean().item()),
            "samples": int(labels.numel()),
        }
    )
    return stats


def load_pretrained_hub_model(model_name: str, device: torch.device) -> torch.nn.Module:
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=True, trust_repo=True)
    model.eval()
    return model.to(device)


def run_structure_sweep(args, device: torch.device) -> None:
    table_dir = ensure_dir(args.output_dir / "tables")
    fig_dir = ensure_dir(args.output_dir / "figures")
    _, _, loader = build_cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        val_size=args.val_size,
        seed=args.seed,
        test_subset=None if args.test_subset <= 0 else args.test_subset,
    )

    existing_path = ROOT / "outputs" / "task1" / "tables" / "cifar_accuracy_alpha.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        existing_summary = summarize_accuracy_sweep(existing)
        existing_summary.to_csv(table_dir / "structure_existing_full_summary.csv", index=False)
        save_correlations(existing_summary, table_dir / "structure_existing_full_spearman.csv")

    if args.skip_structure_sweep:
        return

    rows: list[dict[str, float | int | str]] = []
    failures: list[dict[str, str]] = []
    alphas = parse_float_list(args.structure_alphas)
    for model_name in parse_str_list(args.structure_models):
        try:
            model = load_pretrained_hub_model(model_name, device)
        except Exception as exc:  # noqa: BLE001 - record model load failures in the experiment output.
            failures.append({"model": model_name, "error": repr(exc)})
            continue

        target_layers = list_target_layers(model)
        type_counts = Counter(type(module).__name__ for _, module in target_layers)
        params = count_parameters(model)
        for alpha in alphas:
            result = evaluate_with_context(
                model,
                loader,
                device,
                lambda alpha=alpha: NonlinearityInjector(model, NonlinearityConfig(alpha=alpha, scope=args.scope)),
                desc=f"structure {model_name} alpha={alpha:g}",
                repeats=1,
            )
            rows.append(
                {
                    "dataset": "cifar10",
                    "model": model_name,
                    "family": model_family(model_name),
                    "alpha": float(alpha),
                    "accuracy": result["accuracy_mean"],
                    "loss": result["loss_mean"],
                    "total": 0 if args.test_subset <= 0 else int(args.test_subset),
                    "params": params,
                    "target_layers": len(target_layers),
                    "conv2d_layers": int(type_counts.get("Conv2d", 0)),
                    "linear_layers": int(type_counts.get("Linear", 0)),
                    "scope": args.scope,
                }
            )
        del model
        torch.cuda.empty_cache()

    sweep = pd.DataFrame(rows)
    sweep.to_csv(table_dir / "structure_alpha_sweep.csv", index=False)
    summary = summarize_accuracy_sweep(sweep)
    summary.to_csv(table_dir / "structure_summary.csv", index=False)
    save_correlations(summary, table_dir / "structure_spearman.csv")
    pd.DataFrame(failures).to_csv(table_dir / "structure_model_failures.csv", index=False)
    plot_structure_results(sweep, summary, fig_dir)


def load_checkpoint_model(args, device: torch.device) -> torch.nn.Module:
    model = build_cifar_model(args.model, pretrained=False).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    return model


def run_mechanism_comparison(args, device: torch.device) -> None:
    table_dir = ensure_dir(args.output_dir / "tables")
    fig_dir = ensure_dir(args.output_dir / "figures")
    _, _, loader = build_cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        val_size=args.val_size,
        seed=args.seed,
        test_subset=None if args.test_subset <= 0 else args.test_subset,
    )
    model = load_checkpoint_model(args, device)
    rows: list[dict[str, float | int | str]] = []

    clean = evaluate_with_context(model, loader, device, lambda: nullcontext(), "mechanism clean", repeats=1)
    rows.append({"dataset": "cifar10", "model": args.model, "mode": "clean", "severity": 0.0, "label": "clean", **clean})

    for sigma in parse_float_list(args.gaussian_sigmas):
        result = evaluate_with_context(
            model,
            loader,
            device,
            lambda sigma=sigma: GaussianNoiseInjector(
                model,
                GaussianNoiseConfig(sigma=sigma, relative=True, scope=args.scope),
            ),
            desc=f"gaussian sigma={sigma:g}",
            repeats=args.random_repeats,
        )
        rows.append(
            {
                "dataset": "cifar10",
                "model": args.model,
                "mode": "gaussian",
                "severity": float(sigma),
                "label": f"sigma={sigma:g}",
                **result,
            }
        )

    for alpha in parse_float_list(args.nonlinear_alphas):
        result = evaluate_with_context(
            model,
            loader,
            device,
            lambda alpha=alpha: NonlinearityInjector(model, NonlinearityConfig(alpha=alpha, scope=args.scope)),
            desc=f"nonlinear alpha={alpha:g}",
            repeats=1,
        )
        rows.append(
            {
                "dataset": "cifar10",
                "model": args.model,
                "mode": "nonlinear_fixed_alpha",
                "severity": float(alpha),
                "label": f"alpha={alpha:g}",
                **result,
            }
        )

    random_result = evaluate_with_context(
        model,
        loader,
        device,
        lambda: RandomAlphaNonlinearityInjector(model, alpha_low=-1.0, alpha_high=1.0, scope=args.scope),
        desc="nonlinear random",
        repeats=args.random_repeats,
    )
    rows.append(
        {
            "dataset": "cifar10",
            "model": args.model,
            "mode": "nonlinear_random_alpha",
            "severity": 1.0,
            "label": "alpha~U[-1,1]",
            **random_result,
        }
    )

    accuracy = pd.DataFrame(rows)
    accuracy.to_csv(table_dir / "mechanism_accuracy.csv", index=False)

    drift_rows: list[dict[str, float | int | str]] = []
    drift_specs: list[tuple[str, str, float, ContextFactory]] = [
        ("gaussian", "sigma=0.05", 0.05, lambda: GaussianNoiseInjector(model, GaussianNoiseConfig(sigma=0.05, relative=True, scope=args.scope))),
        ("gaussian", "sigma=0.1", 0.1, lambda: GaussianNoiseInjector(model, GaussianNoiseConfig(sigma=0.1, relative=True, scope=args.scope))),
        ("gaussian", "sigma=0.2", 0.2, lambda: GaussianNoiseInjector(model, GaussianNoiseConfig(sigma=0.2, relative=True, scope=args.scope))),
        (
            "nonlinear_fixed_alpha",
            "alpha=-0.1",
            -0.1,
            lambda: NonlinearityInjector(model, NonlinearityConfig(alpha=-0.1, scope=args.scope)),
        ),
        (
            "nonlinear_fixed_alpha",
            "alpha=0.1",
            0.1,
            lambda: NonlinearityInjector(model, NonlinearityConfig(alpha=0.1, scope=args.scope)),
        ),
        (
            "nonlinear_fixed_alpha",
            "alpha=0.2",
            0.2,
            lambda: NonlinearityInjector(model, NonlinearityConfig(alpha=0.2, scope=args.scope)),
        ),
        (
            "nonlinear_random_alpha",
            "alpha~U[-1,1]",
            1.0,
            lambda: RandomAlphaNonlinearityInjector(model, alpha_low=-1.0, alpha_high=1.0, scope=args.scope),
        ),
    ]
    for mode, label, severity, context_factory in drift_specs:
        stats = logit_drift_with_context(
            model,
            loader,
            device,
            context_factory,
            desc=f"logit drift {label}",
            max_batches=args.drift_batches,
        )
        drift_rows.append({"dataset": "cifar10", "model": args.model, "mode": mode, "label": label, "severity": severity, **stats})
    drift = pd.DataFrame(drift_rows)
    drift.to_csv(table_dir / "mechanism_logit_drift.csv", index=False)
    plot_mechanism_results(accuracy, drift, fig_dir)
    del model
    torch.cuda.empty_cache()


def run_quantization_nonlinearity(args, device: torch.device) -> None:
    table_dir = ensure_dir(args.output_dir / "tables")
    fig_dir = ensure_dir(args.output_dir / "figures")
    _, _, loader = build_cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        val_size=args.val_size,
        seed=args.seed,
        test_subset=None if args.test_subset <= 0 else args.test_subset,
    )
    model = load_checkpoint_model(args, device)
    rows: list[dict[str, float | int | str]] = []
    clean = evaluate_with_context(model, loader, device, lambda: nullcontext(), "quant clean", repeats=1)
    rows.append({"dataset": "cifar10", "model": args.model, "method": "clean", "bits": 0, "label": "clean", **clean})

    random_result = evaluate_with_context(
        model,
        loader,
        device,
        lambda: RandomAlphaNonlinearityInjector(model, alpha_low=-1.0, alpha_high=1.0, scope=args.scope),
        desc="quant baseline random nonlinear",
        repeats=args.random_repeats,
    )
    rows.append(
        {
            "dataset": "cifar10",
            "model": args.model,
            "method": "nonlinear_random_alpha",
            "bits": 0,
            "label": "alpha~U[-1,1]",
            **random_result,
        }
    )

    for bits in parse_int_list(args.quant_bits):
        quant = evaluate_with_context(
            model,
            loader,
            device,
            lambda bits=bits: UniformQuantizationInjector(
                model,
                UniformQuantizationConfig(bits=bits, scope=args.scope, simulate_hardware_nonlinearity=False),
            ),
            desc=f"quant only B={bits}",
            repeats=1,
        )
        rows.append({"dataset": "cifar10", "model": args.model, "method": "uniform_quant", "bits": bits, "label": f"Q{bits}", **quant})

        quant_nl = evaluate_with_context(
            model,
            loader,
            device,
            lambda bits=bits: UniformQuantizationInjector(
                model,
                UniformQuantizationConfig(
                    bits=bits,
                    scope=args.scope,
                    simulate_hardware_nonlinearity=True,
                    random_alpha=True,
                    alpha_low=-1.0,
                    alpha_high=1.0,
                ),
            ),
            desc=f"quant plus nonlinear B={bits}",
            repeats=args.random_repeats,
        )
        rows.append(
            {
                "dataset": "cifar10",
                "model": args.model,
                "method": "uniform_quant_plus_random_nonlinear",
                "bits": bits,
                "label": f"Q{bits}+NL",
                **quant_nl,
            }
        )

        if args.include_fpc_reference:
            fpc = evaluate_with_context(
                model,
                loader,
                device,
                lambda bits=bits: BitSerialFixedPointInjector(
                    model,
                    BitSerialFixedPointConfig(bits=bits, scope=args.scope, simulate_hardware_nonlinearity=True, endpoint_alpha=False),
                ),
                desc=f"fpc B={bits}",
                repeats=1,
            )
            rows.append({"dataset": "cifar10", "model": args.model, "method": "fpc_reference", "bits": bits, "label": f"FPC{bits}", **fpc})

    summary = pd.DataFrame(rows)
    summary.to_csv(table_dir / "quantization_nonlinearity.csv", index=False)

    clean_acc = float(summary.loc[summary["method"] == "clean", "accuracy_mean"].iloc[0])
    decomp_rows: list[dict[str, float | int]] = []
    for bits in parse_int_list(args.quant_bits):
        q = summary[(summary["method"] == "uniform_quant") & (summary["bits"] == bits)]
        qnl = summary[(summary["method"] == "uniform_quant_plus_random_nonlinear") & (summary["bits"] == bits)]
        fpc = summary[(summary["method"] == "fpc_reference") & (summary["bits"] == bits)]
        if q.empty or qnl.empty:
            continue
        q_acc = float(q["accuracy_mean"].iloc[0])
        qnl_acc = float(qnl["accuracy_mean"].iloc[0])
        decomp_rows.append(
            {
                "bits": bits,
                "clean_accuracy": clean_acc,
                "uniform_quant_accuracy": q_acc,
                "uniform_quant_plus_random_nonlinear_accuracy": qnl_acc,
                "fpc_reference_accuracy": float(fpc["accuracy_mean"].iloc[0]) if not fpc.empty else math.nan,
                "quantization_drop": clean_acc - q_acc,
                "nonlinear_extra_drop_after_quantization": q_acc - qnl_acc,
                "total_drop": clean_acc - qnl_acc,
            }
        )
    decomp = pd.DataFrame(decomp_rows)
    decomp.to_csv(table_dir / "quantization_error_decomposition.csv", index=False)
    plot_quantization_results(summary, decomp, fig_dir)
    del model
    torch.cuda.empty_cache()


def plot_structure_results(sweep: pd.DataFrame, summary: pd.DataFrame, fig_dir: Path) -> None:
    if sweep.empty or summary.empty:
        return
    plt.figure(figsize=(8.2, 4.8))
    for model_name, group in sweep.groupby("model", sort=False):
        group = group.sort_values("alpha")
        plt.plot(group["alpha"], group["accuracy"], marker="o", linewidth=2.0, label=model_name.replace("cifar10_", ""))
    plt.xlabel("alpha")
    plt.ylabel("Top-1 accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("Architecture sensitivity to fixed nonlinear distortion")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_dir / "structure_accuracy_alpha.png", dpi=240)
    plt.close()

    plt.figure(figsize=(7.4, 4.8))
    sizes = summary["target_layers"].astype(float).clip(lower=1.0) * 10.0
    for _, row in summary.iterrows():
        plt.scatter(row["params_m"], row["max_accuracy_drop"], s=float(sizes.loc[row.name]), alpha=0.75)
        plt.text(row["params_m"], row["max_accuracy_drop"] + 0.015, str(row["model"]).replace("cifar10_", ""), fontsize=8, ha="center")
    plt.xscale("log")
    plt.xlabel("Parameters (M, log scale)")
    plt.ylabel("Max accuracy drop")
    plt.ylim(0.0, min(1.0, max(summary["max_accuracy_drop"]) + 0.12))
    plt.title("Parameter count is not a sufficient robustness predictor")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "structure_drop_vs_params.png", dpi=240)
    plt.close()


def plot_mechanism_results(accuracy: pd.DataFrame, drift: pd.DataFrame, fig_dir: Path) -> None:
    if accuracy.empty:
        return
    gaussian = accuracy[accuracy["mode"] == "gaussian"].sort_values("severity")
    nonlinear = accuracy[accuracy["mode"] == "nonlinear_fixed_alpha"].sort_values("severity")
    clean = accuracy[accuracy["mode"] == "clean"]
    random_rows = accuracy[accuracy["mode"] == "nonlinear_random_alpha"]

    plt.figure(figsize=(10.0, 4.6))
    ax1 = plt.subplot(1, 2, 1)
    if not clean.empty:
        ax1.axhline(float(clean["accuracy_mean"].iloc[0]), color="#365f9c", linestyle="--", linewidth=1.4, label="clean")
    if not gaussian.empty:
        ax1.plot(gaussian["severity"], gaussian["accuracy_mean"], marker="o", linewidth=2.0, color="#2f8c5a", label="Gaussian")
    if not random_rows.empty:
        ax1.axhline(float(random_rows["accuracy_mean"].iloc[0]), color="#9e1c21", linestyle=":", linewidth=1.6, label="random alpha")
    ax1.set_xlabel("Gaussian sigma")
    ax1.set_ylabel("Top-1 accuracy")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title("Random Gaussian noise")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    ax2 = plt.subplot(1, 2, 2)
    if not clean.empty:
        ax2.axhline(float(clean["accuracy_mean"].iloc[0]), color="#365f9c", linestyle="--", linewidth=1.4, label="clean")
    if not nonlinear.empty:
        ax2.plot(nonlinear["severity"], nonlinear["accuracy_mean"], marker="o", linewidth=2.0, color="#d85b3a", label="fixed alpha")
    if not random_rows.empty:
        ax2.axhline(float(random_rows["accuracy_mean"].iloc[0]), color="#9e1c21", linestyle=":", linewidth=1.6, label="random alpha")
    ax2.set_xlabel("Nonlinear alpha")
    ax2.set_ylabel("Top-1 accuracy")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_title("Input-dependent cubic distortion")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "mechanism_gaussian_vs_nonlinear_accuracy.png", dpi=240)
    plt.close()

    if drift.empty:
        return
    plt.figure(figsize=(7.8, 4.8))
    colors = {"gaussian": "#2f8c5a", "nonlinear_fixed_alpha": "#d85b3a", "nonlinear_random_alpha": "#9e1c21"}
    for _, row in drift.iterrows():
        plt.scatter(row["relative_l2"], row["argmax_flip_ratio"], s=90, color=colors.get(str(row["mode"]), "#394150"), alpha=0.8)
        plt.text(row["relative_l2"], row["argmax_flip_ratio"] + 0.02, str(row["label"]), fontsize=8, ha="center")
    plt.xlabel("Logit relative L2 drift")
    plt.ylabel("Argmax flip ratio")
    plt.ylim(0.0, min(1.0, max(drift["argmax_flip_ratio"].max() + 0.12, 0.2)))
    plt.title("Different perturbations create different output drift")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "mechanism_logit_drift.png", dpi=240)
    plt.close()


def plot_quantization_results(summary: pd.DataFrame, decomp: pd.DataFrame, fig_dir: Path) -> None:
    if summary.empty:
        return
    plt.figure(figsize=(8.2, 4.8))
    method_specs = [
        ("uniform_quant", "uniform quant only", "#365f9c"),
        ("uniform_quant_plus_random_nonlinear", "uniform quant + nonlinear", "#d85b3a"),
        ("fpc_reference", "FPC reference", "#2f8c5a"),
    ]
    for method, label, color in method_specs:
        sub = summary[summary["method"] == method].sort_values("bits")
        if sub.empty:
            continue
        plt.plot(sub["bits"], sub["accuracy_mean"], marker="o", linewidth=2.2, label=label, color=color)
    clean = summary[summary["method"] == "clean"]
    random_rows = summary[summary["method"] == "nonlinear_random_alpha"]
    if not clean.empty:
        plt.axhline(float(clean["accuracy_mean"].iloc[0]), color="#394150", linestyle="--", linewidth=1.3, label="clean")
    if not random_rows.empty:
        plt.axhline(float(random_rows["accuracy_mean"].iloc[0]), color="#8c939e", linestyle=":", linewidth=1.5, label="random nonlinear")
    plt.xlabel("Bits")
    plt.ylabel("Top-1 accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("Quantization error and nonlinear error are not interchangeable")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "quantization_nonlinearity_accuracy.png", dpi=240)
    plt.close()

    if decomp.empty:
        return
    plt.figure(figsize=(8.2, 4.8))
    x = list(range(len(decomp)))
    plt.bar(x, decomp["quantization_drop"], color="#365f9c", label="quantization drop")
    plt.bar(
        x,
        decomp["nonlinear_extra_drop_after_quantization"],
        bottom=decomp["quantization_drop"],
        color="#d85b3a",
        label="extra nonlinear drop",
    )
    plt.xticks(x, decomp["bits"])
    plt.xlabel("Bits")
    plt.ylabel("Accuracy drop")
    plt.ylim(0.0, min(1.0, max(decomp["total_drop"].max() + 0.1, 0.2)))
    plt.title("Drop decomposition: quantization first, nonlinear distortion second")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "quantization_error_decomposition.png", dpi=240)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ordered extension studies for CIM nonlinear robustness.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "extension_research")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "task2" / "checkpoints" / "cifar" / "cifar_resnet20_clean_best.pt")
    parser.add_argument("--model", type=str, default="cifar10_resnet20")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--test-subset", type=int, default=4096, help="0 means full CIFAR-10 test set.")
    parser.add_argument("--scope", choices=["per_tensor", "per_sample", "per_channel"], default="per_tensor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--drift-batches", type=int, default=8)
    parser.add_argument(
        "--structure-models",
        type=str,
        default="cifar10_resnet20,cifar10_resnet56,cifar10_mobilenetv2_x0_5,cifar10_mobilenetv2_x1_0,cifar10_shufflenetv2_x1_0,cifar10_vgg11_bn",
    )
    parser.add_argument("--structure-alphas", type=str, default="-1,-0.5,-0.2,-0.1,0,0.1,0.2,0.5,1")
    parser.add_argument("--skip-structure-sweep", action="store_true")
    parser.add_argument("--gaussian-sigmas", type=str, default="0.01,0.02,0.05,0.1,0.2,0.3")
    parser.add_argument("--nonlinear-alphas", type=str, default="-1,-0.2,-0.1,0.1,0.2,1")
    parser.add_argument("--quant-bits", type=str, default="1,2,3,4,5,6,8")
    parser.add_argument("--no-fpc-reference", action="store_true", help="Skip the FPC reference curve in the quantization study.")
    args = parser.parse_args()
    args.include_fpc_reference = not args.no_fpc_reference

    set_seed(args.seed)
    args.output_dir = ensure_dir(args.output_dir)
    ensure_dir(args.output_dir / "tables")
    ensure_dir(args.output_dir / "figures")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = {
        "device": str(device),
        "data_root": str(args.data_root),
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "test_subset": None if args.test_subset <= 0 else args.test_subset,
        "scope": args.scope,
        "seed": args.seed,
        "random_repeats": args.random_repeats,
        "ordered_studies": [
            "network_structure_and_parameter_count",
            "gaussian_noise_vs_nonlinear_distortion",
            "quantization_error_plus_nonlinear_error",
        ],
    }
    save_json(metadata, args.output_dir / "tables" / "extension_research_metadata.json")

    print("[1/3] Network structure and parameter-count study")
    run_structure_sweep(args, device)
    print("[2/3] Gaussian noise vs nonlinear distortion study")
    run_mechanism_comparison(args, device)
    print("[3/3] Quantization plus nonlinear error study")
    run_quantization_nonlinearity(args, device)
    print(f"Extension research outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
