from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from src.metrics import compare_tensors, count_parameters, top1_accuracy
from src.nonlinear import (
    ActivationRecorder,
    NonlinearityConfig,
    NonlinearityInjector,
    list_target_layers,
)
from src.plotting import (
    plot_accuracy_alpha,
    plot_accuracy_drop,
    plot_activation_histograms,
    plot_layer_heatmap,
    plot_layer_metric_lines,
    plot_nonlinearity_curves,
)


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def build_loader(
    data_root: Path,
    batch_size: int,
    workers: int,
    max_samples: int | None,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    dataset = datasets.CIFAR10(root=str(data_root), train=False, download=False, transform=transform)
    if max_samples is not None and max_samples < len(dataset):
        dataset = Subset(dataset, list(range(max_samples)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_cifar_model(model_name: str, device: torch.device) -> torch.nn.Module:
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        model_name,
        pretrained=True,
        trust_repo=True,
    )
    model.eval()
    return model.to(device)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    alpha: float,
    scope: str,
    enabled_layers: list[str] | None = None,
) -> dict[str, float]:
    correct = 0
    total = 0
    loss_sum = 0.0
    config = NonlinearityConfig(alpha=alpha, scope=scope)
    with NonlinearityInjector(model, config, enabled_layers=enabled_layers):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            batch_correct, batch_total = top1_accuracy(logits, labels)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            correct += batch_correct
            total += batch_total
            loss_sum += float(loss.item())
    return {
        "accuracy": correct / max(total, 1),
        "loss": loss_sum / max(total, 1),
        "correct": correct,
        "total": total,
    }


@torch.inference_mode()
def collect_activation_drift(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    alpha: float,
    scope: str,
    layer_names: list[str],
    keep_batches: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_recorder = ActivationRecorder(model, layer_names, keep_batches=keep_batches)
    with clean_recorder:
        for i, (images, _) in enumerate(loader):
            if i >= keep_batches:
                break
            model(images.to(device, non_blocking=True))
    clean_outputs = clean_recorder.stacked()

    nonlinear_recorder = ActivationRecorder(model, layer_names, keep_batches=keep_batches)
    config = NonlinearityConfig(alpha=alpha, scope=scope)
    with NonlinearityInjector(model, config), nonlinear_recorder:
        for i, (images, _) in enumerate(loader):
            if i >= keep_batches:
                break
            model(images.to(device, non_blocking=True))
    nonlinear_outputs = nonlinear_recorder.stacked()

    rows: list[dict[str, float | int | str]] = []
    hist_rows: list[dict[str, float | str]] = []
    for layer_index, name in enumerate(layer_names):
        if name not in clean_outputs or name not in nonlinear_outputs:
            continue
        stats = compare_tensors(clean_outputs[name], nonlinear_outputs[name])
        rows.append({"layer": name, "layer_index": layer_index, "alpha": alpha, **stats})

        if layer_index in {0, len(layer_names) // 2, len(layer_names) - 1}:
            for run_name, tensor in [("clean", clean_outputs[name]), (f"alpha={alpha:g}", nonlinear_outputs[name])]:
                flat = tensor.flatten().float()
                if flat.numel() > 8000:
                    idx = torch.linspace(0, flat.numel() - 1, steps=8000).long()
                    flat = flat[idx]
                for value in flat.tolist():
                    hist_rows.append({"layer": name, "run": run_name, "value": float(value), "alpha": alpha})
    return pd.DataFrame(rows), pd.DataFrame(hist_rows)


def run_accuracy_sweep(
    models: list[str],
    alphas: list[float],
    loader: DataLoader,
    device: torch.device,
    scope: str,
    tables_dir: Path,
) -> pd.DataFrame:
    rows = []
    out_path = tables_dir / "cifar_accuracy_alpha.csv"
    for model_name in models:
        model = load_cifar_model(model_name, device)
        param_count = count_parameters(model)
        target_layers = list_target_layers(model)
        for alpha in tqdm(alphas, desc=f"{model_name} alpha sweep"):
            result = evaluate(model, loader, device, alpha=alpha, scope=scope)
            rows.append(
                {
                    "dataset": "cifar10",
                    "model": model_name,
                    "alpha": alpha,
                    "accuracy": result["accuracy"],
                    "loss": result["loss"],
                    "correct": result["correct"],
                    "total": result["total"],
                    "params": param_count,
                    "target_layers": len(target_layers),
                    "scope": scope,
                }
            )
            pd.DataFrame(rows).to_csv(out_path, index=False)
        del model
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def run_layer_sensitivity(
    model_name: str,
    alphas: list[float],
    loader: DataLoader,
    device: torch.device,
    scope: str,
    max_layers: int,
    tables_dir: Path,
) -> pd.DataFrame:
    model = load_cifar_model(model_name, device)
    target_layers = list_target_layers(model)
    if max_layers > 0:
        target_layers = target_layers[:max_layers]
    layer_names = [name for name, _ in target_layers]
    clean = evaluate(model, loader, device, alpha=0.0, scope=scope)

    rows = []
    for layer_index, layer_name in enumerate(tqdm(layer_names, desc=f"{model_name} layer sensitivity")):
        for alpha in alphas:
            result = evaluate(model, loader, device, alpha=alpha, scope=scope, enabled_layers=[layer_name])
            rows.append(
                {
                    "dataset": "cifar10",
                    "model": model_name,
                    "layer": layer_name,
                    "layer_index": layer_index,
                    "alpha": alpha,
                    "accuracy": result["accuracy"],
                    "accuracy_drop": clean["accuracy"] - result["accuracy"],
                    "clean_accuracy": clean["accuracy"],
                    "loss": result["loss"],
                    "total": result["total"],
                    "scope": scope,
                }
            )
    df = pd.DataFrame(rows)
    path = tables_dir / f"{model_name}_layer_sensitivity.csv"
    df.to_csv(path, index=False)
    del model
    torch.cuda.empty_cache()
    return df


def run_activation_analysis(
    model_name: str,
    alphas: list[float],
    loader: DataLoader,
    device: torch.device,
    scope: str,
    max_layers: int,
    keep_batches: int,
    tables_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    model = load_cifar_model(model_name, device)
    target_layers = list_target_layers(model)
    if max_layers > 0:
        target_layers = target_layers[:max_layers]
    layer_names = [name for name, _ in target_layers]

    drift_frames = []
    hist_frames = []
    for alpha in tqdm(alphas, desc=f"{model_name} activation drift"):
        drift_df, hist_df = collect_activation_drift(
            model=model,
            loader=loader,
            device=device,
            alpha=alpha,
            scope=scope,
            layer_names=layer_names,
            keep_batches=keep_batches,
        )
        drift_df["model"] = model_name
        hist_df["model"] = model_name
        drift_frames.append(drift_df)
        hist_frames.append(hist_df)

    drift_all = pd.concat(drift_frames, ignore_index=True) if drift_frames else pd.DataFrame()
    hist_all = pd.concat(hist_frames, ignore_index=True) if hist_frames else pd.DataFrame()
    drift_all.to_csv(tables_dir / f"{model_name}_activation_drift.csv", index=False)
    hist_all.to_csv(tables_dir / f"{model_name}_activation_hist_samples.csv", index=False)

    if not hist_all.empty:
        for (layer, alpha), group in hist_all.groupby(["layer", "alpha"]):
            safe_layer = layer.replace(".", "_").replace("/", "_")
            plot_activation_histograms(
                group,
                figures_dir / f"{model_name}_activation_hist_{safe_layer}_alpha_{alpha:g}.png",
                title=f"{model_name}: activation distribution at {layer}, alpha={alpha:g}",
            )

    del model
    torch.cuda.empty_cache()
    return drift_all


def summarize_accuracy(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    rows = []
    for model_name, group in df.groupby("model"):
        clean_candidates = group[group["alpha"] == 0.0]
        clean_acc = float(clean_candidates.iloc[0]["accuracy"]) if not clean_candidates.empty else float(group["accuracy"].max())
        rows.append(
            {
                "model": model_name,
                "clean_accuracy": clean_acc,
                "mean_accuracy": float(group["accuracy"].mean()),
                "worst_accuracy": float(group["accuracy"].min()),
                "max_accuracy_drop": float(clean_acc - group["accuracy"].min()),
                "robust_auc_discrete": float(group.sort_values("alpha")["accuracy"].mean()),
                "params": int(group["params"].iloc[0]),
                "target_layers": int(group["target_layers"].iloc[0]),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(tables_dir / "cifar_accuracy_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1 CIFAR-10 nonlinearity sensitivity analysis")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "task1")
    parser.add_argument(
        "--models",
        type=str,
        default="cifar10_resnet20,cifar10_vgg16_bn,cifar10_mobilenetv2_x1_0",
        help="Comma-separated torch.hub model names from chenyaofo/pytorch-cifar-models.",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="-1.0,-0.8,-0.6,-0.4,-0.2,-0.1,0,0.1,0.2,0.4,0.6,0.8,1.0",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--scope", choices=["per_tensor", "per_sample", "per_channel"], default="per_tensor")
    parser.add_argument("--skip-layer-sensitivity", action="store_true")
    parser.add_argument("--skip-activation-analysis", action="store_true")
    parser.add_argument("--analysis-model", type=str, default="cifar10_resnet20")
    parser.add_argument("--analysis-max-layers", type=int, default=28)
    parser.add_argument("--analysis-samples", type=int, default=2048)
    parser.add_argument("--analysis-batch-size", type=int, default=256)
    parser.add_argument("--sensitivity-alphas", type=str, default="-1.0,-0.5,0.5,1.0")
    parser.add_argument("--drift-alphas", type=str, default="-1.0,-0.5,0.5,1.0")
    parser.add_argument("--activation-keep-batches", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    models = parse_str_list(args.models)
    alphas = parse_float_list(args.alphas)
    sensitivity_alphas = parse_float_list(args.sensitivity_alphas)
    drift_alphas = parse_float_list(args.drift_alphas)

    loader = build_loader(args.data_root, args.batch_size, args.workers, args.max_samples)
    accuracy_df = run_accuracy_sweep(models, alphas, loader, device, args.scope, tables_dir)
    summary_df = summarize_accuracy(accuracy_df, tables_dir)

    plot_nonlinearity_curves(figures_dir / "nonlinearity_curves.png")
    plot_accuracy_alpha(accuracy_df, figures_dir / "cifar_accuracy_alpha.png")
    plot_accuracy_drop(accuracy_df, figures_dir / "cifar_accuracy_drop.png")

    metadata = {
        "device": str(device),
        "models": models,
        "alphas": alphas,
        "scope": args.scope,
        "max_samples": args.max_samples,
        "analysis_model": args.analysis_model,
    }
    (tables_dir / "cifar_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    analysis_loader = build_loader(
        args.data_root,
        args.analysis_batch_size,
        args.workers,
        args.analysis_samples,
    )

    if not args.skip_layer_sensitivity:
        sens_df = run_layer_sensitivity(
            model_name=args.analysis_model,
            alphas=sensitivity_alphas,
            loader=analysis_loader,
            device=device,
            scope=args.scope,
            max_layers=args.analysis_max_layers,
            tables_dir=tables_dir,
        )
        plot_layer_heatmap(
            sens_df,
            figures_dir / f"{args.analysis_model}_layer_sensitivity_heatmap.png",
            value_col="accuracy_drop",
            title=f"{args.analysis_model}: single-layer nonlinearity sensitivity",
        )

    if not args.skip_activation_analysis:
        drift_df = run_activation_analysis(
            model_name=args.analysis_model,
            alphas=drift_alphas,
            loader=analysis_loader,
            device=device,
            scope=args.scope,
            max_layers=args.analysis_max_layers,
            keep_batches=args.activation_keep_batches,
            tables_dir=tables_dir,
            figures_dir=figures_dir,
        )
        if not drift_df.empty:
            plot_layer_metric_lines(
                drift_df,
                figures_dir / f"{args.analysis_model}_activation_relative_l2.png",
                value_col="relative_l2",
                title=f"{args.analysis_model}: layer-wise accumulated relative L2 drift",
            )
            plot_layer_metric_lines(
                drift_df,
                figures_dir / f"{args.analysis_model}_activation_cosine_drift.png",
                value_col="cosine_drift",
                title=f"{args.analysis_model}: layer-wise accumulated cosine drift",
            )

    print("Task 1 CIFAR-10 outputs written to:")
    print(f"  tables:  {tables_dir}")
    print(f"  figures: {figures_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
