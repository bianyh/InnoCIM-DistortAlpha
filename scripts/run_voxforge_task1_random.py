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

from src.asr_models import build_asr_model
from src.metrics import compare_tensors, corpus_error_rate, count_parameters
from src.nonlinear import ActivationRecorder, RandomAlphaNonlinearityInjector, list_target_layers
from src.training import ensure_dir, load_checkpoint, save_json, set_seed
from src.voxforge_data import DEFAULT_CONFIG_NAME, DEFAULT_DATASET_NAME, build_voxforge_loaders


def decode_logits(logits: torch.Tensor, vocab) -> list[str]:
    return vocab.decode_batch(logits.argmax(dim=-1))


@torch.inference_mode()
def evaluate_clean(model: torch.nn.Module, loader, device: torch.device, vocab) -> dict[str, float]:
    model.eval()
    references: list[str] = []
    hypotheses: list[str] = []
    for batch in tqdm(loader, desc="task1 vox clean eval", leave=False):
        waveforms = batch["waveforms"].to(device, non_blocking=True)
        lengths = batch["waveform_lengths"].to(device, non_blocking=True)
        logits, _ = model(waveforms, lengths)
        references.extend(batch["texts"])
        hypotheses.extend(decode_logits(logits, vocab))
    return {
        "wer": corpus_error_rate(references, hypotheses, unit="word"),
        "cer": corpus_error_rate(references, hypotheses, unit="char"),
        "num_examples": len(references),
    }


@torch.inference_mode()
def evaluate_random(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    vocab,
    alpha_low: float,
    alpha_high: float,
    scope: str,
    enabled_layers: set[str] | None = None,
) -> dict[str, float]:
    model.eval()
    references: list[str] = []
    hypotheses: list[str] = []
    for batch in tqdm(loader, desc="task1 vox random-field eval", leave=False):
        waveforms = batch["waveforms"].to(device, non_blocking=True)
        lengths = batch["waveform_lengths"].to(device, non_blocking=True)
        with RandomAlphaNonlinearityInjector(
            model,
            alpha_low=alpha_low,
            alpha_high=alpha_high,
            scope=scope,
            enabled_layers=enabled_layers,
        ):
            logits, _ = model(waveforms, lengths)
        references.extend(batch["texts"])
        hypotheses.extend(decode_logits(logits, vocab))
    return {
        "wer": corpus_error_rate(references, hypotheses, unit="word"),
        "cer": corpus_error_rate(references, hypotheses, unit="char"),
        "num_examples": len(references),
    }


@torch.inference_mode()
def evaluate_random_mc(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    vocab,
    alpha_low: float,
    alpha_high: float,
    scope: str,
    mc: int,
) -> dict[str, float]:
    model.eval()
    references: list[str] = []
    hypotheses: list[str] = []
    mc = max(1, int(mc))
    for batch in tqdm(loader, desc=f"task1 vox MC-{mc}", leave=False):
        waveforms = batch["waveforms"].to(device, non_blocking=True)
        lengths = batch["waveform_lengths"].to(device, non_blocking=True)
        probs = []
        min_time = None
        for _ in range(mc):
            with RandomAlphaNonlinearityInjector(model, alpha_low=alpha_low, alpha_high=alpha_high, scope=scope):
                logits, _ = model(waveforms, lengths)
            probs.append(F.softmax(logits, dim=-1))
            min_time = logits.size(1) if min_time is None else min(min_time, logits.size(1))
        aligned = torch.stack([prob[:, :min_time, :] for prob in probs], dim=0)
        pred_ids = aligned.mean(dim=0).argmax(dim=-1)
        references.extend(batch["texts"])
        hypotheses.extend(vocab.decode_batch(pred_ids))
    return {
        "wer": corpus_error_rate(references, hypotheses, unit="word"),
        "cer": corpus_error_rate(references, hypotheses, unit="char"),
        "num_examples": len(references),
    }


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
        for step, batch in enumerate(tqdm(loader, desc=f"task1 vox drift repeat={repeat + 1}", leave=False)):
            if step >= max_batches:
                break
            waveforms = batch["waveforms"].to(device, non_blocking=True)
            lengths = batch["waveform_lengths"].to(device, non_blocking=True)
            with ActivationRecorder(model, layers, keep_batches=1, detach_cpu=True) as clean_rec:
                model(waveforms, lengths)
            with RandomAlphaNonlinearityInjector(model, alpha_low=alpha_low, alpha_high=alpha_high, scope=scope):
                with ActivationRecorder(model, layers, keep_batches=1, detach_cpu=True) as random_rec:
                    model(waveforms, lengths)
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
                        "dataset": "voxforge",
                        "model": "crnn_ctc",
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
    summary_path = table_dir / "voxforge_random_field_summary.csv"
    layer_path = table_dir / "voxforge_layer_random_sensitivity.csv"
    drift_path = table_dir / "voxforge_activation_drift_random.csv"

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        labels, values = [], []
        for _, row in summary.iterrows():
            if row["eval_type"] == "clean":
                label = "Clean"
            elif row["eval_type"] == "random_single":
                label = "Random Field"
            else:
                label = f"MC-{int(row['mc'])}"
            labels.append(label)
            values.append(float(row["cer"]))
        plt.figure(figsize=(7.2, 4.4))
        plt.bar(labels, values, color=["#4c78a8"] + ["#e45756"] * (len(values) - 1))
        plt.ylabel("CER")
        plt.title("VoxForge Per-Operator Random Alpha Sensitivity")
        plt.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(figure_dir / "voxforge_random_field_summary.png", dpi=220)
        plt.close()

    if layer_path.exists():
        layer = pd.read_csv(layer_path)
        grouped = layer[layer["repeat"] == 0].sort_values("cer_increase", ascending=False)
        plt.figure(figsize=(7.0, 4.0))
        plt.barh(grouped["layer"], grouped["cer_increase"], color="#b279a2")
        plt.gca().invert_yaxis()
        plt.xlabel("CER increase vs clean")
        plt.title("Layer-Only Random Alpha Sensitivity")
        plt.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.savefig(figure_dir / "voxforge_layer_random_sensitivity.png", dpi=220)
        plt.close()

    if drift_path.exists():
        drift = pd.read_csv(drift_path)
        grouped = (
            drift.groupby("layer", as_index=False)
            .agg(relative_l2=("relative_l2", "mean"), cosine_drift=("cosine_drift", "mean"))
            .sort_values("relative_l2", ascending=False)
        )
        plt.figure(figsize=(7.0, 4.0))
        plt.barh(grouped["layer"], grouped["relative_l2"], color="#54a24b")
        plt.gca().invert_yaxis()
        plt.xlabel("Mean relative L2 drift")
        plt.title("Random-Field Activation Drift")
        plt.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.savefig(figure_dir / "voxforge_random_activation_drift.png", dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1 VoxForge sensitivity under per-operator random alpha.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "task1_random_voxforge")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "task2" / "checkpoints" / "voxforge" / "voxforge_crnn_clean_best.pt")
    parser.add_argument("--model", type=str, default="crnn_ctc")
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-size", type=int, default=120)
    parser.add_argument("--val-size", type=int, default=30)
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--max-duration", type=float, default=5.0)
    parser.add_argument("--max-text-length", type=int, default=110)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--config-name", type=str, default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--scope", choices=["per_tensor", "per_sample", "per_channel"], default="per_tensor")
    parser.add_argument("--alpha-low", type=float, default=-1.0)
    parser.add_argument("--alpha-high", type=float, default=1.0)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--layer-repeats", type=int, default=2)
    parser.add_argument("--drift-repeats", type=int, default=2)
    parser.add_argument("--drift-batches", type=int, default=2)
    parser.add_argument("--mc-values", type=str, default="1,3,5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    table_dir = ensure_dir(output_dir / "tables")
    ensure_dir(output_dir / "figures")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, vocab, data_metadata = build_voxforge_loaders(
        data_root=args.data_root,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        max_duration=args.max_duration,
        max_text_length=args.max_text_length,
        batch_size=args.batch_size,
        workers=args.workers,
        dataset_name=args.dataset_name,
        config_name=args.config_name,
    )
    model = build_asr_model(
        args.model,
        vocab_size=len(vocab),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    target_layers = [name for name, _ in list_target_layers(model)]
    metadata = {
        "dataset": "voxforge",
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "params": count_parameters(model),
        "protocol": "per_operator_random_alpha",
        "alpha_low": args.alpha_low,
        "alpha_high": args.alpha_high,
        "target_layers": target_layers,
        "data": data_metadata,
    }
    save_json(metadata, table_dir / "voxforge_random_task1_metadata.json")

    clean = evaluate_clean(model, test_loader, device, vocab)
    summary_rows: list[dict[str, float | str | int]] = [
        {
            "dataset": "voxforge",
            "model": args.model,
            "eval_type": "clean",
            "wer": clean["wer"],
            "cer": clean["cer"],
            "mc": 0,
            "repeat_count": 1,
            "cer_std": 0.0,
            "worst_cer": clean["cer"],
            "num_examples": clean["num_examples"],
        }
    ]

    repeat_rows = []
    cers = []
    wers = []
    for repeat in range(args.random_repeats):
        result = evaluate_random(
            model,
            test_loader,
            device,
            vocab,
            alpha_low=args.alpha_low,
            alpha_high=args.alpha_high,
            scope=args.scope,
        )
        cers.append(float(result["cer"]))
        wers.append(float(result["wer"]))
        repeat_rows.append(
            {
                "dataset": "voxforge",
                "model": args.model,
                "protocol": "per_operator_random_alpha",
                "repeat": repeat + 1,
                "wer": result["wer"],
                "cer": result["cer"],
                "num_examples": result["num_examples"],
            }
        )
    pd.DataFrame(repeat_rows).to_csv(table_dir / "voxforge_random_field_repeats.csv", index=False)
    summary_rows.append(
        {
            "dataset": "voxforge",
            "model": args.model,
            "eval_type": "random_single",
            "wer": sum(wers) / max(len(wers), 1),
            "cer": sum(cers) / max(len(cers), 1),
            "mc": 1,
            "repeat_count": len(cers),
            "cer_std": statistics.pstdev(cers) if len(cers) > 1 else 0.0,
            "worst_cer": max(cers) if cers else math.nan,
            "num_examples": clean["num_examples"],
        }
    )

    for mc in [int(item.strip()) for item in args.mc_values.split(",") if item.strip()]:
        result = evaluate_random_mc(
            model,
            test_loader,
            device,
            vocab,
            alpha_low=args.alpha_low,
            alpha_high=args.alpha_high,
            scope=args.scope,
            mc=mc,
        )
        summary_rows.append(
            {
                "dataset": "voxforge",
                "model": args.model,
                "eval_type": "random_mc",
                "wer": result["wer"],
                "cer": result["cer"],
                "mc": mc,
                "repeat_count": 1,
                "cer_std": 0.0,
                "worst_cer": result["cer"],
                "num_examples": result["num_examples"],
            }
        )

    layer_rows = []
    for layer in target_layers:
        layer_cers = []
        layer_wers = []
        for repeat in range(args.layer_repeats):
            result = evaluate_random(
                model,
                test_loader,
                device,
                vocab,
                alpha_low=args.alpha_low,
                alpha_high=args.alpha_high,
                scope=args.scope,
                enabled_layers={layer},
            )
            layer_cers.append(float(result["cer"]))
            layer_wers.append(float(result["wer"]))
            layer_rows.append(
                {
                    "dataset": "voxforge",
                    "model": args.model,
                    "protocol": "single_layer_per_occurrence_random_alpha",
                    "layer": layer,
                    "repeat": repeat + 1,
                    "wer": result["wer"],
                    "cer": result["cer"],
                    "clean_cer": clean["cer"],
                    "cer_increase": result["cer"] - clean["cer"],
                }
            )
        layer_rows.append(
            {
                "dataset": "voxforge",
                "model": args.model,
                "protocol": "single_layer_per_occurrence_random_alpha_summary",
                "layer": layer,
                "repeat": 0,
                "wer": sum(layer_wers) / max(len(layer_wers), 1),
                "cer": sum(layer_cers) / max(len(layer_cers), 1),
                "clean_cer": clean["cer"],
                "cer_increase": sum(layer_cers) / max(len(layer_cers), 1) - clean["cer"],
            }
        )
    pd.DataFrame(layer_rows).to_csv(table_dir / "voxforge_layer_random_sensitivity.csv", index=False)

    drift_df = activation_drift_random(
        model=model,
        loader=test_loader,
        device=device,
        layers=target_layers,
        alpha_low=args.alpha_low,
        alpha_high=args.alpha_high,
        scope=args.scope,
        repeats=args.drift_repeats,
        max_batches=args.drift_batches,
    )
    drift_df.to_csv(table_dir / "voxforge_activation_drift_random.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(table_dir / "voxforge_random_field_summary.csv", index=False)
    make_figures(output_dir)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Tables: {table_dir}")
    print(f"Figures: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
