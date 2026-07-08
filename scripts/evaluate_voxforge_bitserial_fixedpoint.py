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
from tqdm.auto import tqdm

from src.asr_models import build_asr_model
from src.metrics import corpus_error_rate, count_parameters
from src.nonlinear import BitSerialFixedPointConfig, BitSerialFixedPointInjector, RandomAlphaNonlinearityInjector, list_target_layers
from src.training import ensure_dir, load_checkpoint, save_json, set_seed
from src.voxforge_data import DEFAULT_CONFIG_NAME, DEFAULT_DATASET_NAME, build_voxforge_loaders


def decode_logits(logits: torch.Tensor, vocab) -> list[str]:
    return vocab.decode_batch(logits.argmax(dim=-1))


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader, device: torch.device, vocab, mode: str, bits: int = 0, repeats: int = 1) -> dict[str, float]:
    wers: list[float] = []
    cers: list[float] = []
    for _ in range(max(1, repeats)):
        references: list[str] = []
        hypotheses: list[str] = []
        for batch in tqdm(loader, desc=f"Vox FPC eval {mode} B={bits}", leave=False):
            waveforms = batch["waveforms"].to(device, non_blocking=True)
            lengths = batch["waveform_lengths"].to(device, non_blocking=True)
            if mode == "clean":
                logits, _ = model(waveforms, lengths)
            elif mode == "random_single":
                with RandomAlphaNonlinearityInjector(model, alpha_low=-1.0, alpha_high=1.0):
                    logits, _ = model(waveforms, lengths)
            elif mode == "fpc":
                cfg = BitSerialFixedPointConfig(bits=bits, simulate_hardware_nonlinearity=True, endpoint_alpha=False)
                with BitSerialFixedPointInjector(model, cfg):
                    logits, _ = model(waveforms, lengths)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            references.extend(batch["texts"])
            hypotheses.extend(decode_logits(logits, vocab))
        wers.append(corpus_error_rate(references, hypotheses, unit="word"))
        cers.append(corpus_error_rate(references, hypotheses, unit="char"))
    return {
        "wer_mean": sum(wers) / len(wers),
        "wer_worst": max(wers),
        "wer_std": statistics.pstdev(wers) if len(wers) > 1 else 0.0,
        "cer_mean": sum(cers) / len(cers),
        "cer_worst": max(cers),
        "cer_std": statistics.pstdev(cers) if len(cers) > 1 else 0.0,
    }


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def make_figures(summary: pd.DataFrame, output_dir: Path) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    labels, values, colors = [], [], []
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
        values.append(float(row["cer_mean"]))
    plt.figure(figsize=(8.8, 4.8))
    plt.bar(labels, values, color=colors)
    plt.ylabel("CER")
    plt.title("VoxForge Distribution-Free Fixed-Point Bit-Serial Coding")
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(figure_dir / "voxforge_fpc_cer_summary.png", dpi=220)
    plt.close()

    fpc = summary[summary["method"] == "fpc"].sort_values("bits")
    plt.figure(figsize=(7.2, 4.4))
    plt.plot(fpc["bits"], fpc["cer_mean"], marker="o", color="#54a24b", label="FPC")
    clean = summary[summary["method"] == "clean"]
    random_single = summary[summary["method"] == "random_single"]
    if not clean.empty:
        plt.axhline(float(clean["cer_mean"].iloc[0]), color="#4c78a8", linestyle="--", linewidth=1.2, label="clean")
    if not random_single.empty:
        plt.axhline(float(random_single["cer_mean"].iloc[0]), color="#e45756", linestyle=":", linewidth=1.2, label="random field")
    plt.xlabel("Bit-serial fixed-point precision B")
    plt.ylabel("CER")
    plt.title("FPC CER vs Bit Precision")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "voxforge_fpc_bit_curve.png", dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed-point bit-serial coding on VoxForge.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "task3_fpc_voxforge")
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
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--bits", type=str, default="1,2,3,4,5,6")
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
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        max_duration=args.max_duration,
        max_text_length=args.max_text_length,
        dataset_name=args.dataset_name,
        config_name=args.config_name,
    )
    model = build_asr_model(args.model, vocab_size=len(vocab), hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    metadata = {
        "dataset": "voxforge",
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "method": "FPC",
        "params": count_parameters(model),
        "target_layers": len(list_target_layers(model)),
        "bits": parse_int_list(args.bits),
        "protocol": "per_operator_per_call_alpha_unknown_in_minus1_1",
        "alpha_assumption": "no distribution assumption; each bit-plane is invariant for any alpha",
        "fpc_hardware_alpha_sampling": "random_uniform_minus1_1_per_bit_plane",
        "data": data_metadata,
    }
    save_json(metadata, table_dir / "voxforge_fpc_metadata.json")

    rows: list[dict[str, float | int | str]] = []
    for method, bits, repeats in [("clean", 0, 1), ("random_single", 0, args.random_repeats)]:
        result = evaluate(model, test_loader, device, vocab, method, bits=bits, repeats=repeats)
        rows.append({"dataset": "voxforge", "model": args.model, "method": method, "bits": bits, "repeats": repeats, **result})
    for bits in parse_int_list(args.bits):
        result = evaluate(model, test_loader, device, vocab, "fpc", bits=bits, repeats=1)
        rows.append({"dataset": "voxforge", "model": args.model, "method": "fpc", "bits": bits, "repeats": 1, **result})
    summary = pd.DataFrame(rows)
    summary.to_csv(table_dir / "voxforge_fpc_summary.csv", index=False)
    make_figures(summary, output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
