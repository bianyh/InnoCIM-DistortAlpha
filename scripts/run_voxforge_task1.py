from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from datasets import Audio, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor

from src.metrics import corpus_error_rate, count_parameters
from src.nonlinear import (
    ActivationRecorder,
    NonlinearityConfig,
    NonlinearityInjector,
    list_target_layers,
)
from src.plotting import plot_layer_metric_lines, plot_wer_alpha


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def take_examples(dataset, max_samples: int, audio_column: str, text_column: str) -> list[dict]:
    examples = []
    for example in dataset:
        if audio_column not in example or text_column not in example:
            continue
        if example[text_column] is None or str(example[text_column]).strip() == "":
            continue
        examples.append({audio_column: example[audio_column], text_column: str(example[text_column])})
        if len(examples) >= max_samples:
            break
    return examples


def load_voxforge_examples(args) -> list[dict]:
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        streaming=args.streaming,
        trust_remote_code=True,
    )
    if hasattr(dataset, "cast_column"):
        dataset = dataset.cast_column(args.audio_column, Audio(sampling_rate=args.sampling_rate))
    if not args.streaming and args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    return take_examples(dataset, args.max_samples, args.audio_column, args.text_column)


def prepare_batch(examples: list[dict], processor, audio_column: str, text_column: str, sampling_rate: int):
    arrays = [ex[audio_column]["array"] for ex in examples]
    refs = [ex[text_column] for ex in examples]
    inputs = processor(
        arrays,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    return inputs, refs


@torch.inference_mode()
def transcribe(
    model,
    processor,
    examples: list[dict],
    device: torch.device,
    alpha: float,
    scope: str,
    audio_column: str,
    text_column: str,
    sampling_rate: int,
    batch_size: int,
) -> tuple[list[str], list[str]]:
    references: list[str] = []
    hypotheses: list[str] = []
    config = NonlinearityConfig(alpha=alpha, scope=scope)
    with NonlinearityInjector(model, config):
        for start in tqdm(range(0, len(examples), batch_size), desc=f"alpha={alpha:g}"):
            batch_examples = examples[start : start + batch_size]
            inputs, refs = prepare_batch(batch_examples, processor, audio_column, text_column, sampling_rate)
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            logits = model(input_values=input_values, attention_mask=attention_mask).logits
            pred_ids = torch.argmax(logits, dim=-1)
            hyps = processor.batch_decode(pred_ids)
            references.extend(refs)
            hypotheses.extend(hyps)
    return references, hypotheses


@torch.inference_mode()
def collect_asr_activation_drift(
    model,
    processor,
    examples: list[dict],
    device: torch.device,
    alpha: float,
    scope: str,
    audio_column: str,
    text_column: str,
    sampling_rate: int,
    layer_names: list[str],
) -> pd.DataFrame:
    batch_examples = examples[:1]
    inputs, _ = prepare_batch(batch_examples, processor, audio_column, text_column, sampling_rate)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with ActivationRecorder(model, layer_names, keep_batches=1) as clean_rec:
        model(input_values=input_values, attention_mask=attention_mask)
    clean_outputs = clean_rec.stacked()

    config = NonlinearityConfig(alpha=alpha, scope=scope)
    with NonlinearityInjector(model, config), ActivationRecorder(model, layer_names, keep_batches=1) as nl_rec:
        model(input_values=input_values, attention_mask=attention_mask)
    nl_outputs = nl_rec.stacked()

    from src.metrics import compare_tensors

    rows = []
    for layer_index, name in enumerate(layer_names):
        if name not in clean_outputs or name not in nl_outputs:
            continue
        rows.append(
            {
                "layer": name,
                "layer_index": layer_index,
                "alpha": alpha,
                **compare_tensors(clean_outputs[name], nl_outputs[name]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1 VoxForge ASR nonlinearity sensitivity analysis")
    parser.add_argument("--dataset", type=str, default="ciempiess/voxforge_spanish")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model", type=str, default="jonatasgrosman/wav2vec2-large-xlsr-53-spanish")
    parser.add_argument("--audio-column", type=str, default="audio")
    parser.add_argument("--text-column", type=str, default="normalized_text")
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--alphas", type=str, default="-1.0,-0.5,0,0.5,1.0")
    parser.add_argument("--scope", choices=["per_tensor", "per_sample", "per_channel"], default="per_tensor")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "task1")
    parser.add_argument("--skip-activation-analysis", action="store_true")
    parser.add_argument("--analysis-max-layers", type=int, default=24)
    parser.add_argument("--drift-alphas", type=str, default="-1.0,0.5,1.0")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    examples = load_voxforge_examples(args)
    if not examples:
        raise RuntimeError("No VoxForge examples were loaded. Check dataset/split/audio/text columns.")

    try:
        processor = AutoProcessor.from_pretrained(args.model)
    except ImportError as exc:
        if "pyctcdecode" not in str(exc):
            raise
        processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = AutoModelForCTC.from_pretrained(args.model).to(device).eval()
    target_layers = list_target_layers(model)

    rows = []
    alphas = parse_float_list(args.alphas)
    for alpha in alphas:
        refs, hyps = transcribe(
            model=model,
            processor=processor,
            examples=examples,
            device=device,
            alpha=alpha,
            scope=args.scope,
            audio_column=args.audio_column,
            text_column=args.text_column,
            sampling_rate=args.sampling_rate,
            batch_size=args.batch_size,
        )
        wer = corpus_error_rate(refs, hyps, unit="word")
        cer = corpus_error_rate(refs, hyps, unit="char")
        rows.append(
            {
                "dataset": args.dataset,
                "split": args.split,
                "model": args.model,
                "alpha": alpha,
                "wer": wer,
                "cer": cer,
                "num_samples": len(refs),
                "params": count_parameters(model),
                "target_layers": len(target_layers),
                "scope": args.scope,
            }
        )
        pred_path = tables_dir / f"voxforge_predictions_alpha_{alpha:g}.csv"
        pd.DataFrame({"reference": refs, "hypothesis": hyps}).to_csv(pred_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "voxforge_wer_alpha.csv", index=False)
    plot_wer_alpha(df, figures_dir / "voxforge_wer_alpha.png")

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "device": str(device),
        "max_samples": args.max_samples,
        "alphas": alphas,
        "scope": args.scope,
    }
    (tables_dir / "voxforge_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not args.skip_activation_analysis:
        layer_names = [name for name, _ in target_layers[: args.analysis_max_layers]]
        drift_frames = []
        for alpha in parse_float_list(args.drift_alphas):
            drift = collect_asr_activation_drift(
                model=model,
                processor=processor,
                examples=examples,
                device=device,
                alpha=alpha,
                scope=args.scope,
                audio_column=args.audio_column,
                text_column=args.text_column,
                sampling_rate=args.sampling_rate,
                layer_names=layer_names,
            )
            drift["model"] = args.model
            drift_frames.append(drift)
        drift_df = pd.concat(drift_frames, ignore_index=True)
        drift_df.to_csv(tables_dir / "voxforge_activation_drift.csv", index=False)
        plot_layer_metric_lines(
            drift_df,
            figures_dir / "voxforge_activation_relative_l2.png",
            value_col="relative_l2",
            title="VoxForge ASR: layer-wise accumulated relative L2 drift",
        )
        plot_layer_metric_lines(
            drift_df,
            figures_dir / "voxforge_activation_cosine_drift.png",
            value_col="cosine_drift",
            title="VoxForge ASR: layer-wise accumulated cosine drift",
        )

    print("Task 1 VoxForge outputs written to:")
    print(f"  tables:  {tables_dir}")
    print(f"  figures: {figures_dir}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
