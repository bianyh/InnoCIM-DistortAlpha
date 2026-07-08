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
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.metrics import compare_tensors, corpus_error_rate, count_parameters
from src.nonlinear import (
    ActivationRecorder,
    NonlinearityConfig,
    NonlinearityInjector,
    list_target_layers,
)
from src.plotting import plot_layer_metric_lines, plot_wer_alpha


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def load_examples(args) -> list[dict]:
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        streaming=args.streaming,
        trust_remote_code=True,
    )
    dataset = dataset.cast_column(args.audio_column, Audio(sampling_rate=args.sampling_rate))
    if not args.streaming and args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    examples = []
    for example in dataset:
        text = str(example.get(args.text_column, "")).strip()
        if not text:
            continue
        examples.append({args.audio_column: example[args.audio_column], args.text_column: text})
        if len(examples) >= args.max_samples:
            break
    return examples


def configure_generation(processor, model, language: str, task: str) -> None:
    forced_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    model.generation_config.forced_decoder_ids = forced_ids
    model.generation_config.suppress_tokens = []


@torch.inference_mode()
def transcribe_whisper(
    model,
    processor,
    examples: list[dict],
    device: torch.device,
    alpha: float,
    scope: str,
    args,
) -> tuple[list[str], list[str]]:
    refs: list[str] = []
    hyps: list[str] = []
    config = NonlinearityConfig(alpha=alpha, scope=scope)
    with NonlinearityInjector(model, config):
        for example in tqdm(examples, desc=f"alpha={alpha:g}"):
            audio = example[args.audio_column]
            inputs = processor(
                audio["array"],
                sampling_rate=args.sampling_rate,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(device)
            predicted_ids = model.generate(
                input_features,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            refs.append(example[args.text_column])
            hyps.append(text)
    return refs, hyps


@torch.inference_mode()
def collect_drift(model, processor, example: dict, device: torch.device, alpha: float, scope: str, args) -> pd.DataFrame:
    target_layers = list_target_layers(model)[: args.analysis_max_layers]
    layer_names = [name for name, _ in target_layers]
    audio = example[args.audio_column]
    inputs = processor(audio["array"], sampling_rate=args.sampling_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with ActivationRecorder(model, layer_names, keep_batches=1) as clean_rec:
        model.generate(input_features, max_new_tokens=args.max_new_tokens, do_sample=False)
    clean_outputs = clean_rec.stacked()

    config = NonlinearityConfig(alpha=alpha, scope=scope)
    with NonlinearityInjector(model, config), ActivationRecorder(model, layer_names, keep_batches=1) as nl_rec:
        model.generate(input_features, max_new_tokens=args.max_new_tokens, do_sample=False)
    nl_outputs = nl_rec.stacked()

    rows = []
    for layer_index, layer_name in enumerate(layer_names):
        if layer_name not in clean_outputs or layer_name not in nl_outputs:
            continue
        rows.append(
            {
                "layer": layer_name,
                "layer_index": layer_index,
                "alpha": alpha,
                **compare_tensors(clean_outputs[layer_name], nl_outputs[layer_name]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1 VoxForge sensitivity analysis with Whisper")
    parser.add_argument("--dataset", type=str, default="ciempiess/voxforge_spanish")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model", type=str, default="openai/whisper-tiny")
    parser.add_argument("--language", type=str, default="spanish")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--audio-column", type=str, default="audio")
    parser.add_argument("--text-column", type=str, default="normalized_text")
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--alphas", type=str, default="-1.0,-0.5,0,0.5,1.0")
    parser.add_argument("--scope", choices=["per_tensor", "per_sample", "per_channel"], default="per_tensor")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "task1")
    parser.add_argument("--skip-activation-analysis", action="store_true")
    parser.add_argument("--analysis-max-layers", type=int, default=32)
    parser.add_argument("--drift-alphas", type=str, default="-1.0,0.5,1.0")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args)
    if not examples:
        raise RuntimeError("No VoxForge examples loaded.")

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device).eval()
    configure_generation(processor, model, args.language, args.task)

    target_layers = list_target_layers(model)
    rows = []
    for alpha in parse_float_list(args.alphas):
        refs, hyps = transcribe_whisper(model, processor, examples, device, alpha, args.scope, args)
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
        pd.DataFrame({"reference": refs, "hypothesis": hyps}).to_csv(
            tables_dir / f"voxforge_whisper_predictions_alpha_{alpha:g}.csv",
            index=False,
        )

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "voxforge_whisper_wer_alpha.csv", index=False)
    plot_wer_alpha(df, figures_dir / "voxforge_whisper_wer_alpha.png")

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "device": str(device),
        "max_samples": args.max_samples,
        "alphas": parse_float_list(args.alphas),
        "scope": args.scope,
        "language": args.language,
    }
    (tables_dir / "voxforge_whisper_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not args.skip_activation_analysis:
        drift_frames = []
        for alpha in parse_float_list(args.drift_alphas):
            drift = collect_drift(model, processor, examples[0], device, alpha, args.scope, args)
            drift["model"] = args.model
            drift_frames.append(drift)
        drift_df = pd.concat(drift_frames, ignore_index=True)
        drift_df.to_csv(tables_dir / "voxforge_whisper_activation_drift.csv", index=False)
        plot_layer_metric_lines(
            drift_df,
            figures_dir / "voxforge_whisper_activation_relative_l2.png",
            value_col="relative_l2",
            title="VoxForge Whisper: layer-wise accumulated relative L2 drift",
        )
        plot_layer_metric_lines(
            drift_df,
            figures_dir / "voxforge_whisper_activation_cosine_drift.png",
            value_col="cosine_drift",
            title="VoxForge Whisper: layer-wise accumulated cosine drift",
        )

    print("Task 1 VoxForge Whisper outputs written to:")
    print(f"  tables:  {tables_dir}")
    print(f"  figures: {figures_dir}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
