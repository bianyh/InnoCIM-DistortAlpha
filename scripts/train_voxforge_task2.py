from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.alpha_scheduler import build_alpha_scheduler
from src.asr_models import build_asr_model
from src.metrics import corpus_error_rate, count_parameters
from src.nonlinear import NonlinearityConfig, NonlinearityInjector
from src.training import AverageMeter, append_csv, ensure_dir, load_checkpoint, save_checkpoint, save_json, set_seed
from src.voxforge_data import DEFAULT_CONFIG_NAME, DEFAULT_DATASET_NAME, build_voxforge_loaders


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def ctc_loss(
    logits: torch.Tensor,
    output_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    return F.ctc_loss(
        log_probs,
        targets,
        output_lengths.detach().cpu(),
        target_lengths.detach().cpu(),
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )


def forward_with_alpha(
    model: torch.nn.Module,
    waveforms: torch.Tensor,
    waveform_lengths: torch.Tensor,
    alpha: float,
    scope: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if abs(alpha) < 1e-12:
        return model(waveforms, waveform_lengths)
    with NonlinearityInjector(model, NonlinearityConfig(alpha=alpha, scope=scope)):
        return model(waveforms, waveform_lengths)


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    alpha_scheduler,
    scope: str,
    blank_id: int,
    amp: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    alpha_abs_meter = AverageMeter()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    for step, batch in enumerate(tqdm(loader, desc=f"vox train epoch {epoch + 1}", leave=False)):
        if max_batches is not None and step >= max_batches:
            break
        waveforms = batch["waveforms"].to(device, non_blocking=True)
        waveform_lengths = batch["waveform_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        alpha = alpha_scheduler.sample(epoch, step)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits, output_lengths = forward_with_alpha(model, waveforms, waveform_lengths, alpha=alpha, scope=scope)
            loss = ctc_loss(logits, output_lengths, targets, target_lengths, blank_id=blank_id)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(float(loss.item()), waveforms.size(0))
        alpha_abs_meter.update(abs(alpha), 1)

    return {
        "train_loss": loss_meter.avg,
        "train_alpha_abs_mean": alpha_abs_meter.avg,
    }


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    alpha: float,
    scope: str,
    vocab,
    max_batches: int | None = None,
    collect_examples: int = 0,
) -> dict[str, object]:
    model.eval()
    loss_meter = AverageMeter()
    references: list[str] = []
    hypotheses: list[str] = []
    examples: list[dict[str, object]] = []

    with NonlinearityInjector(model, NonlinearityConfig(alpha=alpha, scope=scope)):
        for step, batch in enumerate(tqdm(loader, desc=f"vox eval alpha={alpha:g}", leave=False)):
            if max_batches is not None and step >= max_batches:
                break
            waveforms = batch["waveforms"].to(device, non_blocking=True)
            waveform_lengths = batch["waveform_lengths"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)
            logits, output_lengths = model(waveforms, waveform_lengths)
            loss = ctc_loss(logits, output_lengths, targets, target_lengths, blank_id=vocab.blank_id)
            loss_meter.update(float(loss.item()), waveforms.size(0))

            pred_ids = logits.argmax(dim=-1)
            batch_hypotheses = vocab.decode_batch(pred_ids)
            batch_references = list(batch["texts"])
            references.extend(batch_references)
            hypotheses.extend(batch_hypotheses)
            if collect_examples > 0 and len(examples) < collect_examples:
                for audio_id, ref, hyp in zip(batch["audio_ids"], batch_references, batch_hypotheses):
                    if len(examples) >= collect_examples:
                        break
                    examples.append(
                        {
                            "audio_id": audio_id,
                            "reference": ref,
                            "hypothesis": hyp,
                        }
                    )

    wer = corpus_error_rate(references, hypotheses, unit="word")
    cer = corpus_error_rate(references, hypotheses, unit="char")
    return {
        "loss": loss_meter.avg,
        "wer": wer,
        "cer": cer,
        "num_examples": len(references),
        "references": references,
        "hypotheses": hypotheses,
        "examples": examples,
    }


def alpha_sweep(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    alphas: list[float],
    scope: str,
    vocab,
    args,
    checkpoint_path: Path,
) -> pd.DataFrame:
    rows = []
    clean_wer = math.nan
    clean_cer = math.nan
    for alpha in alphas:
        result = evaluate(
            model,
            loader,
            device,
            alpha=alpha,
            scope=scope,
            vocab=vocab,
            max_batches=args.eval_limit_batches,
            collect_examples=args.examples_per_alpha,
        )
        row = {
            "dataset": "voxforge",
            "model": args.model,
            "method": args.method,
            "run_name": args.run_name,
            "checkpoint": str(checkpoint_path),
            "alpha": alpha,
            "wer": result["wer"],
            "cer": result["cer"],
            "loss": result["loss"],
            "num_examples": result["num_examples"],
            "scope": args.scope,
            "alpha_mode": args.alpha_mode,
            "train_alpha_low": args.alpha_low,
            "train_alpha_high": args.alpha_high,
            "train_fixed_alpha": args.alpha,
            "epochs": args.epochs,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
        }
        if abs(alpha) < 1e-12:
            clean_wer = float(result["wer"])
            clean_cer = float(result["cer"])
        rows.append(row)
        append_csv(row, args.output_dir / "tables" / "voxforge_alpha_eval.csv")

        for idx, example in enumerate(result["examples"]):
            append_csv(
                {
                    "dataset": "voxforge",
                    "model": args.model,
                    "method": args.method,
                    "run_name": args.run_name,
                    "alpha": alpha,
                    "example_index": idx,
                    **example,
                },
                args.output_dir / "tables" / "voxforge_decode_examples.csv",
            )

    df = pd.DataFrame(rows)
    nonzero = df[df["alpha"].abs() > 1e-12]
    if math.isnan(clean_wer):
        zero = df[df["alpha"].abs() < 1e-12]
        if not zero.empty:
            clean_wer = float(zero.iloc[0]["wer"])
            clean_cer = float(zero.iloc[0]["cer"])
    summary = {
        "dataset": "voxforge",
        "model": args.model,
        "method": args.method,
        "run_name": args.run_name,
        "checkpoint": str(checkpoint_path),
        "clean_wer": clean_wer,
        "clean_cer": clean_cer,
        "avg_wer": float(df["wer"].mean()),
        "avg_cer": float(df["cer"].mean()),
        "avg_robust_wer": float(nonzero["wer"].mean()),
        "avg_robust_cer": float(nonzero["cer"].mean()),
        "worst_wer": float(df["wer"].max()),
        "worst_cer": float(df["cer"].max()),
        "max_relative_wer_increase": float((df["wer"].max() - clean_wer) / max(clean_wer, 1e-12)),
        "scope": args.scope,
        "alpha_mode": args.alpha_mode,
        "epochs": args.epochs,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
    }
    append_csv(summary, args.output_dir / "tables" / "voxforge_method_summary.csv")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2 VoxForge nonlinearity-aware ASR training")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "task2")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--config-name", type=str, default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--model", type=str, default="crnn_ctc")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--max-duration", type=float, default=8.0)
    parser.add_argument("--max-text-length", type=int, default=160)
    parser.add_argument("--shuffle-buffer", type=int, default=5000)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--alpha-mode", choices=["none", "fixed", "uniform", "curriculum"], default="none")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--alpha-low", type=float, default=-1.0)
    parser.add_argument("--alpha-high", type=float, default=1.0)
    parser.add_argument("--alpha-max-abs", type=float, default=1.0)
    parser.add_argument("--alpha-warmup-epochs", type=int, default=20)
    parser.add_argument("--scope", choices=["per_tensor", "per_sample", "per_channel"], default="per_tensor")
    parser.add_argument(
        "--test-alphas",
        type=str,
        default="-1.0,-0.8,-0.6,-0.4,-0.2,-0.1,0,0.1,0.2,0.4,0.6,0.8,1.0",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--train-limit-batches", type=int, default=None)
    parser.add_argument("--eval-limit-batches", type=int, default=None)
    parser.add_argument("--examples-per-alpha", type=int, default=5)
    args = parser.parse_args()

    if not args.run_name:
        args.run_name = args.method

    set_seed(args.seed)
    args.output_dir = Path(args.output_dir)
    checkpoint_dir = ensure_dir(args.output_dir / "checkpoints" / "voxforge")
    ensure_dir(args.output_dir / "tables")
    ensure_dir(args.output_dir / "figures")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, vocab, data_metadata = build_voxforge_loaders(
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
        shuffle_buffer=args.shuffle_buffer,
        refresh_cache=args.refresh_cache,
    )

    model = build_asr_model(
        args.model,
        vocab_size=len(vocab),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    if args.init_checkpoint is not None:
        load_checkpoint(args.init_checkpoint, model, map_location=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    alpha_scheduler = build_alpha_scheduler(
        args.alpha_mode,
        alpha=args.alpha,
        low=args.alpha_low,
        high=args.alpha_high,
        max_abs=args.alpha_max_abs,
        warmup_epochs=args.alpha_warmup_epochs,
    )

    metadata = {
        "dataset": "voxforge",
        "model": args.model,
        "method": args.method,
        "run_name": args.run_name,
        "parameters": count_parameters(model),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "alpha_scheduler": alpha_scheduler.state_dict(),
        "data": data_metadata,
        "vocab": vocab.to_dict(),
    }
    save_json(metadata, args.output_dir / "tables" / f"voxforge_{args.run_name}_metadata.json")

    best_wer = float("inf")
    best_val_loss = float("inf")
    best_path = checkpoint_dir / f"{args.run_name}_best.pt"
    last_path = checkpoint_dir / f"{args.run_name}_last.pt"

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            alpha_scheduler=alpha_scheduler,
            scope=args.scope,
            blank_id=vocab.blank_id,
            amp=args.amp,
            max_batches=args.train_limit_batches,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            alpha=0.0,
            scope=args.scope,
            vocab=vocab,
            max_batches=args.eval_limit_batches,
            collect_examples=0,
        )
        lr_scheduler.step()

        log_row = {
            "dataset": "voxforge",
            "model": args.model,
            "method": args.method,
            "run_name": args.run_name,
            "epoch": epoch + 1,
            "train_loss": train_metrics["train_loss"],
            "train_alpha_abs_mean": train_metrics["train_alpha_abs_mean"],
            "val_loss": val_metrics["loss"],
            "val_wer": val_metrics["wer"],
            "val_cer": val_metrics["cer"],
            "lr": current_lr(optimizer),
            "alpha_mode": args.alpha_mode,
            "scope": args.scope,
            "train_size": args.train_size,
            "val_size": args.val_size,
        }
        append_csv(log_row, args.output_dir / "tables" / "voxforge_training_logs.csv")

        current_val_wer = float(val_metrics["wer"])
        current_val_loss = float(val_metrics["loss"])
        improved = current_val_wer < best_wer - 1e-12 or (
            abs(current_val_wer - best_wer) <= 1e-12 and current_val_loss < best_val_loss
        )
        if improved:
            best_wer = float(val_metrics["wer"])
            best_val_loss = current_val_loss
            save_checkpoint(
                best_path,
                model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                epoch=epoch + 1,
                best_metric=best_wer,
                metadata=metadata,
            )

        print(
            f"epoch={epoch + 1:03d} train_loss={train_metrics['train_loss']:.4f} "
            f"val_wer={float(val_metrics['wer']):.4f} val_cer={float(val_metrics['cer']):.4f} "
            f"best_wer={best_wer:.4f} best_val_loss={best_val_loss:.4f}"
        )

    save_checkpoint(
        last_path,
        model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        epoch=args.epochs,
        best_metric=best_wer,
        metadata=metadata,
    )

    load_checkpoint(best_path, model, map_location=device)
    alpha_sweep(
        model=model,
        loader=test_loader,
        device=device,
        alphas=parse_float_list(args.test_alphas),
        scope=args.scope,
        vocab=vocab,
        args=args,
        checkpoint_path=best_path,
    )


if __name__ == "__main__":
    main()
