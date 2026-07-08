from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.asr_models import build_asr_model
from src.metrics import corpus_error_rate, count_parameters
from src.nonlinear import RandomAlphaNonlinearityInjector, list_target_layers
from src.training import AverageMeter, append_csv, ensure_dir, load_checkpoint, save_checkpoint, save_json, set_seed
from src.voxforge_data import DEFAULT_CONFIG_NAME, DEFAULT_DATASET_NAME, build_voxforge_loaders


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


def forward_random_alpha(
    model: torch.nn.Module,
    waveforms: torch.Tensor,
    waveform_lengths: torch.Tensor,
    alpha_low: float,
    alpha_high: float,
    scope: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    with RandomAlphaNonlinearityInjector(
        model,
        alpha_low=alpha_low,
        alpha_high=alpha_high,
        scope=scope,
    ):
        return model(waveforms, waveform_lengths)


def decode_logits(logits: torch.Tensor, vocab) -> list[str]:
    return vocab.decode_batch(logits.argmax(dim=-1))


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    blank_id: int,
) -> dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    total_meter = AverageMeter()
    clean_meter = AverageMeter()
    random_meter = AverageMeter()

    for batch in tqdm(loader, desc=f"Vox per-occ NAT epoch {epoch + 1}", leave=False):
        waveforms = batch["waveforms"].to(device, non_blocking=True)
        waveform_lengths = batch["waveform_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        batch_size = waveforms.size(0)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            logits_clean, output_lengths_clean = model(waveforms, waveform_lengths)
            loss_clean = ctc_loss(logits_clean, output_lengths_clean, targets, target_lengths, blank_id)

            random_losses: list[torch.Tensor] = []
            for _ in range(max(1, args.random_views)):
                logits_random, output_lengths_random = forward_random_alpha(
                    model,
                    waveforms,
                    waveform_lengths,
                    alpha_low=args.alpha_low,
                    alpha_high=args.alpha_high,
                    scope=args.scope,
                )
                random_losses.append(ctc_loss(logits_random, output_lengths_random, targets, target_lengths, blank_id))
            loss_random = torch.stack(random_losses).mean()
            loss = args.lambda_clean * loss_clean + args.lambda_rand * loss_random

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_meter.update(float(loss.detach().item()), batch_size)
        clean_meter.update(float(loss_clean.detach().item()), batch_size)
        random_meter.update(float(loss_random.detach().item()), batch_size)

    return {
        "train_loss": total_meter.avg,
        "train_loss_clean": clean_meter.avg,
        "train_loss_random": random_meter.avg,
    }


@torch.inference_mode()
def evaluate_clean(model: torch.nn.Module, loader, device: torch.device, vocab) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    references: list[str] = []
    hypotheses: list[str] = []
    for batch in tqdm(loader, desc="Vox per-occ NAT clean eval", leave=False):
        waveforms = batch["waveforms"].to(device, non_blocking=True)
        waveform_lengths = batch["waveform_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        logits, output_lengths = model(waveforms, waveform_lengths)
        loss = ctc_loss(logits, output_lengths, targets, target_lengths, vocab.blank_id)
        loss_meter.update(float(loss.item()), waveforms.size(0))
        references.extend(batch["texts"])
        hypotheses.extend(decode_logits(logits, vocab))
    return {
        "loss": loss_meter.avg,
        "wer": corpus_error_rate(references, hypotheses, unit="word"),
        "cer": corpus_error_rate(references, hypotheses, unit="char"),
        "num_examples": len(references),
    }


@torch.inference_mode()
def evaluate_random_once(model: torch.nn.Module, loader, device: torch.device, vocab, args) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    references: list[str] = []
    hypotheses: list[str] = []
    for batch in tqdm(loader, desc="Vox per-occ NAT random eval", leave=False):
        waveforms = batch["waveforms"].to(device, non_blocking=True)
        waveform_lengths = batch["waveform_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        logits, output_lengths = forward_random_alpha(
            model,
            waveforms,
            waveform_lengths,
            alpha_low=args.alpha_low,
            alpha_high=args.alpha_high,
            scope=args.scope,
        )
        loss = ctc_loss(logits, output_lengths, targets, target_lengths, vocab.blank_id)
        loss_meter.update(float(loss.item()), waveforms.size(0))
        references.extend(batch["texts"])
        hypotheses.extend(decode_logits(logits, vocab))
    return {
        "loss": loss_meter.avg,
        "wer": corpus_error_rate(references, hypotheses, unit="word"),
        "cer": corpus_error_rate(references, hypotheses, unit="char"),
        "num_examples": len(references),
    }


@torch.inference_mode()
def evaluate_random_mc(model: torch.nn.Module, loader, device: torch.device, vocab, args, mc: int) -> dict[str, float]:
    model.eval()
    references: list[str] = []
    hypotheses: list[str] = []
    mc = max(1, int(mc))
    for batch in tqdm(loader, desc=f"Vox per-occ NAT MC-{mc} eval", leave=False):
        waveforms = batch["waveforms"].to(device, non_blocking=True)
        waveform_lengths = batch["waveform_lengths"].to(device, non_blocking=True)
        probs = []
        min_time = None
        for _ in range(mc):
            logits, _ = forward_random_alpha(
                model,
                waveforms,
                waveform_lengths,
                alpha_low=args.alpha_low,
                alpha_high=args.alpha_high,
                scope=args.scope,
            )
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


def final_random_field_eval(model: torch.nn.Module, loader, device: torch.device, vocab, args, checkpoint_path: Path) -> None:
    clean = evaluate_clean(model, loader, device, vocab)
    append_csv(
        {
            "dataset": "voxforge",
            "model": args.model,
            "method": args.method,
            "run_name": args.run_name,
            "checkpoint": str(checkpoint_path),
            "eval_type": "clean",
            "wer": clean["wer"],
            "cer": clean["cer"],
            "loss": clean["loss"],
            "mc": 0,
            "repeat": 0,
        },
        args.output_dir / "tables" / "voxforge_random_field_eval.csv",
    )
    random_wers: list[float] = []
    random_cers: list[float] = []
    for repeat in range(args.eval_repeats):
        result = evaluate_random_once(model, loader, device, vocab, args)
        random_wers.append(float(result["wer"]))
        random_cers.append(float(result["cer"]))
        append_csv(
            {
                "dataset": "voxforge",
                "model": args.model,
                "method": args.method,
                "run_name": args.run_name,
                "checkpoint": str(checkpoint_path),
                "eval_type": "random_single",
                "wer": result["wer"],
                "cer": result["cer"],
                "loss": result["loss"],
                "mc": 1,
                "repeat": repeat + 1,
            },
            args.output_dir / "tables" / "voxforge_random_field_eval.csv",
        )

    for mc in [int(item.strip()) for item in args.mc_values.split(",") if item.strip()]:
        result = evaluate_random_mc(model, loader, device, vocab, args, mc)
        append_csv(
            {
                "dataset": "voxforge",
                "model": args.model,
                "method": args.method,
                "run_name": args.run_name,
                "checkpoint": str(checkpoint_path),
                "eval_type": "random_mc",
                "wer": result["wer"],
                "cer": result["cer"],
                "loss": math.nan,
                "mc": mc,
                "repeat": 0,
            },
            args.output_dir / "tables" / "voxforge_random_field_eval.csv",
        )

    summary = {
        "dataset": "voxforge",
        "model": args.model,
        "method": args.method,
        "run_name": args.run_name,
        "checkpoint": str(checkpoint_path),
        "clean_wer": clean["wer"],
        "clean_cer": clean["cer"],
        "random_mean_wer": sum(random_wers) / max(len(random_wers), 1),
        "random_worst_wer": max(random_wers) if random_wers else math.nan,
        "random_std_wer": statistics.pstdev(random_wers) if len(random_wers) > 1 else 0.0,
        "random_mean_cer": sum(random_cers) / max(len(random_cers), 1),
        "random_worst_cer": max(random_cers) if random_cers else math.nan,
        "random_std_cer": statistics.pstdev(random_cers) if len(random_cers) > 1 else 0.0,
        "eval_repeats": args.eval_repeats,
        "alpha_low": args.alpha_low,
        "alpha_high": args.alpha_high,
        "random_views": args.random_views,
        "lambda_clean": args.lambda_clean,
        "lambda_rand": args.lambda_rand,
    }
    append_csv(summary, args.output_dir / "tables" / "voxforge_method_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxForge per-occurrence NAT under per-operator random alpha.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "task2_random_voxforge_nat")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--config-name", type=str, default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--model", type=str, default="crnn_ctc")
    parser.add_argument("--method", type=str, default="per_occ_nat")
    parser.add_argument("--run-name", type=str, default="voxforge_per_occ_nat")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-size", type=int, default=120)
    parser.add_argument("--val-size", type=int, default=30)
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--max-duration", type=float, default=5.0)
    parser.add_argument("--max-text-length", type=int, default=110)
    parser.add_argument("--shuffle-buffer", type=int, default=5000)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--scope", choices=["per_tensor", "per_sample", "per_channel"], default="per_tensor")
    parser.add_argument("--alpha-low", type=float, default=-1.0)
    parser.add_argument("--alpha-high", type=float, default=1.0)
    parser.add_argument("--random-views", type=int, default=1)
    parser.add_argument("--lambda-clean", type=float, default=1.0)
    parser.add_argument("--lambda-rand", type=float, default=1.0)
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--mc-values", type=str, default="1,3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

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

    metadata = {
        "dataset": "voxforge",
        "model": args.model,
        "method": args.method,
        "run_name": args.run_name,
        "params": count_parameters(model),
        "target_layers": len(list_target_layers(model)),
        "device": str(device),
        "protocol": "per_operator_per_call_random_alpha",
        "alpha_low": args.alpha_low,
        "alpha_high": args.alpha_high,
        "random_views": args.random_views,
        "loss_weights": {"lambda_clean": args.lambda_clean, "lambda_rand": args.lambda_rand},
        "data": data_metadata,
        "vocab": vocab.to_dict(),
    }
    save_json(metadata, args.output_dir / "tables" / f"{args.run_name}_metadata.json")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best_path = checkpoint_dir / f"{args.run_name}_best.pt"
    last_path = checkpoint_dir / f"{args.run_name}_last.pt"

    if args.eval_only or args.epochs <= 0:
        eval_path = args.init_checkpoint if args.init_checkpoint is not None else last_path
        final_random_field_eval(model, test_loader, device, vocab, args, checkpoint_path=eval_path)
        return

    best_random_cer = float("inf")
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch, args, vocab.blank_id)
        clean_val = evaluate_clean(model, val_loader, device, vocab)
        random_val = evaluate_random_once(model, val_loader, device, vocab, args)
        scheduler.step()
        row = {
            "dataset": "voxforge",
            "model": args.model,
            "method": args.method,
            "run_name": args.run_name,
            "epoch": epoch + 1,
            **train_metrics,
            "val_clean_wer": clean_val["wer"],
            "val_clean_cer": clean_val["cer"],
            "val_clean_loss": clean_val["loss"],
            "val_random_wer": random_val["wer"],
            "val_random_cer": random_val["cer"],
            "val_random_loss": random_val["loss"],
            "lr": current_lr(optimizer),
            "protocol": "per_operator_per_call_random_alpha",
        }
        append_csv(row, args.output_dir / "tables" / "voxforge_training_logs.csv")
        if float(random_val["cer"]) < best_random_cer:
            best_random_cer = float(random_val["cer"])
            save_checkpoint(
                best_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_metric=best_random_cer,
                metadata=metadata,
            )
        print(
            f"[{args.run_name}] epoch {epoch + 1}/{args.epochs} "
            f"loss={train_metrics['train_loss']:.4f} "
            f"val_clean_cer={clean_val['cer']:.4f} val_random_cer={random_val['cer']:.4f}"
        )

    save_checkpoint(
        last_path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.epochs,
        best_metric=best_random_cer,
        metadata=metadata,
    )
    load_checkpoint(best_path, model, map_location=device)
    final_random_field_eval(model, test_loader, device, vocab, args, checkpoint_path=best_path)
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()
