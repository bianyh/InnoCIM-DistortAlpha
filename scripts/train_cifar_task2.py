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
from src.cifar_models import build_cifar10_loaders, build_cifar_model
from src.metrics import count_parameters, top1_accuracy
from src.nonlinear import NonlinearityConfig, NonlinearityInjector, list_target_layers
from src.training import AverageMeter, append_csv, ensure_dir, load_checkpoint, save_checkpoint, save_json, set_seed


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def forward_with_alpha(model: torch.nn.Module, images: torch.Tensor, alpha: float, scope: str) -> torch.Tensor:
    if abs(alpha) < 1e-12:
        return model(images)
    with NonlinearityInjector(model, NonlinearityConfig(alpha=alpha, scope=scope)):
        return model(images)


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    alpha_scheduler,
    scope: str,
    amp: bool,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    alpha_abs_meter = AverageMeter()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    for step, (images, labels) in enumerate(tqdm(loader, desc=f"train epoch {epoch + 1}", leave=False)):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        alpha = alpha_scheduler.sample(epoch, step)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = forward_with_alpha(model, images, alpha=alpha, scope=scope)
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_correct, batch_total = top1_accuracy(logits.detach(), labels)
        loss_meter.update(float(loss.item()), batch_total)
        correct += batch_correct
        total += batch_total
        alpha_abs_meter.update(abs(alpha), 1)

    return {
        "train_loss": loss_meter.avg,
        "train_accuracy": correct / max(total, 1),
        "train_alpha_abs_mean": alpha_abs_meter.avg,
    }


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    alpha: float,
    scope: str,
) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    with NonlinearityInjector(model, NonlinearityConfig(alpha=alpha, scope=scope)):
        for images, labels in tqdm(loader, desc=f"eval alpha={alpha:g}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            batch_correct, batch_total = top1_accuracy(logits, labels)
            correct += batch_correct
            total += batch_total
            loss_meter.update(float(loss.item()) / max(batch_total, 1), batch_total)
    return {
        "loss": loss_meter.avg,
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
    }


def alpha_sweep(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    alphas: list[float],
    scope: str,
    args,
    checkpoint_path: Path,
) -> pd.DataFrame:
    rows = []
    clean_accuracy = None
    for alpha in alphas:
        result = evaluate(model, loader, device, alpha=alpha, scope=scope)
        if alpha == 0:
            clean_accuracy = result["accuracy"]
        row = {
            "dataset": "cifar10",
            "model": args.model,
            "method": args.method,
            "run_name": args.run_name,
            "checkpoint": str(checkpoint_path),
            "alpha": alpha,
            "accuracy": result["accuracy"],
            "loss": result["loss"],
            "correct": result["correct"],
            "total": result["total"],
            "scope": args.scope,
            "alpha_mode": args.alpha_mode,
            "train_alpha_low": args.alpha_low,
            "train_alpha_high": args.alpha_high,
            "train_fixed_alpha": args.alpha,
            "epochs": args.epochs,
        }
        rows.append(row)
        append_csv(row, args.output_dir / "tables" / "cifar_alpha_eval.csv")

    df = pd.DataFrame(rows)
    if clean_accuracy is None:
        zero = df[df["alpha"] == 0]
        clean_accuracy = float(zero.iloc[0]["accuracy"]) if not zero.empty else float(df["accuracy"].max())
    summary = {
        "dataset": "cifar10",
        "model": args.model,
        "method": args.method,
        "run_name": args.run_name,
        "checkpoint": str(checkpoint_path),
        "clean_accuracy": clean_accuracy,
        "avg_accuracy": float(df["accuracy"].mean()),
        "avg_robust_accuracy": float(df[df["alpha"] != 0]["accuracy"].mean()),
        "worst_accuracy": float(df["accuracy"].min()),
        "max_accuracy_drop": float(clean_accuracy - df["accuracy"].min()),
        "robust_auc_discrete": float(df.sort_values("alpha")["accuracy"].mean()),
        "scope": args.scope,
        "alpha_mode": args.alpha_mode,
        "epochs": args.epochs,
    }
    append_csv(summary, args.output_dir / "tables" / "cifar_method_summary.csv")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2 CIFAR-10 nonlinearity-aware training")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "task2")
    parser.add_argument("--model", type=str, default="cifar10_resnet20")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--val-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--alpha-mode", choices=["none", "fixed", "uniform", "curriculum"], default="none")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--alpha-low", type=float, default=-1.0)
    parser.add_argument("--alpha-high", type=float, default=1.0)
    parser.add_argument("--alpha-max-abs", type=float, default=1.0)
    parser.add_argument("--alpha-warmup-epochs", type=int, default=80)
    parser.add_argument("--scope", choices=["per_tensor", "per_sample", "per_channel"], default="per_tensor")
    parser.add_argument(
        "--test-alphas",
        type=str,
        default="-1.0,-0.8,-0.6,-0.4,-0.2,-0.1,0,0.1,0.2,0.4,0.6,0.8,1.0",
    )
    parser.add_argument("--selection-metric", choices=["clean_accuracy", "robust_accuracy"], default="clean_accuracy")
    parser.add_argument("--selection-alphas", type=str, default="-1.0,-0.5,0,0.5,1.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    args = parser.parse_args()

    if not args.run_name:
        args.run_name = args.method

    set_seed(args.seed)
    args.output_dir = Path(args.output_dir)
    checkpoint_dir = ensure_dir(args.output_dir / "checkpoints" / "cifar")
    ensure_dir(args.output_dir / "tables")
    ensure_dir(args.output_dir / "figures")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        val_size=args.val_size,
        seed=args.seed,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        test_subset=args.test_subset,
    )
    model = build_cifar_model(args.model, pretrained=args.pretrained).to(device)
    if args.init_checkpoint is not None:
        load_checkpoint(args.init_checkpoint, model, map_location=device)

    alpha_scheduler = build_alpha_scheduler(
        mode=args.alpha_mode,
        alpha=args.alpha,
        low=args.alpha_low,
        high=args.alpha_high,
        max_abs=args.alpha_max_abs,
        warmup_epochs=args.alpha_warmup_epochs,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    metadata = {
        "dataset": "cifar10",
        "model": args.model,
        "method": args.method,
        "run_name": args.run_name,
        "device": str(device),
        "params": count_parameters(model),
        "target_layers": len(list_target_layers(model)),
        "alpha_scheduler": alpha_scheduler.state_dict(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scope": args.scope,
        "selection_metric": args.selection_metric,
        "selection_alphas": args.selection_alphas,
    }
    save_json(metadata, args.output_dir / "tables" / f"cifar_{args.run_name}_metadata.json")

    best_selection_score = -math.inf
    best_path = checkpoint_dir / f"{args.run_name}_best.pt"
    last_path = checkpoint_dir / f"{args.run_name}_last.pt"
    selection_alphas = parse_float_list(args.selection_alphas)

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            alpha_scheduler=alpha_scheduler,
            scope=args.scope,
            amp=args.amp,
        )
        val_metrics = evaluate(model, val_loader, device, alpha=0.0, scope=args.scope)
        selection_results = {0.0: val_metrics}
        if args.selection_metric == "robust_accuracy":
            for selection_alpha in selection_alphas:
                if abs(selection_alpha) < 1e-12:
                    continue
                selection_results[selection_alpha] = evaluate(
                    model, val_loader, device, alpha=selection_alpha, scope=args.scope
                )
            selection_score = sum(result["accuracy"] for result in selection_results.values()) / len(selection_results)
        else:
            selection_score = val_metrics["accuracy"]
        scheduler.step()

        row = {
            "dataset": "cifar10",
            "model": args.model,
            "method": args.method,
            "run_name": args.run_name,
            "epoch": epoch + 1,
            "train_loss": train_metrics["train_loss"],
            "train_accuracy": train_metrics["train_accuracy"],
            "train_alpha_abs_mean": train_metrics["train_alpha_abs_mean"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "selection_score": selection_score,
            "selection_metric": args.selection_metric,
            "lr": current_lr(optimizer),
            "alpha_mode": args.alpha_mode,
            "scope": args.scope,
        }
        append_csv(row, args.output_dir / "tables" / "cifar_training_logs.csv")

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            save_checkpoint(
                best_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_metric=best_selection_score,
                metadata=metadata,
            )
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"{args.run_name}_epoch_{epoch + 1}.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_metric=best_selection_score,
                metadata=metadata,
            )
        print(
            f"[{args.run_name}] epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"train_acc={train_metrics['train_accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"selection={selection_score:.4f} "
            f"lr={current_lr(optimizer):.5f}"
        )

    save_checkpoint(
        last_path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.epochs,
        best_metric=best_selection_score,
        metadata=metadata,
    )
    load_checkpoint(best_path, model, map_location=device)
    alpha_sweep(
        model=model,
        loader=test_loader,
        device=device,
        alphas=parse_float_list(args.test_alphas),
        scope=args.scope,
        args=args,
        checkpoint_path=best_path,
    )
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()
