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

from src.cifar_models import build_cifar10_loaders, build_cifar_model
from src.metrics import count_parameters, top1_accuracy
from src.nonlinear import (
    EndpointAlphaNonlinearityInjector,
    GraphInputRecorder,
    RandomAlphaNonlinearityInjector,
    list_target_layers,
)
from src.sensitivity import (
    default_sensitivity,
    layer_weight_dict,
    load_cifar_resnet20_sensitivity,
    save_sensitivity_weights,
)
from src.task3_losses import activation_vulnerability_loss, kl_consistency
from src.training import AverageMeter, append_csv, ensure_dir, load_checkpoint, save_checkpoint, save_json, set_seed


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def js_consistency(logits_list: list[torch.Tensor], temperature: float = 2.0) -> torch.Tensor:
    if len(logits_list) < 2:
        return logits_list[0].sum() * 0.0
    temperature = max(float(temperature), 1e-6)
    probs = [F.softmax(logits / temperature, dim=-1) for logits in logits_list]
    mean_prob = torch.stack(probs, dim=0).mean(dim=0).clamp_min(1e-12)
    losses = []
    for logits in logits_list:
        log_prob = F.log_softmax(logits / temperature, dim=-1)
        losses.append(F.kl_div(log_prob, mean_prob.detach(), reduction="batchmean"))
    return torch.stack(losses).mean() * (temperature * temperature)


def forward_random_alpha(
    model: torch.nn.Module,
    images: torch.Tensor,
    alpha_low: float,
    alpha_high: float,
    scope: str,
    alpha_mode: str = "uniform",
) -> torch.Tensor:
    injector_cls = EndpointAlphaNonlinearityInjector if alpha_mode == "endpoint" else RandomAlphaNonlinearityInjector
    with injector_cls(
        model,
        alpha_low=alpha_low,
        alpha_high=alpha_high,
        scope=scope,
    ):
        return model(images)


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    sensitive_layers: list[str],
    layer_weights: dict[str, float],
) -> dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    total_meter = AverageMeter()
    clean_meter = AverageMeter()
    rand_meter = AverageMeter()
    kd_meter = AverageMeter()
    cons_meter = AverageMeter()
    vuln_meter = AverageMeter()
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=f"per-occ NAT train epoch {epoch + 1}", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.numel()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            with GraphInputRecorder(model, sensitive_layers) as input_rec:
                logits_clean = model(images)
            loss_clean = F.cross_entropy(logits_clean, labels)

            random_logits: list[torch.Tensor] = []
            random_losses: list[torch.Tensor] = []
            kd_losses: list[torch.Tensor] = []
            for _ in range(max(1, args.random_views)):
                logits_random = forward_random_alpha(
                    model,
                    images,
                    alpha_low=args.alpha_low,
                    alpha_high=args.alpha_high,
                    scope=args.scope,
                    alpha_mode=args.train_alpha_mode,
                )
                random_logits.append(logits_random)
                random_losses.append(F.cross_entropy(logits_random, labels))
                kd_losses.append(kl_consistency(logits_clean, logits_random, temperature=args.temperature))

            loss_random = torch.stack(random_losses).mean()
            loss_kd = torch.stack(kd_losses).mean()
            loss_cons = js_consistency(random_logits, temperature=args.temperature)
            loss_vuln = activation_vulnerability_loss(input_rec.inputs, layer_weights, scope=args.scope)
            loss = (
                args.lambda_clean * loss_clean
                + args.lambda_rand * loss_random
                + args.lambda_kd * loss_kd
                + args.lambda_cons * loss_cons
                + args.lambda_vuln * loss_vuln
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_correct, batch_total = top1_accuracy(logits_clean.detach(), labels)
        correct += batch_correct
        total += batch_total
        total_meter.update(float(loss.detach().item()), batch_size)
        clean_meter.update(float(loss_clean.detach().item()), batch_size)
        rand_meter.update(float(loss_random.detach().item()), batch_size)
        kd_meter.update(float(loss_kd.detach().item()), batch_size)
        cons_meter.update(float(loss_cons.detach().item()), batch_size)
        vuln_meter.update(float(loss_vuln.detach().item()), batch_size)

    return {
        "train_loss": total_meter.avg,
        "train_loss_clean": clean_meter.avg,
        "train_loss_random": rand_meter.avg,
        "train_loss_kd": kd_meter.avg,
        "train_loss_cons": cons_meter.avg,
        "train_loss_vuln": vuln_meter.avg,
        "train_clean_accuracy": correct / max(total, 1),
    }


@torch.inference_mode()
def evaluate_clean(model: torch.nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_meter = AverageMeter()
    for images, labels in tqdm(loader, desc="per-occ NAT clean eval", leave=False):
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
def evaluate_random_once(model: torch.nn.Module, loader, device: torch.device, args) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_meter = AverageMeter()
    for images, labels in tqdm(loader, desc="per-occ NAT random-field eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = forward_random_alpha(
            model,
            images,
            alpha_low=args.alpha_low,
            alpha_high=args.alpha_high,
            scope=args.scope,
            alpha_mode="uniform",
        )
        loss = F.cross_entropy(logits, labels, reduction="sum")
        batch_correct, batch_total = top1_accuracy(logits, labels)
        correct += batch_correct
        total += batch_total
        loss_meter.update(float(loss.item()) / max(batch_total, 1), batch_total)
    return {"accuracy": correct / max(total, 1), "loss": loss_meter.avg}


@torch.inference_mode()
def evaluate_random_mc(model: torch.nn.Module, loader, device: torch.device, args, mc: int) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_meter = AverageMeter()
    mc = max(1, int(mc))
    for images, labels in tqdm(loader, desc=f"per-occ NAT MC-{mc} eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        probs = []
        for _ in range(mc):
            logits = forward_random_alpha(
                model,
                images,
                alpha_low=args.alpha_low,
                alpha_high=args.alpha_high,
                scope=args.scope,
                alpha_mode="uniform",
            )
            probs.append(F.softmax(logits, dim=-1))
        mean_prob = torch.stack(probs, dim=0).mean(dim=0).clamp_min(1e-12)
        log_prob = torch.log(mean_prob)
        loss = F.nll_loss(log_prob, labels, reduction="sum")
        pred = mean_prob.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += labels.numel()
        loss_meter.update(float(loss.item()) / max(labels.numel(), 1), labels.numel())
    return {"accuracy": correct / max(total, 1), "loss": loss_meter.avg}


def final_random_field_eval(model: torch.nn.Module, loader, device: torch.device, args, checkpoint_path: Path) -> None:
    clean = evaluate_clean(model, loader, device)
    append_csv(
        {
            "dataset": "cifar10",
            "model": args.model,
            "method": args.method,
            "run_name": args.run_name,
            "checkpoint": str(checkpoint_path),
            "eval_type": "clean",
            "accuracy": clean["accuracy"],
            "loss": clean["loss"],
            "mc": 0,
            "repeat": 0,
        },
        args.output_dir / "tables" / "cifar_random_field_eval.csv",
    )

    repeat_accs: list[float] = []
    for repeat in range(args.eval_repeats):
        result = evaluate_random_once(model, loader, device, args)
        repeat_accs.append(float(result["accuracy"]))
        append_csv(
            {
                "dataset": "cifar10",
                "model": args.model,
                "method": args.method,
                "run_name": args.run_name,
                "checkpoint": str(checkpoint_path),
                "eval_type": "random_single",
                "accuracy": result["accuracy"],
                "loss": result["loss"],
                "mc": 1,
                "repeat": repeat + 1,
            },
            args.output_dir / "tables" / "cifar_random_field_eval.csv",
        )

    mc_values = [int(item.strip()) for item in args.mc_values.split(",") if item.strip()]
    for mc in mc_values:
        result = evaluate_random_mc(model, loader, device, args, mc=mc)
        append_csv(
            {
                "dataset": "cifar10",
                "model": args.model,
                "method": args.method,
                "run_name": args.run_name,
                "checkpoint": str(checkpoint_path),
                "eval_type": "random_mc",
                "accuracy": result["accuracy"],
                "loss": result["loss"],
                "mc": mc,
                "repeat": 0,
            },
            args.output_dir / "tables" / "cifar_random_field_eval.csv",
        )

    summary = {
        "dataset": "cifar10",
        "model": args.model,
        "method": args.method,
        "run_name": args.run_name,
        "checkpoint": str(checkpoint_path),
        "clean_accuracy": clean["accuracy"],
        "random_mean_accuracy": sum(repeat_accs) / max(len(repeat_accs), 1),
        "random_worst_accuracy": min(repeat_accs) if repeat_accs else math.nan,
        "random_std_accuracy": statistics.pstdev(repeat_accs) if len(repeat_accs) > 1 else 0.0,
        "eval_repeats": args.eval_repeats,
        "alpha_low": args.alpha_low,
        "alpha_high": args.alpha_high,
        "random_views": args.random_views,
        "lambda_clean": args.lambda_clean,
        "lambda_rand": args.lambda_rand,
        "lambda_kd": args.lambda_kd,
        "lambda_cons": args.lambda_cons,
        "lambda_vuln": args.lambda_vuln,
    }
    append_csv(summary, args.output_dir / "tables" / "cifar_method_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10 per-occurrence NAT for per-operator random alpha.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "task2_random_cifar_nat")
    parser.add_argument("--model", type=str, default="cifar10_resnet20")
    parser.add_argument("--method", type=str, default="per_occ_nat")
    parser.add_argument("--run-name", type=str, default="cifar_per_occ_nat")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--val-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--scope", choices=["per_tensor", "per_sample", "per_channel"], default="per_tensor")
    parser.add_argument("--alpha-low", type=float, default=-1.0)
    parser.add_argument("--alpha-high", type=float, default=1.0)
    parser.add_argument("--train-alpha-mode", choices=["uniform", "endpoint"], default="uniform")
    parser.add_argument("--random-views", type=int, default=2)
    parser.add_argument("--lambda-clean", type=float, default=1.0)
    parser.add_argument("--lambda-rand", type=float, default=1.0)
    parser.add_argument("--lambda-kd", type=float, default=1.0)
    parser.add_argument("--lambda-cons", type=float, default=0.5)
    parser.add_argument("--lambda-vuln", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--top-k-sensitive-layers", type=int, default=8)
    parser.add_argument("--sensitivity-temperature", type=float, default=0.5)
    parser.add_argument("--sensitivity-csv", type=Path, default=ROOT / "outputs" / "task1" / "tables" / "cifar10_resnet20_layer_sensitivity.csv")
    parser.add_argument("--drift-csv", type=Path, default=ROOT / "outputs" / "task1" / "tables" / "cifar10_resnet20_activation_drift.csv")
    parser.add_argument("--eval-repeats", type=int, default=5)
    parser.add_argument("--mc-values", type=str, default="1,3,5,9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

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
    model = build_cifar_model(args.model, pretrained=False).to(device)
    if args.init_checkpoint is not None:
        load_checkpoint(args.init_checkpoint, model, map_location=device)

    sensitivity_items = load_cifar_resnet20_sensitivity(
        args.sensitivity_csv,
        args.drift_csv,
        top_k=args.top_k_sensitive_layers,
        temperature=args.sensitivity_temperature,
    )
    module_names = {name for name, _ in model.named_modules()}
    sensitivity_items = [item for item in sensitivity_items if item.layer in module_names]
    if not sensitivity_items:
        sensitivity_items = [
            item for item in default_sensitivity("cifar10", args.model, args.top_k_sensitive_layers) if item.layer in module_names
        ]
    sensitive_layers = [item.layer for item in sensitivity_items]
    layer_weights = layer_weight_dict(sensitivity_items)
    save_sensitivity_weights(sensitivity_items, args.output_dir / "tables" / "cifar_sensitivity_weights.csv")

    metadata = {
        "dataset": "cifar10",
        "model": args.model,
        "method": args.method,
        "run_name": args.run_name,
        "device": str(device),
        "params": count_parameters(model),
        "target_layers": len(list_target_layers(model)),
        "sensitive_layers": sensitive_layers,
        "layer_weights": layer_weights,
        "protocol": "per_operator_random_alpha",
        "alpha_low": args.alpha_low,
        "alpha_high": args.alpha_high,
        "train_alpha_mode": args.train_alpha_mode,
        "random_views": args.random_views,
        "loss_weights": {
            "lambda_clean": args.lambda_clean,
            "lambda_rand": args.lambda_rand,
            "lambda_kd": args.lambda_kd,
            "lambda_cons": args.lambda_cons,
            "lambda_vuln": args.lambda_vuln,
        },
    }
    save_json(metadata, args.output_dir / "tables" / f"{args.run_name}_metadata.json")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best_path = checkpoint_dir / f"{args.run_name}_best.pt"
    last_path = checkpoint_dir / f"{args.run_name}_last.pt"

    if args.eval_only or args.epochs <= 0:
        eval_path = args.init_checkpoint if args.init_checkpoint is not None else last_path
        final_random_field_eval(model, test_loader, device, args, checkpoint_path=eval_path)
        return

    best_random = -math.inf
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            sensitive_layers=sensitive_layers,
            layer_weights=layer_weights,
        )
        clean_val = evaluate_clean(model, val_loader, device)
        random_val = evaluate_random_once(model, val_loader, device, args)
        scheduler.step()
        row = {
            "dataset": "cifar10",
            "model": args.model,
            "method": args.method,
            "run_name": args.run_name,
            "epoch": epoch + 1,
            **train_metrics,
            "val_clean_accuracy": clean_val["accuracy"],
            "val_clean_loss": clean_val["loss"],
            "val_random_accuracy": random_val["accuracy"],
            "val_random_loss": random_val["loss"],
            "lr": current_lr(optimizer),
            "protocol": "per_operator_random_alpha",
        }
        append_csv(row, args.output_dir / "tables" / "cifar_training_logs.csv")
        if random_val["accuracy"] > best_random:
            best_random = float(random_val["accuracy"])
            save_checkpoint(
                best_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_metric=best_random,
                metadata=metadata,
            )
        print(
            f"[{args.run_name}] epoch {epoch + 1}/{args.epochs} "
            f"loss={train_metrics['train_loss']:.4f} "
            f"train_clean={train_metrics['train_clean_accuracy']:.4f} "
            f"val_clean={clean_val['accuracy']:.4f} "
            f"val_random={random_val['accuracy']:.4f}"
        )

    save_checkpoint(
        last_path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.epochs,
        best_metric=best_random,
        metadata=metadata,
    )
    load_checkpoint(best_path, model, map_location=device)
    final_random_field_eval(model, test_loader, device, args, checkpoint_path=best_path)
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()
