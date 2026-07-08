from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

import numpy as np
import torch


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return int(correct), int(labels.numel())


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    flat = x.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "abs_mean": float(flat.abs().mean().item()),
        "abs_max": float(flat.abs().max().item()),
        "zero_ratio": float((flat == 0).float().mean().item()),
    }


def compare_tensors(clean: torch.Tensor, perturbed: torch.Tensor, bins: int = 128) -> dict[str, float]:
    clean_flat = clean.detach().float().reshape(-1)
    pert_flat = perturbed.detach().float().reshape(-1)
    n = min(clean_flat.numel(), pert_flat.numel())
    if n == 0:
        return {}
    clean_flat = clean_flat[:n]
    pert_flat = pert_flat[:n]

    clean_norm = clean_flat.norm().clamp_min(1e-12)
    diff = pert_flat - clean_flat
    rel_l2 = diff.norm() / clean_norm
    cosine = torch.nn.functional.cosine_similarity(
        clean_flat.unsqueeze(0), pert_flat.unsqueeze(0), dim=1, eps=1e-12
    ).item()

    lo = float(min(clean_flat.min().item(), pert_flat.min().item()))
    hi = float(max(clean_flat.max().item(), pert_flat.max().item()))
    if lo == hi:
        js = 0.0
    else:
        clean_hist = torch.histc(clean_flat.cpu(), bins=bins, min=lo, max=hi)
        pert_hist = torch.histc(pert_flat.cpu(), bins=bins, min=lo, max=hi)
        js = js_divergence(clean_hist.numpy(), pert_hist.numpy())

    clean_stats = tensor_stats(clean_flat)
    pert_stats = tensor_stats(pert_flat)
    return {
        "relative_l2": float(rel_l2.item()),
        "cosine_similarity": float(cosine),
        "cosine_drift": float(1.0 - cosine),
        "js_divergence": float(js),
        "mean_shift": float(pert_stats["mean"] - clean_stats["mean"]),
        "std_ratio": float(pert_stats["std"] / max(clean_stats["std"], 1e-12)),
        "abs_mean_ratio": float(pert_stats["abs_mean"] / max(clean_stats["abs_mean"], 1e-12)),
        "zero_ratio_shift": float(pert_stats["zero_ratio"] - clean_stats["zero_ratio"]),
    }


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-12))))


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9áéíóúüñ'\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, ref_token in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, hyp_token in enumerate(hyp, start=1):
            if ref_token == hyp_token:
                cur[j] = prev[j - 1]
            else:
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + 1)
        prev = cur
    return prev[-1]


def corpus_error_rate(references: Sequence[str], hypotheses: Sequence[str], unit: str = "word") -> float:
    total_dist = 0
    total_units = 0
    for ref, hyp in zip(references, hypotheses):
        ref_norm = normalize_text(ref)
        hyp_norm = normalize_text(hyp)
        if unit == "word":
            ref_units = ref_norm.split()
            hyp_units = hyp_norm.split()
        elif unit == "char":
            ref_units = list(ref_norm.replace(" ", ""))
            hyp_units = list(hyp_norm.replace(" ", ""))
        else:
            raise ValueError(f"Unsupported unit: {unit}")
        total_dist += edit_distance(ref_units, hyp_units)
        total_units += len(ref_units)
    return float(total_dist / max(total_units, 1))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def class_histogram(labels: Sequence[int]) -> dict[int, int]:
    return dict(Counter(int(x) for x in labels))


def direction_gap(values: dict[float, float]) -> float:
    pos = [float(value) for alpha, value in values.items() if float(alpha) > 0]
    neg = [float(value) for alpha, value in values.items() if float(alpha) < 0]
    if not pos or not neg:
        return 0.0
    return abs(sum(pos) / len(pos) - sum(neg) / len(neg))


def robust_accuracy_score(
    alpha_to_accuracy: dict[float, float],
    clean_baseline: float | None = None,
    rho: float = 0.5,
    gamma: float = 0.5,
    kappa: float = 1.0,
    clean_tolerance: float = 0.01,
) -> float:
    if not alpha_to_accuracy:
        return -math.inf
    values = [float(value) for value in alpha_to_accuracy.values()]
    mean_acc = sum(values) / len(values)
    worst_acc = min(values)
    gap = direction_gap(alpha_to_accuracy)
    clean_acc = alpha_to_accuracy.get(0.0)
    clean_penalty = 0.0
    if clean_baseline is not None and clean_acc is not None:
        clean_penalty = max(0.0, float(clean_baseline) - float(clean_acc) - float(clean_tolerance))
    return mean_acc + float(rho) * worst_acc - float(gamma) * gap - float(kappa) * clean_penalty


def robust_error_score(
    alpha_to_error: dict[float, float],
    clean_baseline: float | None = None,
    rho: float = 0.5,
    gamma: float = 0.5,
    kappa: float = 1.0,
    clean_tolerance: float = 0.01,
) -> float:
    if not alpha_to_error:
        return -math.inf
    values = [float(value) for value in alpha_to_error.values()]
    mean_error = sum(values) / len(values)
    worst_error = max(values)
    gap = direction_gap(alpha_to_error)
    clean_error = alpha_to_error.get(0.0)
    clean_penalty = 0.0
    if clean_baseline is not None and clean_error is not None:
        clean_penalty = max(0.0, float(clean_error) - float(clean_baseline) - float(clean_tolerance))
    return -mean_error - float(rho) * worst_error - float(gamma) * gap - float(kappa) * clean_penalty
