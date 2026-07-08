from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from src.nonlinear import vulnerability_regularizer


def smooth_max(losses: Sequence[torch.Tensor], tau: float = 0.2) -> torch.Tensor:
    if not losses:
        raise ValueError("smooth_max requires at least one loss tensor")
    stacked = torch.stack([loss.reshape(()) for loss in losses])
    if tau <= 0:
        return stacked.max()
    return torch.logsumexp(stacked / tau, dim=0) * tau


def kl_consistency(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature * temperature)


def ctc_frame_kl_consistency(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    time_steps = min(teacher_logits.size(1), student_logits.size(1))
    if time_steps <= 0:
        return student_logits.sum() * 0.0
    teacher = teacher_logits[:, :time_steps, :]
    student = student_logits[:, :time_steps, :]
    teacher_prob = F.softmax(teacher.detach() / temperature, dim=-1)
    student_log_prob = F.log_softmax(student / temperature, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature * temperature)


def feature_consistency(
    clean_features: dict[str, torch.Tensor],
    nonlinear_features: dict[str, torch.Tensor],
    layer_weights: dict[str, float],
    eta: float = 0.5,
    detach_clean: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    for layer, weight in layer_weights.items():
        clean = clean_features.get(layer)
        nonlinear = nonlinear_features.get(layer)
        if clean is None or nonlinear is None:
            continue
        if detach_clean:
            clean = clean.detach()
        clean_flat = clean.float().reshape(clean.size(0), -1)
        nonlinear_flat = nonlinear.float().reshape(nonlinear.size(0), -1)
        width = min(clean_flat.size(1), nonlinear_flat.size(1))
        if width <= 0:
            continue
        clean_flat = clean_flat[:, :width]
        nonlinear_flat = nonlinear_flat[:, :width]
        diff = nonlinear_flat - clean_flat
        rel_l2 = diff.norm(dim=1) / clean_flat.norm(dim=1).clamp_min(eps)
        cosine = F.cosine_similarity(nonlinear_flat, clean_flat, dim=1, eps=eps)
        layer_loss = rel_l2.mean() + float(eta) * (1.0 - cosine).mean()
        losses.append(layer_loss * float(weight))
        weights.append(float(weight))
    if not losses:
        return _zero_from_features(clean_features, nonlinear_features)
    return torch.stack(losses).sum() / max(sum(weights), eps)


def activation_vulnerability_loss(
    features_or_inputs: dict[str, torch.Tensor],
    layer_weights: dict[str, float],
    scope: str = "per_tensor",
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    for layer, weight in layer_weights.items():
        tensor = features_or_inputs.get(layer)
        if tensor is None:
            continue
        losses.append(vulnerability_regularizer(tensor, scope=scope) * float(weight))
        weights.append(float(weight))
    if not losses:
        return _zero_from_features(features_or_inputs)
    return torch.stack(losses).sum() / max(sum(weights), 1e-12)


def _zero_from_features(*feature_dicts: dict[str, torch.Tensor]) -> torch.Tensor:
    for feature_dict in feature_dicts:
        for tensor in feature_dict.values():
            return tensor.sum() * 0.0
    return torch.tensor(0.0)

