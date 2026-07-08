from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Iterable

import torch
from torch import nn


DEFAULT_TARGET_TYPES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.Linear,
)


@dataclass(frozen=True)
class NonlinearityConfig:
    alpha: float = 0.0
    eps: float = 1e-12
    scope: str = "per_tensor"


@dataclass(frozen=True)
class PreCompensationConfig:
    beta: float = 0.0
    eps: float = 1e-12
    scope: str = "per_tensor"
    alpha_aware: bool = False
    inverse: bool = False


@dataclass(frozen=True)
class FixedPointProjectionConfig:
    threshold: float = 0.25
    mode: str = "ternary"
    mix: float = 1.0
    ste: bool = True
    eps: float = 1e-12
    scope: str = "per_tensor"


def normalization_max(x: torch.Tensor, scope: str = "per_tensor", eps: float = 1e-12) -> torch.Tensor:
    """Return the detached max magnitude used by the contest nonlinearity."""
    if scope == "per_tensor":
        return x.detach().abs().amax().clamp_min(eps)
    if scope == "per_sample":
        reduce_dims = tuple(range(1, x.ndim))
        return x.detach().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(eps)
    if scope == "per_channel":
        if x.ndim < 3:
            return x.detach().abs().amax().clamp_min(eps)
        reduce_dims = tuple(i for i in range(x.ndim) if i != 1)
        return x.detach().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(eps)
    raise ValueError(f"Unsupported nonlinearity scope: {scope}")


def nonlinearize_tensor(x: torch.Tensor, config: NonlinearityConfig) -> torch.Tensor:
    """Apply the contest cubic input nonlinearity to one activation tensor."""
    if config.alpha == 0.0:
        return x

    max_val = normalization_max(x, scope=config.scope, eps=config.eps)
    u = x / max_val
    y = config.alpha * (u**3) + (1.0 - config.alpha) * u
    return y * max_val


def nonlinear_error_basis(x: torch.Tensor, scope: str = "per_tensor", eps: float = 1e-12) -> torch.Tensor:
    """Return g(x) where x_alpha = x + alpha * g(x)."""
    max_val = normalization_max(x, scope=scope, eps=eps)
    u = x / max_val
    return max_val * (u**3 - u)


def vulnerability_regularizer(x: torch.Tensor, scope: str = "per_tensor", eps: float = 1e-12) -> torch.Tensor:
    """Return mean((u^3 - u)^2), the closed-form cubic nonlinearity vulnerability."""
    max_val = normalization_max(x, scope=scope, eps=eps)
    u = x / max_val
    return ((u**3 - u) ** 2).mean()


def fixed_point_project_tensor(x: torch.Tensor, config: FixedPointProjectionConfig) -> torch.Tensor:
    """Project activations onto fixed points of the contest cubic map.

    For normalized values u in {-1, 0, 1}, alpha * u^3 + (1-alpha) * u = u
    for every alpha.  This projection is therefore alpha-independent; STE keeps
    the projected model trainable during fine-tuning.
    """
    mix = float(config.mix)
    if mix <= 0.0:
        return x
    max_val = normalization_max(x, scope=config.scope, eps=config.eps)
    u = x / max_val
    if config.mode == "binary":
        q_u = torch.where(u >= 0, torch.ones_like(u), -torch.ones_like(u))
    elif config.mode == "ternary":
        threshold = max(0.0, float(config.threshold))
        q_u = torch.where(
            u.abs() >= threshold,
            u.sign(),
            torch.zeros_like(u),
        )
    else:
        raise ValueError(f"Unsupported fixed-point projection mode: {config.mode}")

    projected = q_u * max_val
    if mix < 1.0:
        projected = (1.0 - mix) * x + mix * projected
    if config.ste and x.requires_grad:
        return x + (projected - x).detach()
    return projected


def precompensate_tensor(
    x: torch.Tensor,
    beta: float | torch.Tensor,
    scope: str = "per_tensor",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Apply first-order cubic pre-compensation x - beta * g(x)."""
    if isinstance(beta, float) and abs(beta) < 1e-12:
        return x
    return x - beta * nonlinear_error_basis(x, scope=scope, eps=eps)


def inverse_precompensate_tensor(
    x: torch.Tensor,
    alpha: float,
    scope: str = "per_tensor",
    eps: float = 1e-12,
    iterations: int = 20,
) -> torch.Tensor:
    """Approximate z such that nonlinearize_tensor(z, alpha) reconstructs x.

    The contest nonlinearity is applied to normalized inputs:
    y = alpha * u^3 + (1 - alpha) * u.  This routine solves that cubic with a
    bracketed inverse on the normalized target x / max(|x|).  For alpha < -0.5
    the cubic is not globally monotone on [-1, 1], so the inverse is computed on
    its central monotone branch and exact boundary targets are pinned to ±1.
    This preserves the max-value normalization used by the hardware model.
    """
    if abs(float(alpha)) < 1e-12:
        return x
    max_val = normalization_max(x, scope=scope, eps=eps)
    target = (x / max_val).clamp(min=-1.0, max=1.0)
    alpha_value = float(alpha)
    if alpha_value < -0.5:
        branch_limit = math.sqrt((1.0 - alpha_value) / (-3.0 * alpha_value))
        low = torch.full_like(target, -branch_limit)
        high = torch.full_like(target, branch_limit)
    else:
        low = torch.full_like(target, -1.0)
        high = torch.full_like(target, 1.0)

    alpha_t = torch.as_tensor(alpha_value, device=x.device, dtype=x.dtype)
    one_minus_alpha = 1.0 - alpha_t
    for _ in range(max(1, int(iterations))):
        mid = 0.5 * (low + high)
        value = alpha_t * (mid**3) + one_minus_alpha * mid
        move_right = value < target
        low = torch.where(move_right, mid, low)
        high = torch.where(move_right, high, mid)
    u = 0.5 * (low + high)
    edge = target.abs() >= (1.0 - 1e-7)
    u = torch.where(edge, target.sign(), u)
    solved = u * max_val
    if x.requires_grad:
        return x + (solved - x).detach()
    return solved


def list_target_layers(
    model: nn.Module,
    target_types: tuple[type[nn.Module], ...] = DEFAULT_TARGET_TYPES,
    include: Callable[[str, nn.Module], bool] | None = None,
) -> list[tuple[str, nn.Module]]:
    layers: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, target_types) and (include is None or include(name, module)):
            layers.append((name, module))
    return layers


class NonlinearityInjector:
    """Installs forward pre-hooks that distort inputs to selected matrix operators."""

    def __init__(
        self,
        model: nn.Module,
        config: NonlinearityConfig,
        enabled_layers: Iterable[str] | None = None,
        target_types: tuple[type[nn.Module], ...] = DEFAULT_TARGET_TYPES,
        include: Callable[[str, nn.Module], bool] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.enabled_layers = set(enabled_layers) if enabled_layers is not None else None
        self.target_types = target_types
        self.include = include
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "NonlinearityInjector":
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    def install(self) -> None:
        self.remove()
        for name, module in list_target_layers(self.model, self.target_types, self.include):
            if self.enabled_layers is not None and name not in self.enabled_layers:
                continue
            self.handles.append(module.register_forward_pre_hook(self._make_hook(name)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]):
            if not inputs:
                return inputs
            x = inputs[0]
            if not torch.is_tensor(x):
                return inputs
            distorted = nonlinearize_tensor(x, self.config)
            return (distorted, *inputs[1:])

        return hook


class RandomAlphaNonlinearityInjector(NonlinearityInjector):
    """Distorts each target-operator input with a freshly sampled alpha.

    This matches the stricter contest interpretation where every matrix
    operator invocation may see its own random nonlinearity strength instead of
    sharing one global alpha across the whole inference pass.
    """

    def __init__(
        self,
        model: nn.Module,
        alpha_low: float = -1.0,
        alpha_high: float = 1.0,
        eps: float = 1e-12,
        scope: str = "per_tensor",
        enabled_layers: Iterable[str] | None = None,
        target_types: tuple[type[nn.Module], ...] = DEFAULT_TARGET_TYPES,
        include: Callable[[str, nn.Module], bool] | None = None,
        record_alphas: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            config=NonlinearityConfig(alpha=0.0, eps=eps, scope=scope),
            enabled_layers=enabled_layers,
            target_types=target_types,
            include=include,
        )
        self.alpha_low = float(alpha_low)
        self.alpha_high = float(alpha_high)
        self.record_alphas = record_alphas
        self.alpha_trace: list[tuple[str, float]] = []

    def _sample_alpha(self) -> float:
        return random.uniform(self.alpha_low, self.alpha_high)

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]):
            if not inputs:
                return inputs
            x = inputs[0]
            if not torch.is_tensor(x):
                return inputs
            alpha = self._sample_alpha()
            if self.record_alphas:
                self.alpha_trace.append((name, alpha))
            distorted = nonlinearize_tensor(
                x,
                NonlinearityConfig(alpha=alpha, eps=self.config.eps, scope=self.config.scope),
            )
            return (distorted, *inputs[1:])

        return hook


class EndpointAlphaNonlinearityInjector(RandomAlphaNonlinearityInjector):
    """Samples each operator-call alpha from interval endpoints {-1, +1}.

    This is used for distribution-free training: if a model is stable under
    independently sampled endpoint fields, interior alpha values in [-1, 1] are
    directly covered by the same cubic error coefficient instead of relying on a
    particular alpha probability law.
    """

    def _sample_alpha(self) -> float:
        return -1.0 if random.random() < 0.5 else 1.0


class FixedPointInputProjector(NonlinearityInjector):
    """Projects target-operator inputs onto alpha-invariant cubic fixed points."""

    def __init__(
        self,
        model: nn.Module,
        config: FixedPointProjectionConfig | None = None,
        enabled_layers: Iterable[str] | None = None,
        target_types: tuple[type[nn.Module], ...] = DEFAULT_TARGET_TYPES,
        include: Callable[[str, nn.Module], bool] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            config=NonlinearityConfig(alpha=0.0, eps=(config.eps if config else 1e-12), scope=(config.scope if config else "per_tensor")),
            enabled_layers=enabled_layers,
            target_types=target_types,
            include=include,
        )
        self.projection_config = config or FixedPointProjectionConfig()

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]):
            if not inputs:
                return inputs
            x = inputs[0]
            if not torch.is_tensor(x):
                return inputs
            projected = fixed_point_project_tensor(x, self.projection_config)
            return (projected, *inputs[1:])

        return hook


@dataclass(frozen=True)
class BitSerialFixedPointConfig:
    bits: int = 4
    eps: float = 1e-12
    scope: str = "per_tensor"
    simulate_hardware_nonlinearity: bool = True
    endpoint_alpha: bool = True


def fixed_point_bitserial_planes(
    x: torch.Tensor,
    config: BitSerialFixedPointConfig,
) -> list[tuple[float, torch.Tensor]]:
    """Decompose x into weighted {-max, 0, +max} fixed-point planes.

    Each plane is invariant to the official nonlinearity for any alpha because
    its normalized entries are only -1, 0, or +1.  The weighted sum of all
    planes is a uniform quantization of the original activation.
    """
    bits = max(1, int(config.bits))
    levels = (1 << bits) - 1
    max_val = normalization_max(x, scope=config.scope, eps=config.eps)
    u = (x / max_val).clamp(min=-1.0, max=1.0)
    magnitude = torch.round(u.abs() * levels).to(torch.int64)
    sign = torch.sign(u)
    planes: list[tuple[float, torch.Tensor]] = []
    for bit in range(bits):
        weight = float(1 << bit) / float(levels)
        mask = ((magnitude >> bit) & 1).to(dtype=x.dtype)
        plane = sign * mask * max_val
        planes.append((weight, plane))
    return planes


class BitSerialFixedPointInjector(NonlinearityInjector):
    """Runs matrix operators with alpha-invariant fixed-point bit-serial inputs."""

    def __init__(
        self,
        model: nn.Module,
        bitserial_config: BitSerialFixedPointConfig | None = None,
        enabled_layers: Iterable[str] | None = None,
        target_types: tuple[type[nn.Module], ...] = DEFAULT_TARGET_TYPES,
        include: Callable[[str, nn.Module], bool] | None = None,
    ) -> None:
        cfg = bitserial_config or BitSerialFixedPointConfig()
        super().__init__(
            model=model,
            config=NonlinearityConfig(alpha=0.0, eps=cfg.eps, scope=cfg.scope),
            enabled_layers=enabled_layers,
            target_types=target_types,
            include=include,
        )
        self.bitserial_config = cfg
        self._inside_hook = False

    def install(self) -> None:
        self.remove()
        for name, module in list_target_layers(self.model, self.target_types, self.include):
            if self.enabled_layers is not None and name not in self.enabled_layers:
                continue
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output):
            if self._inside_hook or not inputs:
                return output
            x = inputs[0]
            if not torch.is_tensor(x):
                return output
            self._inside_hook = True
            try:
                acc = None
                for weight, plane in fixed_point_bitserial_planes(x, self.bitserial_config):
                    if self.bitserial_config.simulate_hardware_nonlinearity:
                        alpha = 1.0 if self.bitserial_config.endpoint_alpha else random.uniform(-1.0, 1.0)
                        plane = nonlinearize_tensor(
                            plane,
                            NonlinearityConfig(
                                alpha=alpha,
                                eps=self.bitserial_config.eps,
                                scope=self.bitserial_config.scope,
                            ),
                        )
                    partial = module.forward(plane, *inputs[1:])
                    weighted = partial * weight
                    acc = weighted if acc is None else acc + weighted
                return acc
            finally:
                self._inside_hook = False

        return hook


@dataclass(frozen=True)
class FP32IEEEBitSerialConfig:
    mantissa_bits: int = 23
    include_implicit_bit: bool = True
    eps: float = 1e-12
    scope: str = "per_tensor"
    simulate_hardware_nonlinearity: bool = True
    alpha_low: float = -1.0
    alpha_high: float = 1.0


def fp32_ieee_value_planes(
    x: torch.Tensor,
    config: FP32IEEEBitSerialConfig,
) -> list[torch.Tensor]:
    """Return additive value planes derived from the IEEE754 FP32 bit fields.

    Raw sign and exponent storage bits are not additive real-value planes.  This
    routine therefore decodes the IEEE754 sign/exponent fields and emits the
    additive implicit/mantissa value contributions that make linear operators
    well-defined under bit-serial execution.
    """
    x32 = x.to(torch.float32)
    raw = x32.view(torch.int32)
    dtype = x32.dtype
    sign_bit = ((raw >> 31) & 1).to(dtype=dtype)
    sign = 1.0 - 2.0 * sign_bit
    exp_field = (raw >> 23) & 0xFF
    mantissa = raw & 0x7FFFFF
    finite = torch.isfinite(x32)
    normal = (exp_field > 0) & (exp_field < 255) & finite
    subnormal = (exp_field == 0) & (mantissa != 0) & finite

    normal_exp = exp_field.to(dtype=dtype) - 127.0
    normal_scale = torch.pow(torch.full_like(x32, 2.0), normal_exp)
    subnormal_scale = torch.full_like(x32, 2.0 ** -126)
    scale = torch.where(normal, normal_scale, subnormal_scale)
    active = normal | subnormal

    planes: list[torch.Tensor] = []
    if config.include_implicit_bit:
        planes.append(torch.where(normal, sign * scale, torch.zeros_like(x32)))

    bits = max(0, min(23, int(config.mantissa_bits)))
    for fraction_bit in range(1, bits + 1):
        storage_bit = 23 - fraction_bit
        bit_mask = ((mantissa >> storage_bit) & 1).to(dtype=dtype)
        contribution = sign * bit_mask * scale * float(2.0 ** -fraction_bit)
        planes.append(torch.where(active, contribution, torch.zeros_like(x32)))
    return planes


def _module_bias_as_output(module: nn.Module, output: torch.Tensor) -> torch.Tensor | None:
    bias = getattr(module, "bias", None)
    if bias is None or not torch.is_tensor(bias):
        return None
    if isinstance(module, nn.Linear):
        shape = [1] * output.ndim
        shape[-1] = bias.numel()
        return bias.reshape(shape)
    if output.ndim >= 2:
        shape = [1] * output.ndim
        shape[1] = bias.numel()
        return bias.reshape(shape)
    return bias


class FP32IEEEBitSerialInjector(NonlinearityInjector):
    """Runs operators on IEEE754-derived additive FP32 value planes."""

    def __init__(
        self,
        model: nn.Module,
        fp32_config: FP32IEEEBitSerialConfig | None = None,
        enabled_layers: Iterable[str] | None = None,
        target_types: tuple[type[nn.Module], ...] = DEFAULT_TARGET_TYPES,
        include: Callable[[str, nn.Module], bool] | None = None,
    ) -> None:
        cfg = fp32_config or FP32IEEEBitSerialConfig()
        super().__init__(
            model=model,
            config=NonlinearityConfig(alpha=0.0, eps=cfg.eps, scope=cfg.scope),
            enabled_layers=enabled_layers,
            target_types=target_types,
            include=include,
        )
        self.fp32_config = cfg
        self._inside_hook = False

    def install(self) -> None:
        self.remove()
        for name, module in list_target_layers(self.model, self.target_types, self.include):
            if self.enabled_layers is not None and name not in self.enabled_layers:
                continue
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output):
            if self._inside_hook or not inputs:
                return output
            x = inputs[0]
            if not torch.is_tensor(x):
                return output
            self._inside_hook = True
            try:
                acc = None
                plane_count = 0
                for plane in fp32_ieee_value_planes(x, self.fp32_config):
                    plane_count += 1
                    if self.fp32_config.simulate_hardware_nonlinearity:
                        alpha = random.uniform(self.fp32_config.alpha_low, self.fp32_config.alpha_high)
                        plane = nonlinearize_tensor(
                            plane,
                            NonlinearityConfig(
                                alpha=alpha,
                                eps=self.fp32_config.eps,
                                scope=self.fp32_config.scope,
                            ),
                        )
                    partial = module.forward(plane, *inputs[1:])
                    acc = partial if acc is None else acc + partial
                if acc is not None and plane_count > 1:
                    bias = _module_bias_as_output(module, acc)
                    if bias is not None:
                        acc = acc - float(plane_count - 1) * bias
                return acc if acc is not None else output
            finally:
                self._inside_hook = False

        return hook


@dataclass(frozen=True)
class BlackBoxDistortionConfig:
    family: str = "contest_cubic"
    anchors: int = 65
    eps: float = 1e-12
    scope: str = "per_tensor"
    alpha_low: float = -1.0
    alpha_high: float = 1.0
    gamma_low: float = 0.5
    gamma_high: float = 2.0
    tanh_gain_low: float = 0.5
    tanh_gain_high: float = 3.0
    sinusoid_low: float = -0.35
    sinusoid_high: float = 0.35


def _sample_blackbox_params(config: BlackBoxDistortionConfig) -> tuple[str, dict[str, float]]:
    family = config.family
    if family == "mixed":
        family = random.choice(["contest_cubic", "gamma", "tanh", "sinusoid"])
    if family == "contest_cubic":
        return family, {"alpha": random.uniform(config.alpha_low, config.alpha_high)}
    if family == "gamma":
        return family, {"gamma": random.uniform(config.gamma_low, config.gamma_high)}
    if family == "tanh":
        return family, {"gain": random.uniform(config.tanh_gain_low, config.tanh_gain_high)}
    if family == "sinusoid":
        return family, {"beta": random.uniform(config.sinusoid_low, config.sinusoid_high)}
    raise ValueError(f"Unsupported black-box distortion family: {config.family}")


def _distort_unit_interval(u: torch.Tensor, family: str, params: dict[str, float]) -> torch.Tensor:
    if family == "contest_cubic":
        alpha = float(params["alpha"])
        return alpha * (u**3) + (1.0 - alpha) * u
    if family == "gamma":
        gamma = float(params["gamma"])
        return torch.sign(u) * torch.pow(u.abs().clamp_min(0.0), gamma)
    if family == "tanh":
        gain = float(params["gain"])
        denom = math.tanh(gain)
        return torch.tanh(gain * u) / max(denom, 1e-12)
    if family == "sinusoid":
        beta = float(params["beta"])
        return (u + beta * torch.sin(math.pi * u)).clamp(min=-1.0, max=1.0)
    raise ValueError(f"Unsupported black-box distortion family: {family}")


def blackbox_distort_tensor(
    x: torch.Tensor,
    config: BlackBoxDistortionConfig,
    family: str,
    params: dict[str, float],
) -> torch.Tensor:
    max_val = normalization_max(x, scope=config.scope, eps=config.eps)
    u = (x / max_val).clamp(min=-1.0, max=1.0)
    return _distort_unit_interval(u, family, params) * max_val


def _interp1d_sorted(x_query: torch.Tensor, x_points: torch.Tensor, y_points: torch.Tensor) -> torch.Tensor:
    flat = x_query.reshape(-1)
    idx = torch.searchsorted(x_points.contiguous(), flat.contiguous(), right=False)
    idx = idx.clamp(min=1, max=x_points.numel() - 1)
    x0 = x_points[idx - 1]
    x1 = x_points[idx]
    y0 = y_points[idx - 1]
    y1 = y_points[idx]
    denom = (x1 - x0).clamp_min(1e-12)
    t = ((flat - x0) / denom).clamp(min=0.0, max=1.0)
    out = y0 + t * (y1 - y0)
    return out.reshape_as(x_query)


def pilot_inverse_precompensate(
    x: torch.Tensor,
    config: BlackBoxDistortionConfig,
    family: str,
    params: dict[str, float],
) -> torch.Tensor:
    """Estimate a black-box inverse from pilot anchors and pre-distort x."""
    max_val = normalization_max(x, scope=config.scope, eps=config.eps)
    u = (x / max_val).clamp(min=-1.0, max=1.0)
    anchors = max(5, int(config.anchors))
    if anchors % 2 == 0:
        anchors += 1
    z = torch.linspace(-1.0, 1.0, anchors, device=x.device, dtype=x.dtype)
    observed = _distort_unit_interval(z, family, params)
    observed_sorted, order = torch.sort(observed)
    z_sorted = z[order]
    u_clamped = u.clamp(min=float(observed_sorted[0].item()), max=float(observed_sorted[-1].item()))
    z_hat = _interp1d_sorted(u_clamped, observed_sorted, z_sorted)
    return z_hat * max_val


class BlackBoxRandomDistortionInjector(NonlinearityInjector):
    """Applies an unknown random distortion family to target-operator inputs."""

    def __init__(
        self,
        model: nn.Module,
        blackbox_config: BlackBoxDistortionConfig | None = None,
        enabled_layers: Iterable[str] | None = None,
        target_types: tuple[type[nn.Module], ...] = DEFAULT_TARGET_TYPES,
        include: Callable[[str, nn.Module], bool] | None = None,
    ) -> None:
        cfg = blackbox_config or BlackBoxDistortionConfig()
        super().__init__(
            model=model,
            config=NonlinearityConfig(alpha=0.0, eps=cfg.eps, scope=cfg.scope),
            enabled_layers=enabled_layers,
            target_types=target_types,
            include=include,
        )
        self.blackbox_config = cfg

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]):
            if not inputs:
                return inputs
            x = inputs[0]
            if not torch.is_tensor(x):
                return inputs
            family, params = _sample_blackbox_params(self.blackbox_config)
            distorted = blackbox_distort_tensor(x, self.blackbox_config, family, params)
            return (distorted, *inputs[1:])

        return hook


class BlindPilotInverseInjector(BlackBoxRandomDistortionInjector):
    """Formula-agnostic pilot-calibrated inverse for random distortions."""

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]):
            if not inputs:
                return inputs
            x = inputs[0]
            if not torch.is_tensor(x):
                return inputs
            family, params = _sample_blackbox_params(self.blackbox_config)
            precompensated = pilot_inverse_precompensate(x, self.blackbox_config, family, params)
            distorted = blackbox_distort_tensor(precompensated, self.blackbox_config, family, params)
            return (distorted, *inputs[1:])

        return hook


class PreCompensatedNonlinearityInjector(NonlinearityInjector):
    """Distorts operator inputs after optional first-order cubic pre-compensation."""

    def __init__(
        self,
        model: nn.Module,
        nonlinearity_config: NonlinearityConfig,
        precomp_config: PreCompensationConfig | None = None,
        layer_betas: dict[str, float | torch.Tensor] | None = None,
        enabled_layers: Iterable[str] | None = None,
        target_types: tuple[type[nn.Module], ...] = DEFAULT_TARGET_TYPES,
        include: Callable[[str, nn.Module], bool] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            config=nonlinearity_config,
            enabled_layers=enabled_layers,
            target_types=target_types,
            include=include,
        )
        self.precomp_config = precomp_config or PreCompensationConfig(
            beta=0.0,
            eps=nonlinearity_config.eps,
            scope=nonlinearity_config.scope,
        )
        self.layer_betas = layer_betas or {}

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]):
            if not inputs:
                return inputs
            x = inputs[0]
            if not torch.is_tensor(x):
                return inputs
            if self.precomp_config.inverse:
                precompensated = inverse_precompensate_tensor(
                    x,
                    alpha=float(self.config.alpha),
                    scope=self.precomp_config.scope,
                    eps=self.precomp_config.eps,
                )
            else:
                beta = self.layer_betas.get(name, self.precomp_config.beta)
                if self.precomp_config.alpha_aware:
                    beta = beta * float(self.config.alpha)
                precompensated = precompensate_tensor(
                    x,
                    beta=beta,
                    scope=self.precomp_config.scope,
                    eps=self.precomp_config.eps,
                )
            distorted = nonlinearize_tensor(precompensated, self.config)
            return (distorted, *inputs[1:])

        return hook


class ActivationRecorder:
    """Records selected module outputs for distribution and propagation analysis."""

    def __init__(
        self,
        model: nn.Module,
        layer_names: Iterable[str],
        keep_batches: int = 1,
        detach_cpu: bool = True,
    ) -> None:
        self.model = model
        self.layer_names = list(layer_names)
        self.keep_batches = keep_batches
        self.detach_cpu = detach_cpu
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.outputs: dict[str, list[torch.Tensor]] = {name: [] for name in self.layer_names}
        self._name_to_module = dict(model.named_modules())

    def __enter__(self) -> "ActivationRecorder":
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    def install(self) -> None:
        self.remove()
        for name in self.layer_names:
            module = self._name_to_module.get(name)
            if module is None:
                continue
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs, output):
            if len(self.outputs[name]) >= self.keep_batches:
                return
            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor):
                return
            tensor = tensor.detach()
            if self.detach_cpu:
                tensor = tensor.cpu()
            self.outputs[name].append(tensor)

        return hook

    def stacked(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for name, chunks in self.outputs.items():
            if not chunks:
                continue
            try:
                result[name] = torch.cat(chunks, dim=0)
            except RuntimeError:
                result[name] = torch.stack([c.flatten() for c in chunks], dim=0)
        return result


class GraphActivationRecorder:
    """Records selected module outputs while keeping the computation graph."""

    def __init__(self, model: nn.Module, layer_names: Iterable[str]) -> None:
        self.model = model
        self.layer_names = list(layer_names)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.outputs: dict[str, torch.Tensor] = {}
        self._name_to_module = dict(model.named_modules())

    def __enter__(self) -> "GraphActivationRecorder":
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    def clear(self) -> None:
        self.outputs.clear()

    def install(self) -> None:
        self.remove()
        self.clear()
        for name in self.layer_names:
            module = self._name_to_module.get(name)
            if module is None:
                continue
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(tensor):
                self.outputs[name] = tensor

        return hook


class GraphInputRecorder:
    """Records selected module inputs while keeping the computation graph."""

    def __init__(self, model: nn.Module, layer_names: Iterable[str]) -> None:
        self.model = model
        self.layer_names = list(layer_names)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.inputs: dict[str, torch.Tensor] = {}
        self._name_to_module = dict(model.named_modules())

    def __enter__(self) -> "GraphInputRecorder":
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    def clear(self) -> None:
        self.inputs.clear()

    def install(self) -> None:
        self.remove()
        self.clear()
        for name in self.layer_names:
            module = self._name_to_module.get(name)
            if module is None:
                continue
            self.handles.append(module.register_forward_pre_hook(self._make_hook(name)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]):
            if inputs and torch.is_tensor(inputs[0]):
                self.inputs[name] = inputs[0]
            return inputs

        return hook
