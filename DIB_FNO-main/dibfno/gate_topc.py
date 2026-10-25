"""Hard top-C gating and gate-family ablations for review request 5.

Review request 5 offers two routes.  Route B (delete the strong top-C claim) is
satisfied by the documentation in this package.  Route A (keep the claim) needs
the experiments implemented here:

* an explicit hard top-C inference operator that retains exactly ``C`` Fourier
  coefficients for *every* sample, so C is a real, per-sample-constant budget;
* a C-sensitivity sweep on real validation data;
* soft-gate versus hard-gate comparison at matched retention;
* retained-mode statistics, including how many coefficients the soft gate
  actually keeps;
* an accuracy-versus-sparsity Pareto curve driven by a gate threshold
  (soft thresholding) and by C (hard selection).

Nothing here retrains or alters existing checkpoints; every mode is an
inference-time override of the gate on a frozen trained backbone.
"""

from __future__ import annotations

import numpy as np
import torch

from .model import DIBFNO, FixedFNO, StaticGateFNO


GATE_MODES = (
    "soft_input_dependent",
    "soft_frequency_mean",
    "soft_elementwise_mean",
    "hard_top_c",
    "soft_threshold",
)


class HardTopCGate(torch.nn.Module):
    """Retain exactly C coefficients per sample, chosen from a reference gate."""

    def __init__(self, reference: np.ndarray, top_c: int):
        super().__init__()
        if reference.shape != (32, 33):
            raise ValueError("reference gate must be 32x33")
        flat = np.asarray(reference, dtype=np.float64).reshape(-1)
        if not np.isfinite(flat).all():
            raise ValueError("reference gate must be finite")
        if not 1 <= top_c <= flat.size:
            raise ValueError(f"top_c must be in 1..{flat.size}")
        order = np.argsort(-flat, kind="stable")
        mask = np.zeros(flat.size, dtype=np.float32)
        mask[order[:top_c]] = 1.0
        self.top_c = int(top_c)
        self.register_buffer("value", torch.from_numpy(mask.reshape(32, 33)).view(1, 1, 32, 33))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value.expand(x.shape[0], -1, -1, -1)


class SoftThresholdGate(torch.nn.Module):
    """Zero every soft gate weight below a threshold; keep the rest as-is."""

    def __init__(self, reference: np.ndarray, threshold: float):
        super().__init__()
        if reference.shape != (32, 33):
            raise ValueError("reference gate must be 32x33")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be inside (0, 1)")
        values = np.where(reference >= threshold, reference, 0.0)
        self.threshold = float(threshold)
        self.retained_frequencies = int((reference >= threshold).sum())
        self.register_buffer(
            "value", torch.from_numpy(values.astype(np.float32)).view(1, 1, 32, 33)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value.expand(x.shape[0], -1, -1, -1)


def mode_statistics(gate: np.ndarray, *, thresholds=(0.25, 0.5, 0.75, 0.9)) -> dict:
    """Pure descriptive statistics of one 32x33 soft gate map.

    The soft gate is a bounded weight map, not a probability distribution, so the
    entropy fields are computed on the *normalised* top-C support: the smallest
    leading set of coefficients that carries 99% of the total gate weight.  That
    makes "effective modes" (exp of the entropy) a well-defined sparsity number
    instead of an artefact of unnormalised weights.
    """
    values = np.asarray(gate, dtype=np.float64)
    if values.shape != (32, 33) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("gate must be a finite, non-negative 32x33 array")
    flat = np.sort(values.reshape(-1))[::-1]
    total = float(values.sum())
    if total <= 0:
        raise ValueError("gate has zero total weight")
    cumulative = np.cumsum(flat) / total
    support = int(np.searchsorted(cumulative, 0.99) + 1)
    normalized = flat[:support] / float(flat[:support].sum())
    entropy = float(-(normalized * np.log(normalized)).sum())
    result = {
        "weight_sum": total,
        "grid_frequencies": int(values.size),
        "weight_sum_fraction_of_grid": total / values.size,
        "above_threshold_counts": {str(t): int((values >= t).sum()) for t in thresholds},
        "top_coefficients_to_reach_50pct_weight": int(np.searchsorted(cumulative, 0.5) + 1),
        "top_coefficients_to_reach_80pct_weight": int(np.searchsorted(cumulative, 0.8) + 1),
        "top_coefficients_to_reach_90pct_weight": int(np.searchsorted(cumulative, 0.9) + 1),
        "max_weight": float(flat[0]),
        "median_weight": float(np.median(values)),
        "support_for_99pct_weight": support,
        "entropy_nats_on_99pct_support": entropy,
        "effective_modes_exp_entropy": float(np.exp(entropy)),
        "effective_modes_fraction_of_support": float(np.exp(entropy) / support),
        "gate_uniformity": total / (values.size * float(flat[0])),
    }
    return result


def effective_discrete_modes(gate: np.ndarray, *, threshold: float = 0.5) -> int:
    """Count of discrete coefficients the soft gate keeps at a stated threshold.

    A soft gate has no intrinsic mode count; this number is only defined
    together with the threshold, and is reported that way everywhere.
    """
    return int((np.asarray(gate, dtype=np.float64) >= threshold).sum())


def load_frozen_backbone(checkpoint_path, *, gate: torch.nn.Module | None = None):
    """Load a trained backbone and optionally replace its gate wholesale."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_type = checkpoint.get("model_type", "dibfno")
    classes = {"dibfno": DIBFNO, "fixed_fno": FixedFNO, "static_fno": StaticGateFNO}
    if model_type not in classes:
        raise ValueError(f"unsupported checkpoint model_type: {model_type}")
    model = classes[model_type](**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    if gate is not None:
        if model_type == "fixed_fno":
            raise ValueError("a gate override is meaningless for a gate-free Fixed FNO")
        model.mask = gate
    model.eval()
    return model, checkpoint
