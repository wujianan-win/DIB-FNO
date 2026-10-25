"""Run a data-free DIB-FNO API smoke test on synthetic atmospheric tensors.

This example checks tensor shapes and input-dependent gate behavior. It does
not reproduce forecast skill and must not be interpreted as a trained model.
"""

from __future__ import annotations

import json

import torch

from dibfno.model import DIBFNO


def main() -> None:
    torch.manual_seed(0)
    model = DIBFNO(hidden_dim=16, n_blocks=1, mask_hidden_dim=8).eval()
    inputs = torch.randn(2, 8, 32, 64)

    with torch.no_grad():
        prediction, gate = model(inputs, return_mask=True)

    summary = {
        "input_shape": list(inputs.shape),
        "prediction_shape": list(prediction.shape),
        "gate_shape": list(gate.shape),
        "prediction_is_finite": bool(torch.isfinite(prediction).all()),
        "gate_min": float(gate.min()),
        "gate_max": float(gate.max()),
        "mean_absolute_gate_difference_between_samples": float(
            (gate[0] - gate[1]).abs().mean()
        ),
        "interpretation": (
            "Synthetic API smoke test only; use the documented WeatherBench2 "
            "workflow and released checkpoints for scientific evaluation."
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
