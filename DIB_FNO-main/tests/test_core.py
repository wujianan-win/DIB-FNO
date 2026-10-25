import torch

from dibfno.data import LATITUDES
from dibfno.metrics import latitude_weights, weighted_rmse
from dibfno.model import DIBFNO, FixedFNO, StaticGateFNO


def test_model_output_and_gate_shapes() -> None:
    model = DIBFNO(hidden_dim=16, n_blocks=1, mask_hidden_dim=8)
    inputs = torch.randn(2, 8, 32, 64)
    prediction, gate = model(inputs, return_mask=True)
    assert prediction.shape == inputs.shape
    assert gate.shape == (2, 1, 32, 33)
    assert torch.isfinite(prediction).all()
    assert torch.all((gate > 0) & (gate < 1))


def test_dynamic_and_static_gate_semantics() -> None:
    dynamic = DIBFNO(hidden_dim=16, n_blocks=1, mask_hidden_dim=8)
    static = StaticGateFNO(hidden_dim=17, n_blocks=1)
    samples = torch.stack((torch.zeros(8, 32, 64), torch.ones(8, 32, 64)))
    _, dynamic_gate = dynamic(samples, return_mask=True)
    _, static_gate = static(samples, return_mask=True)
    assert not torch.allclose(dynamic_gate[0], dynamic_gate[1])
    assert torch.allclose(static_gate[0], static_gate[1])


def test_fixed_baseline_has_no_learned_gate() -> None:
    fixed = FixedFNO(hidden_dim=16, n_blocks=1)
    inputs = torch.randn(1, 8, 32, 64)
    _, gate = fixed(inputs, return_mask=True)
    assert torch.equal(gate, torch.ones_like(gate))


def test_weighted_rmse_identity_and_constant_error() -> None:
    weights = latitude_weights(torch.from_numpy(LATITUDES.copy()))
    truth = torch.zeros(3, 8, 32, 64)
    assert torch.equal(weighted_rmse(truth, truth, weights), torch.zeros(8))
    assert torch.allclose(weighted_rmse(truth + 2, truth, weights), torch.full((8,), 2.0))
