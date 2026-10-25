"""Complex-valued spectral models for the corrected 32x64 WB2 grid."""

from __future__ import annotations

import torch
from torch import nn


class DynamicSpectralMask(nn.Module):
    """Sample-conditioned mask computed from the input spectrum itself."""

    def __init__(self, in_channels: int, hidden: int = 32, temperature: float = 1.0):
        super().__init__()
        groups = max(1, min(8, hidden // 4))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.dc_logit = nn.Parameter(torch.tensor(4.0))
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def set_temperature(self, value: float) -> None:
        if value <= 0:
            raise ValueError("temperature must be greater than zero")
        self.temperature.fill_(float(value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft2(x.float(), norm="ortho")
        features = torch.log1p(spectrum.abs())
        logits = self.net(features)
        mask = torch.sigmoid(logits / self.temperature.clamp_min(1e-4))
        # The mean field is retained to prevent a trivial zero-frequency collapse.
        dc = torch.sigmoid(self.dc_logit).view(1, 1, 1, 1)
        head = torch.cat((dc.expand(x.shape[0], 1, 1, 1), mask[:, :, :1, 1:]), dim=-1)
        return torch.cat((head, mask[:, :, 1:, :]), dim=-2)


class ComplexSpectralBlock(nn.Module):
    """Residual spectral channel mixer that transforms both real and imaginary parts."""

    def __init__(self, dim: int, modes_lat: int = 8, modes_lon: int = 17, mlp_ratio: int = 2):
        super().__init__()
        self.modes_lat = modes_lat
        self.modes_lon = modes_lon
        self.norm = nn.GroupNorm(1, dim)
        self.real_mix = nn.Conv2d(dim, dim, 1, bias=False)
        self.imag_mix = nn.Conv2d(dim, dim, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_ratio, 1),
            nn.GELU(),
            nn.Conv2d(dim * mlp_ratio, dim, 1),
        )

    def _mix(self, z: torch.Tensor) -> torch.Tensor:
        real = self.real_mix(z.real) - self.imag_mix(z.imag)
        imag = self.real_mix(z.imag) + self.imag_mix(z.real)
        return torch.complex(real, imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x)
        spectrum = torch.fft.rfft2(y, norm="ortho")
        h, wf = spectrum.shape[-2:]
        ml = min(self.modes_lat, h // 2)
        mw = min(self.modes_lon, wf)
        modified = spectrum.clone()
        modified[:, :, :ml, :mw] = self._mix(spectrum[:, :, :ml, :mw])
        modified[:, :, -ml:, :mw] = self._mix(spectrum[:, :, -ml:, :mw])
        y = torch.fft.irfft2(modified, s=x.shape[-2:], norm="ortho")
        return residual + self.mlp(y)


class DIBFNO(nn.Module):
    """Residual six-hour forecast operator in per-channel standardized space."""

    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 64,
        n_blocks: int = 4,
        modes_lat: int = 8,
        modes_lon: int = 17,
        mask_hidden_dim: int = 32,
    ):
        super().__init__()
        self.mask = DynamicSpectralMask(in_channels, mask_hidden_dim)
        self.input_projection = nn.Conv2d(in_channels, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            [ComplexSpectralBlock(hidden_dim, modes_lat, modes_lon) for _ in range(n_blocks)]
        )
        self.output_projection = nn.Conv2d(hidden_dim, in_channels, 1)

    def set_temperature(self, value: float) -> None:
        self.mask.set_temperature(value)

    def forward(self, x: torch.Tensor, return_mask: bool = False):
        mask = self.mask(x)
        spectrum = torch.fft.rfft2(x, norm="ortho")
        filtered = torch.fft.irfft2(spectrum * mask, s=x.shape[-2:], norm="ortho")
        z = self.input_projection(filtered)
        for block in self.blocks:
            z = block(z)
        prediction = x + self.output_projection(z)
        return (prediction, mask) if return_mask else prediction


class FixedFNO(nn.Module):
    """Same complex spectral backbone without a sample-conditioned gate."""

    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 21,
        n_blocks: int = 1,
        modes_lat: int = 8,
        modes_lon: int = 17,
    ):
        super().__init__()
        self.input_projection = nn.Conv2d(in_channels, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            [ComplexSpectralBlock(hidden_dim, modes_lat, modes_lon) for _ in range(n_blocks)]
        )
        self.output_projection = nn.Conv2d(hidden_dim, in_channels, 1)

    def set_temperature(self, value: float) -> None:
        pass  # No mask or temperature in the fixed baseline.

    def forward(self, x: torch.Tensor, return_mask: bool = False):
        z = self.input_projection(x)
        for block in self.blocks:
            z = block(z)
        prediction = x + self.output_projection(z)
        if return_mask:
            mask = torch.ones(
                (x.shape[0], 1, x.shape[-2], x.shape[-1] // 2 + 1),
                dtype=x.dtype, device=x.device,
            )
            return prediction, mask
        return prediction


class LearnedStaticSpectralGate(nn.Module):
    """Trainable, input-independent soft gate with the same temperature schedule."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(1, 1, 32, 33))
        self.dc_logit = nn.Parameter(torch.tensor(4.0))
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def set_temperature(self, value: float) -> None:
        if value <= 0:
            raise ValueError("temperature must be positive")
        self.temperature.fill_(float(value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.sigmoid(self.logits / self.temperature.clamp_min(1e-4))
        dc = torch.sigmoid(self.dc_logit).view(1, 1, 1, 1)
        head = torch.cat((dc, mask[:, :, :1, 1:]), dim=-1)
        return torch.cat((head, mask[:, :, 1:, :]), dim=-2).expand(x.shape[0], -1, -1, -1)


class StaticGateFNO(nn.Module):
    """Same residual spectral backbone as DIB-FNO, but no input conditioning."""

    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 17,
        n_blocks: int = 1,
        modes_lat: int = 8,
        modes_lon: int = 17,
    ):
        super().__init__()
        self.mask = LearnedStaticSpectralGate()
        self.input_projection = nn.Conv2d(in_channels, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            [ComplexSpectralBlock(hidden_dim, modes_lat, modes_lon) for _ in range(n_blocks)]
        )
        self.output_projection = nn.Conv2d(hidden_dim, in_channels, 1)

    def set_temperature(self, value: float) -> None:
        self.mask.set_temperature(value)

    def forward(self, x: torch.Tensor, return_mask: bool = False):
        mask = self.mask(x)
        spectrum = torch.fft.rfft2(x, norm="ortho")
        filtered = torch.fft.irfft2(spectrum * mask, s=x.shape[-2:], norm="ortho")
        z = self.input_projection(filtered)
        for block in self.blocks:
            z = block(z)
        prediction = x + self.output_projection(z)
        return (prediction, mask) if return_mask else prediction


def effective_modes(mask: torch.Tensor) -> torch.Tensor:
    """Sum of soft gate weights, not an independent Fourier degree count."""
    return mask.sum(dim=(-2, -1)).squeeze(1)


def mask_entropy(mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = mask.flatten(1)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(eps)
    return -(probs * probs.clamp_min(eps).log()).sum(dim=1)
