"""
models.py — архитектура A (ТЗ, п. 4): режим → поля через сверточный декодер с FiLM,
и одномерная базовая модель (профиль по ρ с hard-ограничениями по прил. А.8).

Декодер: MLP-энкодер 3×64 (SiLU) → вектор условия t; затравочная карта C × n_r/8 × n_z/8;
3 блока «upsample ×2 → conv 3×3 → GroupNorm → FiLM → SiLU» и 1 такой же блок на полном
разрешении без upsample; выходной conv 1×1. В каждый conv подаются каналы координат ρ, ζ
(CoordConv). Upsample стоит перед свёрткой, а не последней операцией блока: иначе выход
кусочно-линеен и M_ζζ в σ_θ даёт шахматный узор (вопрос Т2).
Padding: на оси (ρ = 0) — reflect, на поверхности и по краям ζ — replicate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from physics import Grid, hard_fields, profile_fields


def mlp(n_in: int, width: int, depth: int) -> nn.Sequential:
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    return nn.Sequential(*layers)


def pad_rz(x: torch.Tensor) -> torch.Tensor:
    x = F.pad(x, (0, 0, 1, 0), mode="reflect")       # ось ρ = 0
    x = F.pad(x, (0, 0, 0, 1), mode="replicate")     # поверхность ρ = 1
    return F.pad(x, (1, 1, 0, 0), mode="replicate")  # края по ζ


def coords(x: torch.Tensor) -> torch.Tensor:
    B, _, nr, nz = x.shape
    rho = torch.linspace(0, 1, nr, device=x.device, dtype=x.dtype)[:, None].expand(nr, nz)
    zeta = torch.linspace(0, 1, nz, device=x.device, dtype=x.dtype)[None, :].expand(nr, nz)
    return torch.cat([x, torch.stack([rho, zeta])[None].expand(B, -1, -1, -1)], dim=1)


class FiLMBlock(nn.Module):
    def __init__(self, c: int, t_dim: int, up: bool, groups: int = 8):
        super().__init__()
        self.up = up
        self.conv = nn.Conv2d(c + 2, c, 3)
        self.norm = nn.GroupNorm(groups, c)
        self.film = nn.Linear(t_dim, 2 * c)

    def forward(self, x, t):
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.norm(self.conv(pad_rz(coords(x))))
        gamma, beta = self.film(t).chunk(2, dim=-1)
        x = x * (1 + gamma[..., None, None]) + beta[..., None, None]
        return F.silu(x)


class FiLMDecoder(nn.Module):
    """Архитектура A. variant='hard' — 2 канала (N, M), 'data' — 4 канала (σ напрямую)."""

    def __init__(self, grid: Grid, variant: str, n_feat: int = 6, width: int = 64, t_dim: int = 64):
        super().__init__()
        assert grid.nr % 8 == 0 and grid.nz % 8 == 0, "сетка должна быть кратна 8"
        assert variant in ("hard", "data")
        self.grid, self.variant, self.width = grid, variant, width
        self.enc = mlp(n_feat, t_dim, 3)
        self.seed_shape = (width, grid.nr // 8, grid.nz // 8)
        self.seed = nn.Linear(t_dim, width * (grid.nr // 8) * (grid.nz // 8))
        self.blocks = nn.ModuleList([FiLMBlock(width, t_dim, up=True) for _ in range(3)]
                                    + [FiLMBlock(width, t_dim, up=False)])
        self.out = nn.Conv2d(width + 2, 2 if variant == "hard" else 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_feat) стандартизованные признаки → (B, 4, n_r, n_z) в единицах масштаба."""
        t = self.enc(x)
        h = self.seed(t).view(-1, *self.seed_shape)
        for blk in self.blocks:
            h = blk(h, t)
        o = self.out(coords(h))
        if self.variant == "hard":
            return hard_fields(o[:, 0], o[:, 1], self.grid)
        return o


class Profile1D(nn.Module):
    """Одномерная базовая модель: режим → N(ρ), S(ρ) → профиль с hard-ограничениями (прил. А.8)."""

    def __init__(self, grid: Grid, n_feat: int = 6, width: int = 128, depth: int = 3):
        super().__init__()
        self.grid = grid
        self.net = nn.Sequential(mlp(n_feat, width, depth), nn.Linear(width, 2 * grid.nr))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.net(x).view(-1, 2, self.grid.nr)
        return profile_fields(o[:, 0], o[:, 1], self.grid)


def build(variant: str, grid: Grid, width: int = 64) -> nn.Module:
    if variant in ("hard", "data"):
        return FiLMDecoder(grid, variant, width=width)
    if variant == "profile1d":
        return Profile1D(grid)
    raise ValueError(f"неизвестный вариант {variant!r}")
