"""
Модульные тесты физики (ТЗ, п. 9): разностные операторы, hard-блок, невязка, разбиение.

    python tests/test_physics.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from models import build  # noqa: E402
from physics import Grid, d1, d1_eval, d2, hard_fields, profile_fields, residuals, section_force  # noqa: E402
from train import make_split  # noqa: E402

torch.set_default_dtype(torch.float64)


def smooth_NM(g: Grid):
    rho = torch.as_tensor(g.rho)[:, None]
    zeta = torch.as_tensor(g.zeta)[None, :]
    N = 100 * (1 + 0.5 * rho ** 2) * torch.cos(0.7 * zeta) - 40 * rho
    M = -150 * (1 + 0.3 * rho) + 30 * torch.sin(0.9 * zeta) * (1 - 0.5 * rho ** 2)
    return N[None], M[None]


def test_operators():
    g = Grid(16, 24, 2.0)
    x = torch.as_tensor(g.zeta)[None, :].expand(16, 24)
    assert torch.allclose(d1(3 * x ** 2 - x, g.hz, -1), 6 * x - 1, atol=1e-9)
    assert torch.allclose(d2(3 * x ** 2 - x, g.hz, -1), torch.full_like(x, 6.0), atol=1e-7)
    xe = g.zeta[None, :].repeat(16, 0)
    assert np.allclose(d1_eval(xe ** 4, g.hz, -1), 4 * xe ** 3, atol=1e-8)
    print("  ✓ операторы точны на многочленах своего порядка")


def test_hard_bc_random():
    g = Grid(32, 64, 2.0)
    N, M = torch.randn(3, 32, 64), torch.randn(3, 32, 64)
    y = hard_fields(N, M, g)
    assert torch.allclose(y[:, 0, -1], torch.zeros(3, 64))           # σ_r(1) = 0
    assert torch.allclose(y[:, 3, -1], torch.zeros(3, 64))           # τ_rz(1) = 0
    assert torch.allclose(y[:, 3, 0], torch.zeros(3, 64))            # τ_rz(0) = 0
    assert torch.allclose(y[:, 0, 0], y[:, 1, 0])                    # σ_r = σ_θ на оси
    print("  ✓ hard: ГУ на поверхности и на оси точно для случайных N, M")


def test_hard_equilibrium_converges():
    errs = []
    for nr, nz in ((16, 32), (32, 64), (64, 128)):
        g = Grid(nr, nz, 2.0)
        y = hard_fields(*smooth_NM(g), g).numpy()
        R1, R2 = residuals(y, g)
        F = section_force(y, g)
        errs.append((np.sqrt(np.mean(R1 ** 2)), np.sqrt(np.mean(R2 ** 2)), np.abs(F).max()))
    for k in range(3):
        assert errs[2][k] < errs[1][k] < errs[0][k], errs
    assert errs[1][0] < 1.0 and errs[1][1] < 1.0 and errs[1][2] < 0.5, errs
    print(f"  ✓ hard: невязка и осевая сила убывают с сеткой {[tuple(round(v, 3) for v in e) for e in errs]}")


def test_profile1d():
    g = Grid(32, 64, 2.0)
    y = profile_fields(torch.randn(2, 32), torch.randn(2, 32), g).numpy()
    assert np.allclose(y[:, 0, -1], 0) and np.allclose(y[:, 3], 0)
    assert np.allclose(y[:, 0, 0], y[:, 1, 0])
    assert np.abs(section_force(y, g)).max() < 1e-9
    print("  ✓ 1D: σ_r(1) = 0, τ_rz ≡ 0, σ_r = σ_θ на оси, нулевая осевая сила")


def test_models_shapes():
    g = Grid(32, 64, 2.0)
    x = torch.randn(5, 6, dtype=torch.float32)
    for v in ("data", "hard", "profile1d"):
        m = build(v, g).float()
        y = m(x)
        assert y.shape == (5, 4, 32, 64), (v, y.shape)
        if v == "hard":
            assert torch.allclose(y[:, 0, -1], torch.zeros_like(y[:, 0, -1]), atol=1e-6)
    print("  ✓ модели дают (B, 4, 32, 64), hard-модель держит σ_r(1) = 0")


def test_split_groups():
    rng = np.random.default_rng(0)
    P = np.array([[q, k, a, mu, v] for q in (0.05, 0.1) for k in (0, 0.5) for a in (4, 8, 12)
                  for v in (10, 40) for mu in (0.025, 0.05, 0.1) if rng.random() > 0.2])
    tr, va, te = make_split(P, 0)
    assert len(set(tr) | set(va) | set(te)) == len(P) and not (set(tr) & set(te))
    fam = lambda i: tuple(P[i, [0, 1, 2, 4]])  # noqa: E731
    sets = [set(map(fam, s)) for s in (tr, va, te)]
    assert not (sets[0] & sets[2]) and not (sets[0] & sets[1]) and not (sets[1] & sets[2])
    print(f"  ✓ разбиение: семейства по μ не разрываются ({len(tr)}/{len(va)}/{len(te)})")


if __name__ == "__main__":
    for t in (test_operators, test_hard_bc_random, test_hard_equilibrium_converges,
              test_profile1d, test_models_shapes, test_split_groups):
        t()
    print("все тесты прошли")
