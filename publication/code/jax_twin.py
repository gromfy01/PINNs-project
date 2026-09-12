"""
jax_twin.py — тот же PINN на JAX/Flax-подобном стеке, для проверки замечания
Reviewer #2 про «difference in autodiff backend».

Вопрос «важен ли бэкенд» распадается на два, и путать их нельзя.

**1. Считают ли движки одно и то же?** Это проверяется прямо:
`check_gradient_agreement()` строит одну и ту же сеть с одними и теми же весами
в torch и в JAX и сравнивает `∂σ_rr/∂r`. Ответ — совпадают до 8.7e-15 по модулю
(9.8e-16 относительно), то есть до машинного нуля. Разнице в метриках нечем
взяться со стороны автодифференцирования.

**2. Различаются ли две независимые реализации сильнее, чем прогоны одной?**
Это и есть содержательный вопрос, и здесь он ставится честно: обе реализации
делят архитектуру, протокол, данные, разбиение на train/val и гиперпараметры, но
начальные веса и порядок батчей берут из СВОИХ генераторов (torch —
`torch.manual_seed`, JAX-версия — `numpy.random.default_rng`). Каждая гоняется на
пяти сидах, и сравниваются распределения. Требовать побитово одинаковой
инициализации здесь не нужно и даже вредно: это свело бы эксперимент к проверке
детерминизма, а не к вопросу «отличима ли смена движка от разброса перезапусков»
— именно того, на который в отклонённой версии ответа не было.

Отдельно: в репозитории «третий бэкенд», DeepXDE, сконфигурирован как
`DDE_BACKEND=pytorch` и использует `dde.grad.jacobian`, который под этим
бэкендом вызывает `torch.autograd.grad`. То есть это не третий движок, а тот
же PyTorch за другим API (ERRATA E-19).

"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

from trainer import (COMPONENTS, IDX_SRR, IDX_STT, IDX_SZZ, IDX_TRZ, N_R,  # noqa: E402
                     Config, Scalers, TrainResult, metrics)


def init_params(cfg: Config, n_in: int = 6, n_out: int = 4) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Glorot normal из numpy-RNG, засеянного cfg.seed. Это НЕ тот же поток, что у
    torch.manual_seed, и так и задумано: см. пункт 2 в докстринге модуля.
    Для check_gradient_agreement веса из этой функции копируются в torch-сеть,
    так что там сравнение идёт при побитово одинаковых весах.
    """
    rng = np.random.default_rng(cfg.seed)
    d = n_in + 2 * cfg.n_fourier
    dims = [d] + [cfg.n_units] * cfg.n_layers + [n_out]
    ps = []
    for i in range(len(dims) - 1):
        fan_in, fan_out = dims[i], dims[i + 1]
        std = math.sqrt(2.0 / (fan_in + fan_out))
        ps.append((rng.normal(0.0, std, size=(fan_in, fan_out)), np.zeros(fan_out)))
    return ps


def fourier(x, n):
    if n == 0:
        return x
    r = x[:, -1:]
    k = jnp.arange(1, n + 1, dtype=x.dtype) * 2.0 * jnp.pi
    a = r * k
    return jnp.concatenate([x, jnp.sin(a), jnp.cos(a)], axis=1)


def forward(params, x, n_fourier):
    h = fourier(x, n_fourier)
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return h @ W + b


class JaxTrainer:
    """Публичный интерфейс совпадает с trainer.Trainer."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fit(self, proc, y, r, r_phys, verbose: bool = False) -> TrainResult:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        n = proc.shape[0]
        perm = rng.permutation(n)
        n_val = max(1, int(round(n * cfg.val_frac)))
        val_sets, tr_sets = perm[:n_val], perm[n_val:]

        X = np.concatenate([np.repeat(proc[tr_sets], N_R, 0), r[tr_sets].reshape(-1, 1)], 1)
        Y = y[tr_sets].reshape(-1, 4)
        self.sc = Scalers.fit(X, Y)
        xm, xs = self.sc.x_mean[None, :], self.sc.x_std[None, :]
        ym, ys = self.sc.y_mean[None, :], self.sc.y_std[None, :]
        R_char = float(np.mean(r_phys[tr_sets][:, -1]))
        res_scale = (self.sc.y_std[IDX_SRR] + self.sc.y_std[IDX_STT]) / R_char

        P = jnp.asarray(proc[tr_sets]); Rn = jnp.asarray(r[tr_sets])
        RP = jnp.asarray(r_phys[tr_sets])
        Ysc = jnp.asarray((y[tr_sets] - ym.reshape(1, 1, 4)) / ys.reshape(1, 1, 4))
        xm_j, xs_j = jnp.asarray(xm), jnp.asarray(xs)
        ym_j, ys_j = jnp.asarray(ym), jnp.asarray(ys)
        nf = cfg.n_fourier
        lam_p, lam_bc = cfg.lambda_physics, cfg.lambda_bc
        use_phys = cfg.family == "pinn"
        rclip = cfg.r_clip

        def net(params, Xraw):
            return forward(params, (Xraw - xm_j) / xs_j, nf)

        def loss_fn(params, pb, rb, rpb, yb_sc):
            B = pb.shape[0]
            Xd = jnp.concatenate(
                [jnp.broadcast_to(pb[:, None, :], (B, N_R, 5)).reshape(-1, 5),
                 rb.reshape(-1, 1)], axis=1)
            pred = net(params, Xd).reshape(B, N_R, 4)
            l = jnp.mean((pred - yb_sc) ** 2)
            if use_phys:
                r1 = jnp.ones((B, 1), dtype=pb.dtype)
                pb_sc = net(params, jnp.concatenate([pb, r1], 1)) * ys_j + ym_j
                l = l + lam_bc * jnp.mean((pb_sc[:, IDX_SRR] / ys_j[0, IDX_SRR]) ** 2
                                          + (pb_sc[:, IDX_TRZ] / ys_j[0, IDX_TRZ]) ** 2)

                def srr_of_r(rv, pv):
                    xx = jnp.concatenate([pv, rv.reshape(1)])[None, :]
                    return (net(params, xx) * ys_j + ym_j)[0, IDX_SRR]

                p_rep = jnp.broadcast_to(pb[:, None, :], (B, N_R, 5)).reshape(-1, 5)
                r_flat = rb.reshape(-1)
                d_srr = jax.vmap(jax.grad(srr_of_r))(r_flat, p_rep)
                Xf = jnp.concatenate([p_rep, r_flat.reshape(-1, 1)], 1)
                yph = net(params, Xf) * ys_j + ym_j
                Rmax = jnp.broadcast_to(rpb[:, -1:], (B, N_R)).reshape(-1)
                r_ph = rpb.reshape(-1)
                r_safe = jnp.maximum(r_ph, Rmax * rclip)
                res = d_srr / Rmax + (yph[:, IDX_SRR] - yph[:, IDX_STT]) / r_safe
                l = l + lam_p * jnp.mean(jnp.log1p((res / res_scale) ** 2))
            return l

        # Скейлеры и параметры эмбеддинга проставляются ДО цикла: внутри цикла
        # вызывается self.evaluate() для ранней остановки, а он ходит через
        # self._net и падал бы на отсутствующих атрибутах.
        self._nf = nf
        self._xm, self._xs, self._ym, self._ys = xm_j, xs_j, ym_j, ys_j

        params = [(jnp.asarray(W), jnp.asarray(b)) for W, b in init_params(cfg)]
        tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip),
                         optax.adamw(cfg.learning_rate, weight_decay=cfg.weight_decay))
        opt_state = tx.init(params)
        grad_fn = jax.jit(jax.value_and_grad(loss_fn))

        @jax.jit
        def step(params, opt_state, pb, rb, rpb, yb):
            _l, g = grad_fn(params, pb, rb, rpb, yb)
            upd, opt_state = tx.update(g, opt_state, params)
            return optax.apply_updates(params, upd), opt_state

        best, bad, hist, best_p = float("inf"), 0, [], None
        n_tr = len(tr_sets)
        gen = np.random.default_rng(cfg.seed + 10_000)
        for epoch in range(cfg.max_epochs):
            order = gen.permutation(n_tr)
            for s in range(0, n_tr, cfg.batch_sets):
                idx = jnp.asarray(order[s:s + cfg.batch_sets])
                params, opt_state = step(params, opt_state, P[idx], Rn[idx], RP[idx], Ysc[idx])
            self.params = params
            v = self.evaluate(proc[val_sets], y[val_sets], r[val_sets])["macro_rmse"]
            hist.append(v)
            if v < best - 1e-9:
                best, bad, best_p = v, 0, [(np.asarray(W), np.asarray(b)) for W, b in params]
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_p is not None:
            params = [(jnp.asarray(W), jnp.asarray(b)) for W, b in best_p]
        self.params = params
        return TrainResult(cfg.as_dict(), len(hist), best, hist)

    def _net(self, Xraw):
        return forward(self.params, (jnp.asarray(Xraw) - self._xm) / self._xs, self._nf)

    def predict(self, proc, r) -> np.ndarray:
        n = proc.shape[0]
        X = np.concatenate([np.repeat(proc, N_R, 0), r.reshape(-1, 1)], 1)
        p = self._net(X) * self._ys + self._ym
        return np.asarray(p).reshape(n, N_R, 4)

    def predict_at(self, proc, r_scalar: float) -> np.ndarray:
        n = proc.shape[0]
        X = np.concatenate([proc, np.full((n, 1), r_scalar)], 1)
        return np.asarray(self._net(X) * self._ys + self._ym)

    def evaluate(self, proc, y, r, global_std=None) -> Dict[str, float]:
        return metrics(y, self.predict(proc, r), global_std)


def check_gradient_agreement(cfg: Config, n_points: int = 64) -> Dict[str, float]:
    """
    Прямая проверка: ∂σ_rr/∂r, посчитанная torch.autograd и jax.grad на ОДНИХ И
    ТЕХ ЖЕ весах и входах, должна совпадать до машинной точности. Если это так,
    разница в метриках между «бэкендами» не может быть отнесена на автодифф.
    """
    import torch
    from trainer import Net

    ps = init_params(cfg)
    net = Net(cfg).to(dtype=torch.float64)
    with torch.no_grad():
        lin = [m for m in net.body if isinstance(m, torch.nn.Linear)]
        for m, (W, b) in zip(lin, ps):
            m.weight.copy_(torch.tensor(W.T)); m.bias.copy_(torch.tensor(b))

    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_points, 6))
    xt = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    yt = net(xt)[:, IDX_SRR].sum()
    gt = torch.autograd.grad(yt, xt)[0][:, 5].detach().numpy()

    def f(xrow):
        return forward([(jnp.asarray(W), jnp.asarray(b)) for W, b in ps],
                       xrow[None, :], cfg.n_fourier)[0, IDX_SRR]

    gj = np.asarray(jax.vmap(jax.grad(f))(jnp.asarray(X))[:, 5])
    fwd_t = net(torch.tensor(X, dtype=torch.float64)).detach().numpy()
    fwd_j = np.asarray(forward([(jnp.asarray(W), jnp.asarray(b)) for W, b in ps],
                               jnp.asarray(X), cfg.n_fourier))
    return {"max_abs_diff_forward": float(np.abs(fwd_t - fwd_j).max()),
            "max_abs_diff_grad": float(np.abs(gt - gj).max()),
            "max_rel_diff_grad": float(np.abs(gt - gj).max() / (np.abs(gt).max() + 1e-300))}
