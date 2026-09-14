"""
trainer.py — ЕДИНЫЙ тренер для всех семейств моделей.

Один класс сети, один оптимизатор, один планировщик (его нет), одинаковые
эпохи, батчи, ранняя остановка и сиды. Семейство модели меняет ТОЛЬКО состав
функции потерь и способ вычисления физической невязки — то есть ровно тот
фактор, который изучается. Это и есть ответ на замечание Reviewer #1 про
«partly different optimization strategies».

Семейства:
    mlp          data-loss only                        (датадривен-базлайн)
    pinn         сильная форма, невязка поточечно      (autograd по r)
    vpinn        слабая форма, проекция на тест-функции (квадратура Гаусса–Лежандра,
                 интегрирование по частям — производная сети не нужна вовсе)

Физика (см. publication/code/physics_check.py и ERRATA E-02):
    R₁(r) = ∂σ_rr/∂r + (σ_rr − σ_θθ)/r          — радиальное равновесие
Осевое уравнение НЕ используется. Причин две, и вторая тяжелее первой:
отброшенный ∂σ_zz/∂z даёт 68 % от удержанных членов, а оставшееся выражение
∂τ_rz/∂r + τ_rz/r тождественно равно (1/r)·∂(r·τ_rz)/∂r и потому означает не
«приближённое равновесие», а «τ_rz ≡ 0 на всём радиусе». Оно доступно
отдельным членом λ_shear, называется своим именем (_zero_shear) и по
умолчанию выключено.

ГУ: σ_rr(r=1) = τ_rz(r=1) = 0 — на СВОБОДНОЙ ПОВЕРХНОСТИ (ERRATA E-01),
а не при r = 1.5, как в моделях репозитория.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

N_R = 20
COMPONENTS = ("sigma_rr", "sigma_tt", "sigma_zz", "tau_rz")
IDX_SRR, IDX_STT, IDX_SZZ, IDX_TRZ = 0, 1, 2, 3
FAMILIES = ("mlp", "pinn", "vpinn")


# ────────────────────────────── конфиг ──────────────────────────────────

@dataclass
class Config:
    family: str = "pinn"
    n_layers: int = 4
    n_units: int = 128
    activation: str = "tanh"
    n_fourier: int = 4            # Fourier-эмбеддинг по r, одинаков у всех

    optimizer: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_sets: int = 64
    max_epochs: int = 400
    patience: int = 40
    val_frac: float = 0.15

    lambda_physics: float = 0.3      # доля нормы градиента data-члена, см. calibrate
    lambda_bc: float = 0.5
    calibrate_physics: bool = True
    lambda_shear: float = 0.0     # «сдвига нет вовсе»; НЕ осевое равновесие, см. _zero_shear
    n_quadrature: int = 32        # узлы Гаусса–Лежандра (vpinn)
    n_test_funcs: int = 8         # тест-функции sin(kπr) (vpinn)
    r_clip: float = 0.05          # клип 1/r у оси, в долях радиуса

    seed: int = 0
    device: str = "cpu"

    def as_dict(self) -> Dict:
        return asdict(self)


# ────────────────────────────── сеть ────────────────────────────────────

class FourierR(nn.Module):
    """sin/cos(2πk·r) по последней колонке входа; k = 1..n."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.register_buffer("k", torch.arange(1, n + 1, dtype=torch.float64) * 2.0 * math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n == 0:
            return x
        r = x[:, -1:]
        a = r * self.k
        return torch.cat([x, torch.sin(a), torch.cos(a)], dim=1)


ACTS = {"tanh": nn.Tanh, "selu": nn.SELU, "softplus": nn.Softplus}


class Net(nn.Module):
    def __init__(self, cfg: Config, n_in: int = 6, n_out: int = 4):
        super().__init__()
        self.emb = FourierR(cfg.n_fourier)
        d = n_in + 2 * cfg.n_fourier
        act = ACTS[cfg.activation]
        layers: List[nn.Module] = []
        for _ in range(cfg.n_layers):
            layers += [nn.Linear(d, cfg.n_units), act()]
            d = cfg.n_units
        layers += [nn.Linear(d, n_out)]
        self.body = nn.Sequential(*layers)
        for m in self.body:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(self.emb(x))


# ─────────────────────────── стандартизация ─────────────────────────────

@dataclass
class Scalers:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    @staticmethod
    def fit(X: np.ndarray, Y: np.ndarray) -> "Scalers":
        return Scalers(X.mean(0), X.std(0) + 1e-12, Y.mean(0), Y.std(0) + 1e-12)


# ──────────────────────────── физика ────────────────────────────────────

def _radial_residual(y_phys: torch.Tensor, dsrr_dr: torch.Tensor,
                     r_phys: torch.Tensor, r_clip_abs: torch.Tensor) -> torch.Tensor:
    """R₁ = ∂σ_rr/∂r + (σ_rr − σ_θθ)/r, физические единицы (МПа/м)."""
    s_rr = y_phys[..., IDX_SRR]
    s_tt = y_phys[..., IDX_STT]
    r_safe = torch.maximum(r_phys, r_clip_abs)
    return dsrr_dr + (s_rr - s_tt) / r_safe


def _gauss_legendre(n: int) -> Tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w          # отображение [-1,1] → [0,1]


# ──────────────────────────── обучение ──────────────────────────────────

@dataclass
class TrainResult:
    cfg: Dict
    epochs_run: int
    best_val: float
    history: List[float] = field(default_factory=list)
    phys_gain: float = 1.0


class Trainer:
    """
    Один экземпляр = одно обучение. Данные передаются наборами:
        proc (n,5), y (n,20,4) МПа, r (n,20) нормированный, r_phys (n,20) м
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.use_deterministic_algorithms(False)
        self.dtype = torch.float64
        self.net: Optional[Net] = None
        self.sc: Optional[Scalers] = None

    # ---- подготовка тензоров -------------------------------------------
    def _pack(self, proc, y, r, r_phys):
        n = proc.shape[0]
        X = np.concatenate([np.repeat(proc, N_R, 0), r.reshape(-1, 1)], 1)
        Y = y.reshape(n * N_R, 4)
        return X, Y

    def _to_t(self, a):
        return torch.tensor(a, dtype=self.dtype, device=self.cfg.device)

    # ---- потери ---------------------------------------------------------
    def _data_loss(self, pred_sc, tgt_sc):
        return ((pred_sc - tgt_sc) ** 2).mean()

    # ---- калибровка веса физики -----------------------------------------
    def _grad_norm(self, loss) -> float:
        """‖∂loss/∂θ‖₂ по параметрам сети, без обновления весов."""
        gs = torch.autograd.grad(loss, list(self.net.parameters()),
                                 retain_graph=False, allow_unused=True)
        return float(torch.sqrt(sum((g ** 2).sum() for g in gs if g is not None)))

    def _calibrate(self, pb, rb, rpb, rcb, yb_sc) -> float:
        """
        Множитель, при котором норма градиента физического члена равна норме
        градиента data-члена. После него `lambda_physics` читается одинаково у
        всех семейств: «физический градиент составляет λ от data-градиента».

        Зачем. Сильная и слабая формы дают невязки принципиально разной
        амплитуды: слабая — это проекция того же остатка на осциллирующие
        тест-функции, и она гасится примерно в 200 раз. При одинаковом
        номинальном λ слабый член оказывается численно инертным, и сравнение
        «сильная против слабой формы» вырождается в «физика включена против
        выключенной». Калибровка убирает эту подмену: сравниваются формы при
        равном влиянии на обучение, а не веса.
        """
        B = pb.shape[0]
        X = torch.cat([pb.unsqueeze(1).expand(B, N_R, 5).reshape(-1, 5),
                       rb.reshape(-1, 1)], 1)
        pred_sc = self.net((X - self.xm) / self.xs).reshape(B, N_R, 4)
        g_data = self._grad_norm(self._data_loss(pred_sc, yb_sc))
        if self.cfg.family == "pinn":
            l = self._physics_strong(pb, rb, rpb, rcb)
        else:
            l = self._physics_weak(pb, rpb, rcb)
        g_phys = self._grad_norm(l)
        if g_phys <= 1e-30:
            return 1.0
        return g_data / g_phys

    def _bc_loss(self, proc_b):
        """σ_rr = τ_rz = 0 на r = 1 (свободная поверхность)."""
        B = proc_b.shape[0]
        r1 = torch.ones((B, 1), dtype=self.dtype, device=self.cfg.device)
        Xb = torch.cat([proc_b, r1], 1)
        Xb_sc = (Xb - self.xm) / self.xs
        p_sc = self.net(Xb_sc)
        p = p_sc * self.ys + self.ym
        return ((p[:, IDX_SRR] / self.ys[0, IDX_SRR]) ** 2
                + (p[:, IDX_TRZ] / self.ys[0, IDX_TRZ]) ** 2).mean()

    def _physics_strong(self, proc_b, r_b, rphys_b, rclip_b):
        """Сильная форма: невязка поточечно, ∂/∂r через autograd."""
        B, nr = r_b.shape
        r_flat = r_b.reshape(-1, 1).clone().requires_grad_(True)
        p_rep = proc_b.unsqueeze(1).expand(B, nr, 5).reshape(-1, 5)
        X = torch.cat([p_rep, r_flat], 1)
        X_sc = (X - self.xm) / self.xs
        y_sc = self.net(X_sc)
        y_phys = y_sc * self.ys + self.ym
        dsrr_dnorm = torch.autograd.grad(
            y_phys[:, IDX_SRR].sum(), r_flat, create_graph=True)[0][:, 0]
        # цепное правило: r_norm = r_phys / R  ⇒  d/dr_phys = (1/R) d/dr_norm
        R = (rphys_b[:, -1:] / 1.0).reshape(-1, 1).expand(B, nr).reshape(-1)
        dsrr_dr = dsrr_dnorm / R
        res = _radial_residual(y_phys, dsrr_dr, rphys_b.reshape(-1),
                               rclip_b.reshape(-1))
        return torch.log1p((res / self.res_scale) ** 2).mean()

    def _physics_weak(self, proc_b, rphys_b, rclip_b):
        """
        Слабая форма. После интегрирования по частям с v_k = sin(kπr),
        v_k(0) = v_k(1) = 0:

            W_k = −∫₀¹ σ_rr (v_k + r v_k') dr + ∫₀¹ (σ_rr − σ_θθ) (r/r_safe) v_k dr

        Производная сети не вычисляется вообще — это и есть содержательное
        отличие слабой формы от сильной, а не другой способ её посчитать.
        """
        B = proc_b.shape[0]
        nq = self.qx.shape[0]
        rq = self.qx.reshape(1, nq).expand(B, nq)
        p_rep = proc_b.unsqueeze(1).expand(B, nq, 5).reshape(-1, 5)
        X = torch.cat([p_rep, rq.reshape(-1, 1)], 1)
        X_sc = (X - self.xm) / self.xs
        y = (self.net(X_sc) * self.ys + self.ym).reshape(B, nq, 4)
        s_rr, s_tt = y[..., IDX_SRR], y[..., IDX_STT]

        Rmax = rphys_b[:, -1:]                       # (B,1) м
        r_ph = rq * Rmax                             # (B,nq)
        r_safe = torch.maximum(r_ph, rclip_b[:, :1])
        w = self.qw.reshape(1, nq)

        # v_k и v_k' на узлах квадратуры: (K, nq)
        term1 = -(s_rr.unsqueeze(1) * (self.vk.unsqueeze(0) + rq.unsqueeze(1) * self.dvk.unsqueeze(0))
                  * w.unsqueeze(1)).sum(-1)          # (B,K)
        term2 = (((s_rr - s_tt) * rq * Rmax / r_safe).unsqueeze(1)
                 * self.vk.unsqueeze(0) * w.unsqueeze(1)).sum(-1)
        W = term1 + term2
        return torch.log1p((W / self.weak_scale) ** 2).mean()

    def _zero_shear(self, proc_b, r_b, rphys_b, rclip_b):
        """
        ∂τ_rz/∂r + τ_rz/r — то, что остаётся от осевого равновесия без ∂σ_zz/∂z.

        Тождественно (1/r)·∂(r·τ_rz)/∂r, поэтому обнуление означает r·τ_rz = const,
        а с регулярностью на оси — τ_rz ≡ 0 на всём радиусе. Это жёсткое условие
        «сдвига нет», а не ослабленное равновесие и не сглаживание. Данными не
        подтверждается (медианный относительный размах r·τ_rz равен 3.15), поэтому
        по умолчанию λ_shear = 0 и в прогонах статьи член не участвует.
        """
        B, nr = r_b.shape
        r_flat = r_b.reshape(-1, 1).clone().requires_grad_(True)
        p_rep = proc_b.unsqueeze(1).expand(B, nr, 5).reshape(-1, 5)
        X_sc = (torch.cat([p_rep, r_flat], 1) - self.xm) / self.xs
        y = self.net(X_sc) * self.ys + self.ym
        dtau = torch.autograd.grad(y[:, IDX_TRZ].sum(), r_flat, create_graph=True)[0][:, 0]
        R = rphys_b[:, -1:].expand(B, nr).reshape(-1)
        r_safe = torch.maximum(rphys_b.reshape(-1), rclip_b.reshape(-1))
        res = dtau / R + y[:, IDX_TRZ] / r_safe
        return torch.log1p((res / self.shear_scale) ** 2).mean()

    # ---- главный цикл ---------------------------------------------------
    def fit(self, proc, y, r, r_phys, verbose: bool = False) -> TrainResult:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        n = proc.shape[0]
        perm = rng.permutation(n)
        n_val = max(1, int(round(n * cfg.val_frac)))
        val_sets, tr_sets = perm[:n_val], perm[n_val:]

        Xtr, Ytr = self._pack(proc[tr_sets], y[tr_sets], r[tr_sets], r_phys[tr_sets])
        self.sc = Scalers.fit(Xtr, Ytr)
        self.xm = self._to_t(self.sc.x_mean.reshape(1, -1))
        self.xs = self._to_t(self.sc.x_std.reshape(1, -1))
        self.ym = self._to_t(self.sc.y_mean.reshape(1, -1))
        self.ys = self._to_t(self.sc.y_std.reshape(1, -1))

        # характерные масштабы невязок: σ_std / R. Без них log1p(res²/scale)
        # уводит физлосс на пять порядков ниже data-loss, и физика фактически
        # выключается независимо от λ (эта же ошибка была в отклонённой версии).
        R_char = float(np.mean(r_phys[tr_sets][:, -1]))
        sd = self.sc.y_std
        self.res_scale = self._to_t(np.array((sd[IDX_SRR] + sd[IDX_STT]) / R_char))
        self.weak_scale = self._to_t(np.array(sd[IDX_SRR] + sd[IDX_STT]))
        self.shear_scale = self._to_t(np.array(sd[IDX_TRZ] / R_char))
        self.R_char = R_char

        self.net = Net(cfg).to(dtype=self.dtype, device=cfg.device)

        if cfg.family == "vpinn":
            qx, qw = _gauss_legendre(cfg.n_quadrature)
            self.qx = self._to_t(qx)
            self.qw = self._to_t(qw)
            ks = torch.arange(1, cfg.n_test_funcs + 1, dtype=self.dtype,
                              device=cfg.device).reshape(-1, 1) * math.pi
            self.vk = torch.sin(ks * self.qx.reshape(1, -1))
            self.dvk = ks * torch.cos(ks * self.qx.reshape(1, -1))

        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.learning_rate,
                                weight_decay=cfg.weight_decay)

        P_tr = self._to_t(proc[tr_sets])
        Y_tr = self._to_t(y[tr_sets])
        R_tr = self._to_t(r[tr_sets])
        RP_tr = self._to_t(r_phys[tr_sets])
        RC_tr = RP_tr[:, -1:].expand_as(RP_tr) * cfg.r_clip
        Ytr_sc = (Y_tr - self.ym.reshape(1, 1, 4)) / self.ys.reshape(1, 1, 4)

        best, bad, hist = float("inf"), 0, []
        best_state = None
        n_tr = len(tr_sets)
        g = torch.Generator().manual_seed(cfg.seed)

        self.phys_gain = 1.0
        if cfg.family in ("pinn", "vpinn") and cfg.calibrate_physics and cfg.lambda_physics > 0:
            idx0 = torch.arange(min(cfg.batch_sets, n_tr))
            self.phys_gain = self._calibrate(P_tr[idx0], R_tr[idx0], RP_tr[idx0],
                                             RC_tr[idx0], Ytr_sc[idx0])

        for epoch in range(cfg.max_epochs):
            self.net.train()
            order = torch.randperm(n_tr, generator=g)
            for s in range(0, n_tr, cfg.batch_sets):
                idx = order[s:s + cfg.batch_sets]
                pb, rb, rpb, rcb = P_tr[idx], R_tr[idx], RP_tr[idx], RC_tr[idx]
                yb_sc = Ytr_sc[idx]
                B = pb.shape[0]

                X = torch.cat([pb.unsqueeze(1).expand(B, N_R, 5).reshape(-1, 5),
                               rb.reshape(-1, 1)], 1)
                pred_sc = self.net((X - self.xm) / self.xs).reshape(B, N_R, 4)
                loss = self._data_loss(pred_sc, yb_sc)

                if cfg.family in ("pinn", "vpinn"):
                    loss = loss + cfg.lambda_bc * self._bc_loss(pb)
                    w_phys = cfg.lambda_physics * self.phys_gain
                    if cfg.family == "pinn":
                        loss = loss + w_phys * self._physics_strong(pb, rb, rpb, rcb)
                    else:
                        loss = loss + w_phys * self._physics_weak(pb, rpb, rcb)
                    if cfg.lambda_shear > 0:
                        loss = loss + cfg.lambda_shear * self._zero_shear(pb, rb, rpb, rcb)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
                opt.step()

            v = self.evaluate(proc[val_sets], y[val_sets], r[val_sets])["macro_rmse"]
            hist.append(v)
            if v < best - 1e-9:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
            if verbose and epoch % 25 == 0:
                print(f"  epoch {epoch:4d}  val macro-RMSE {v:.4f}  (best {best:.4f})")

        if best_state is not None:
            self.net.load_state_dict(best_state)
        res = TrainResult(cfg.as_dict(), len(hist), best, hist)
        res.phys_gain = float(self.phys_gain)
        return res

    # ---- предсказание и метрики -----------------------------------------
    @torch.no_grad()
    def predict(self, proc, r) -> np.ndarray:
        self.net.eval()
        n = proc.shape[0]
        X = np.concatenate([np.repeat(proc, N_R, 0), r.reshape(-1, 1)], 1)
        Xs = (self._to_t(X) - self.xm) / self.xs
        p = self.net(Xs) * self.ys + self.ym
        return p.cpu().numpy().reshape(n, N_R, 4)

    def evaluate(self, proc, y, r, global_std=None) -> Dict[str, float]:
        return metrics(y, self.predict(proc, r), global_std)

    @torch.no_grad()
    def predict_at(self, proc, r_scalar: float) -> np.ndarray:
        """Предсказание в одной радиальной точке — для аудита ГУ."""
        self.net.eval()
        n = proc.shape[0]
        X = np.concatenate([proc, np.full((n, 1), r_scalar)], 1)
        Xs = (self._to_t(X) - self.xm) / self.xs
        return (self.net(Xs) * self.ys + self.ym).cpu().numpy()


# ──────────────────────────── метрики ───────────────────────────────────

def metrics(y_true: np.ndarray, y_pred: np.ndarray,
            global_std: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    y_*: (n_sets, 20, 4) МПа.

    macro_rmse — среднее RMSE по четырём компонентам (как в статье).

    Нормированные величины считаются ДВАЖДЫ и это принципиально:

      *_local  — знаменатель = std компоненты В ЭТОМ ЖЕ тестовом срезе;
      *_global — знаменатель = std по всему датасету (аргумент global_std).

    Локальный вариант нельзя сравнивать между сплитами: у разных held-out
    регионов разная дисперсия, и одна и та же абсолютная ошибка даёт разный
    R². На τ_rz это давало ложный вывод: локальный R² уходил в −4.4 и,
    попадая в среднее по четырём компонентам, утаскивал macro-R² ниже нуля,
    из-за чего казалось, что модель хуже предсказания средним. На деле по
    macro-RMSE она лучше среднего в 2.8–5.9 раза на всех сплитах.

    macro_r2 оставлено равным локальному — так считала отклонённая версия,
    и совместимость нужна для сверки; ОТЧЁТ обязан печатать глобальный.
    """
    t = y_true.reshape(-1, 4)
    p = y_pred.reshape(-1, 4)
    out: Dict[str, float] = {}
    rmses, nrmses, r2s = [], [], []
    for c, name in enumerate(COMPONENTS):
        e = p[:, c] - t[:, c]
        rmse = float(np.sqrt(np.mean(e ** 2)))
        sd = float(t[:, c].std()) + 1e-12
        ss_res = float((e ** 2).sum())
        ss_tot = float(((t[:, c] - t[:, c].mean()) ** 2).sum()) + 1e-12
        out[f"rmse_{name}"] = rmse
        out[f"nrmse_{name}"] = rmse / sd
        out[f"r2_{name}"] = 1.0 - ss_res / ss_tot
        rmses.append(rmse); nrmses.append(rmse / sd); r2s.append(1.0 - ss_res / ss_tot)
    out["macro_rmse"] = float(np.mean(rmses))
    out["macro_nrmse"] = float(np.mean(nrmses))
    out["macro_r2"] = float(np.mean(r2s))
    out["rmse_normal_only"] = float(np.mean(rmses[:3]))
    out["r2_normal_only"] = float(np.mean(r2s[:3]))
    if global_std is not None:
        gs = np.asarray(global_std, dtype=float) + 1e-12
        g_nr = np.asarray(rmses) / gs
        g_r2 = 1.0 - g_nr ** 2
        for c, name in enumerate(COMPONENTS):
            out[f"nrmse_{name}_global"] = float(g_nr[c])
            out[f"r2_{name}_global"] = float(g_r2[c])
        out["macro_nrmse_global"] = float(np.mean(g_nr))
        out["macro_r2_global"] = float(np.mean(g_r2))
        out["r2_normal_only_global"] = float(np.mean(g_r2[:3]))
    return out


def equilibrium_audit(y_sets: np.ndarray, r_phys: np.ndarray,
                      r_clip: float = 0.05) -> Dict[str, float]:
    """
    Апостериорная невязка радиального равновесия по профилям (МПа/м),
    считается ровно тем же выражением, что в лоссе. Годится и для предсказаний
    модели, и для самих данных FEM — базовую линию источника печатать обязательно.
    """
    n = y_sets.shape[0]
    med = np.zeros(n)
    for i in range(n):
        s_rr, s_tt = y_sets[i, :, IDX_SRR], y_sets[i, :, IDX_STT]
        rp = r_phys[i]
        d = np.gradient(s_rr, rp)
        r_safe = np.maximum(rp, rp[-1] * r_clip)
        res = d + (s_rr - s_tt) / r_safe
        med[i] = np.median(np.abs(res[1:-1]))
    return {"eq_res_median": float(np.median(med)),
            "eq_res_mean": float(np.mean(med))}


def bc_audit(y_surface: np.ndarray) -> Dict[str, float]:
    """|σ_rr| и |τ_rz| на свободной поверхности, МПа. y_surface: (n, 4)."""
    return {"bc_sigma_rr": float(np.abs(y_surface[:, IDX_SRR]).mean()),
            "bc_tau_rz": float(np.abs(y_surface[:, IDX_TRZ]).mean())}
