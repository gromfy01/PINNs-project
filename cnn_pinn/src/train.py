"""
train.py — обучение одного варианта по конфигу (ТЗ, п. 7 и п. 9).

    python src/train.py --config configs/hard.yaml [--seed 1] [--out runs/hard_s1]

Варианты: 'data' (только данные, 4 канала), 'hard' (потенциалы N, M), 'profile1d'
(одномерная базовая модель). Функция потерь — MSE в единицах общего масштаба напряжений
(одно число для всех компонент, п. 3 шаг 4). Ранняя остановка по macro-RMSE на валидации.
В каталоге прогона сохраняются конфиг, git-хеш, время, веса, метрики и предсказания на тесте.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch
import yaml

from evaluate import evaluate
from models import build
from physics import Grid

HERE = os.path.dirname(os.path.abspath(__file__))


def make_split(P: np.ndarray, seed: int, frac=(0.70, 0.15, 0.15)):
    """
    Делятся режимы, а не точки (п. 3 шаг 6). Семейства, отличающиеся только μ, целиком
    попадают в одну выборку. Стратификация по ячейкам (Q_d, α): в каждой ячейке семейства
    перемешиваются и раскладываются по долям, так что каждый уровень есть во всех выборках.
    P: (n, 5) = (Q_d, k, α, μ, v). Возвращает индексы train, val, test.
    """
    rng = np.random.default_rng(seed)
    fam = {}
    for i, (q, k, a, _mu, v) in enumerate(P):
        fam.setdefault((q, k, a, v), []).append(i)
    cells = {}
    for key, idx in fam.items():
        cells.setdefault((key[0], key[2]), []).append(idx)
    tr, va, te = [], [], []
    for groups in cells.values():
        order = rng.permutation(len(groups))
        n = sum(len(groups[j]) for j in order)
        acc = 0
        for j in order:
            share = (acc + 0.5 * len(groups[j])) / n          # середина вклада семейства
            (te if share < frac[2] else va if share < frac[2] + frac[1] else tr).extend(groups[j])
            acc += len(groups[j])
    tr, va, te = np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))
    got = np.array([len(tr), len(va), len(te)]) / len(P)
    assert np.all(np.abs(got - np.array(frac)[[0, 1, 2]]) < 0.02) or len(P) < 500, got
    return tr, va, te


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    if a.seed is not None:
        cfg["seed"] = a.seed
    out = a.out or os.path.join("runs", f"{cfg['name']}_s{cfg['seed']}")
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    d = np.load(cfg["data"], allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    Y, X, P = d["Y"].astype(np.float32), d["X"].astype(np.float32), d["P"]
    grid = Grid(*meta["grid"], meta["window_R"])
    tr, va, te = make_split(P, cfg.get("split_seed", 0))
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12                # только по обучающей
    Xs = (X - mu) / sd
    scale = float(cfg["scale"])

    model = build(cfg["variant"], grid, width=cfg.get("width", 64))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps = int(np.ceil(len(tr) / cfg["batch"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps)
    xt, yt = torch.from_numpy(Xs), torch.from_numpy(Y / scale)

    def predict(idx):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(xt[idx[i:i + 64]]).numpy() for i in range(0, len(idx), 64)]) * scale

    def val_rmse():
        e = predict(va) - Y[va]
        return float(np.mean(np.sqrt(np.mean(e ** 2, axis=(0, 2, 3)))))

    best, best_ep, best_state, hist = np.inf, -1, None, []
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.from_numpy(np.random.permutation(tr))
        tot = 0.0
        for i in range(0, len(perm), cfg["batch"]):
            b = perm[i:i + cfg["batch"]]
            loss = torch.mean((model(xt[b]) - yt[b]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        v = val_rmse()
        hist.append({"epoch": ep, "train_loss": tot / len(tr), "val_rmse_macro": v,
                     "lr": opt.param_groups[0]["lr"]})
        if v < best:
            best, best_ep = v, ep
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        if ep % 10 == 0:
            print(f"ep {ep:3d}  loss {tot / len(tr):.3e}  val {v:7.2f} MPa  best {best:7.2f} @ {best_ep}", flush=True)
        if ep - best_ep >= cfg["patience"]:
            break
    train_s = time.time() - t0
    model.load_state_dict(best_state)

    t1 = time.time()
    pred = predict(te)
    pred_ms = (time.time() - t1) / len(te) * 1e3
    flags = {"axial_rms_gt10": d["axial_rms"][te] > 10, "thinning_gt0.1": d["thinning"][te] > 0.1}
    met = evaluate(pred, Y[te], grid, Y[tr].mean(0), flags)
    met.update(best_epoch=best_ep, epochs_run=len(hist), val_rmse_macro=best, train_seconds=train_s,
               predict_ms_per_regime=pred_ms, n_params=sum(p.numel() for p in model.parameters()),
               n_train=len(tr), n_val=len(va), n_test=len(te),
               lr_at_best=float(hist[best_ep]["lr"]))
    torch.save({"state": model.state_dict(), "x_mean": mu, "x_std": sd, "scale": scale,
                "grid": meta["grid"], "window_R": meta["window_R"], "variant": cfg["variant"],
                "width": cfg.get("width", 64)}, os.path.join(out, "model.pt"))
    np.savez_compressed(os.path.join(out, "test_pred.npz"), pred=pred.astype(np.float32), idx=te)
    json.dump({"config": cfg, "git": git_hash(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
               "metrics": met, "history": hist}, open(os.path.join(out, "metrics.json"), "w"),
              ensure_ascii=False, indent=1)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in met.items()}, indent=1))


if __name__ == "__main__":
    main()
