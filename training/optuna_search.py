import argparse
import os

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from training.train_stress import (
    load_config, build_model, parse_args as _parse_args,
)


def make_objective(cfg, X_tv_std, y_tv_std, device, search_space):
    X_t = torch.from_numpy(X_tv_std).float().to(device)
    y_t = torch.from_numpy(y_tv_std).float().to(device)
    n = X_t.shape[0]
    n_tr = int(n * 0.85)

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", *search_space["n_layers"])
        hidden = [trial.suggest_int(f"n_units_{i}", *search_space["n_units"])
                  for i in range(n_layers)]
        lr = trial.suggest_float("lr", *search_space["lr"], log=True)
        activation = trial.suggest_categorical("activation",
                                                search_space["activation"])

        arch_cfg = dict(cfg["model"])
        arch_cfg["hidden"] = hidden
        arch_cfg["activation"] = activation

        torch.manual_seed(cfg["data"]["seed"])
        model = build_model(arch_cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        loader = DataLoader(
            TensorDataset(X_t[:n_tr], y_t[:n_tr]),
            batch_size=cfg["training"]["batch_size"], shuffle=True,
        )

        best = float("inf")
        bad = 0
        for epoch in range(cfg["training"]["epochs"]):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(model(xb), yb)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val = nn.functional.mse_loss(
                    model(X_t[n_tr:]), y_t[n_tr:],
                ).item()
            if val < best - 1e-6:
                best = val; bad = 0
            else:
                bad += 1
                if bad >= cfg["training"].get("patience", 30):
                    break
            trial.report(val, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return best

    return objective


def run_study(cfg_path, X_tv_std, y_tv_std, search_space, n_trials=20,
              storage=None, study_name=None, device=None):
    cfg = load_config(cfg_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    storage = storage or os.environ.get("OPTUNA_STORAGE")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        sampler=optuna.samplers.TPESampler(seed=cfg["data"]["seed"]),
        storage=storage, study_name=study_name,
        load_if_exists=storage is not None,
    )
    study.optimize(
        make_objective(cfg, X_tv_std, y_tv_std, device, search_space),
        n_trials=n_trials,
    )
    return study
