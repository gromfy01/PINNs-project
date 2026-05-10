import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from training.data_pipeline import (
    build_strain_dataset, kfold_indices, set_indices_to_rows,
    standardize_features, standardize_targets,
)
from models.architectures import MLP, DDENet, VPINN
from models.strain_system import TotalStrainSystem


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--data-x", default="data/processed/X_strain.pkl")
    p.add_argument("--data-y", default="data/processed/y_strain.pkl")
    p.add_argument("--out", default="results/bundles/strain/run.pkl")
    p.add_argument("--device", default=None)
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(arch_cfg):
    arch_type = arch_cfg["type"]
    in_dim = arch_cfg.get("in_dim", 6)
    out_dim = arch_cfg.get("out_dim", 4)
    hidden = arch_cfg["hidden"]
    activation = arch_cfg.get("activation", "tanh")

    if arch_type == "mlp":
        return MLP(in_dim, hidden, out_dim, activation)
    if arch_type == "dde":
        return DDENet([in_dim] + hidden + [out_dim], activation)
    if arch_type == "vpinn":
        return VPINN(hidden, out_dim, activation,
                     n_freq=arch_cfg.get("n_freq", 8))
    if arch_type == "strain_system":
        return TotalStrainSystem(hidden,
                                 n_freq=arch_cfg.get("n_freq", 16),
                                 use_z=arch_cfg.get("use_z", False))
    raise ValueError(f"Unknown architecture: {arch_type}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.data_x, "rb") as f: X_full = pickle.load(f)
    with open(args.data_y, "rb") as f: y_full = pickle.load(f)

    ds = build_strain_dataset(
        X_full, y_full,
        holdout_frac=cfg["data"]["holdout_frac"],
        seed=cfg["data"]["seed"],
    )

    mean_X, std_X = standardize_features(ds["X_trainval"])
    mean_y, std_y = standardize_targets(ds["y_trainval"])

    X_tv_std = (ds["X_trainval"] - mean_X) / std_X
    y_tv_std = (ds["y_trainval"] - mean_y) / std_y

    fold_results = []
    for fold, (train_sets, val_sets) in enumerate(
        kfold_indices(ds["trainval_sets"], cfg["data"].get("n_folds", 5))
    ):
        train_rows = set_indices_to_rows(np.where(np.isin(ds["trainval_sets"], train_sets))[0])
        val_rows = set_indices_to_rows(np.where(np.isin(ds["trainval_sets"], val_sets))[0])

        X_tr = torch.from_numpy(X_tv_std[train_rows]).float()
        y_tr = torch.from_numpy(y_tv_std[train_rows]).float()
        X_va = torch.from_numpy(X_tv_std[val_rows]).float()
        y_va = torch.from_numpy(y_tv_std[val_rows]).float()

        loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            generator=torch.Generator(device="cpu"),
        )

        torch.manual_seed(cfg["data"]["seed"] + fold)
        model = build_model(cfg["model"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

        best_val = float("inf")
        bad = 0
        patience = cfg["training"].get("patience", 50)
        t0 = time.time()
        for epoch in range(cfg["training"]["epochs"]):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(model(xb), yb)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = nn.functional.mse_loss(
                    model(X_va.to(device)), y_va.to(device),
                ).item()
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                bad = 0
                best_state = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break

        elapsed = time.time() - t0
        fold_results.append({
            "fold": fold, "val_loss": best_val,
            "time_sec": elapsed, "state_dict": best_state,
        })
        print(f"  fold {fold}: val_loss={best_val:.5f}  time={elapsed:.0f}s")

    best = min(fold_results, key=lambda r: r["val_loss"])

    bundle = {
        "framework": cfg["model"]["type"],
        "model_state_dict": best["state_dict"],
        "mean_X": mean_X, "std_X": std_X,
        "scaler_y_mean": mean_y, "scaler_y_std": std_y,
        "best_fold": best["fold"],
        "holdout_set_indices": ds["hold_sets"],
        "trainval_set_indices": ds["trainval_sets"],
        "holdout_frac": cfg["data"]["holdout_frac"],
        "split_seed": cfg["data"]["seed"],
        "config": cfg,
        "activation": cfg["model"]["activation"],
        "val_plane_order": [1, 0, 2, 3],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
