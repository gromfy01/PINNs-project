"""
analyze.py — результаты прогонов (csv) → таблицы статьи.

Ничего не усредняет по сплитам и ничего не сравнивает без оценки межсидового
разброса: и то и другое было ошибкой отклонённой версии (ERRATA E-07, E-08).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from stats import (SeedSummary, build_seed_summary, delta_vs_matched,  # noqa: E402
                   nemenyi, significance_gate)

FAMILY_LABEL = {"mlp": "MLP (data-driven)", "pinn": "PINN (strong form)",
                "vpinn": "VPINN (weak form)"}
METRIC = "macro_rmse"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    bad = df["error"].notna() & (df["error"].astype(str).str.len() > 0)
    if bad.any():
        print(f"⚠ прогонов с ошибкой: {int(bad.sum())} — исключены\n")
        for e in df.loc[bad, "error"].head(3):
            print("   ", str(e)[:160].replace("\n", " "))
    return df.loc[~bad].copy()


def _summary(df: pd.DataFrame, split: str, metric: str = METRIC,
             family_col: str = "family") -> SeedSummary:
    sub = df[df["split"] == split]
    recs = [{"model": r[family_col], "seed": int(r["seed"]), "split": split,
             "macro_rmse": float(r[metric])} for _, r in sub.iterrows()]
    return build_seed_summary(recs)


def table_main(df: pd.DataFrame, metric: str = METRIC) -> pd.DataFrame:
    rows = []
    for split in sorted(df["split"].unique()):
        s = _summary(df, split, metric)
        for r in s.table():
            rows.append({"split": split, "model": r["model"],
                         "mean": r["mean"], "std": r["std"], "n_seeds": r["n_seeds"]})
    return pd.DataFrame(rows)


def table_delta(df: pd.DataFrame, pairs: List[tuple], metric: str = METRIC) -> pd.DataFrame:
    rows = []
    for extrap, matched in pairs:
        if extrap not in set(df["split"]) or matched not in set(df["split"]):
            continue
        e, m = _summary(df, extrap, metric), _summary(df, matched, metric)
        for r in delta_vs_matched(e, m):
            rows.append({"extrap_split": extrap, "control": matched, **r})
    return pd.DataFrame(rows)


def report_ranks(df: pd.DataFrame, metric: str = METRIC) -> str:
    out = []
    for split in sorted(df["split"].unique()):
        s = _summary(df, split, metric)
        if len(s.models) < 3:
            continue
        try:
            res = nemenyi(s)
        except ValueError as exc:
            out.append(f"{split}: ранговый тест невозможен — {exc}")
            continue
        out.append(res.to_text())
        out.append("")
    return "\n".join(out)


def report_gate(df: pd.DataFrame, metric: str = METRIC) -> str:
    out = []
    for split in sorted(df["split"].unique()):
        s = _summary(df, split, metric)
        out.append(f"— {split} —")
        for v in significance_gate(s):
            out.append("   " + str(v))
    return "\n".join(out)


def table_curves(df: pd.DataFrame, metric: str = METRIC) -> pd.DataFrame:
    rows = []
    for split in sorted(df["split"].unique()):
        for frac in sorted(df["frac"].dropna().unique()):
            sub = df[(df["split"] == split) & (df["frac"] == frac)]
            for fam in sorted(sub["family"].unique()):
                v = sub[sub["family"] == fam][metric].to_numpy()
                rows.append({"split": split, "frac": float(frac), "model": fam,
                             "mean": v.mean(), "std": v.std(ddof=1) if len(v) > 1 else 0.0,
                             "n_seeds": len(v)})
    return pd.DataFrame(rows)


def table_corrupt(df: pd.DataFrame, metric: str = METRIC) -> pd.DataFrame:
    rows = []
    for rate in sorted(df["corrupt_rate"].dropna().unique()):
        sub = df[df["corrupt_rate"] == rate]
        for fam in sorted(sub["family"].unique()):
            v = sub[sub["family"] == fam][metric].to_numpy()
            rows.append({"rate": float(rate), "model": fam, "mean": v.mean(),
                         "std": v.std(ddof=1) if len(v) > 1 else 0.0, "n_seeds": len(v)})
    return pd.DataFrame(rows)


def table_physics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["split", "family"]).agg(
        eq_res=("eq_res_median", "mean"), eq_std=("eq_res_median", "std"),
        bc_rr=("bc_sigma_rr", "mean"), bc_rr_std=("bc_sigma_rr", "std"),
        bc_rz=("bc_tau_rz", "mean"), bc_rz_std=("bc_tau_rz", "std")).reset_index()
    return g


def table_backend(df: pd.DataFrame, metric: str = METRIC) -> pd.DataFrame:
    rows = []
    for split in sorted(df["split"].unique()):
        sub = df[df["split"] == split]
        for be in sorted(sub["backend"].unique()):
            v = sub[sub["backend"] == be][metric].to_numpy()
            rows.append({"split": split, "backend": be, "mean": v.mean(),
                         "std": v.std(ddof=1) if len(v) > 1 else 0.0, "n_seeds": len(v)})
    out = pd.DataFrame(rows)
    for split in out["split"].unique():
        s = out[out["split"] == split]
        if len(s) == 2:
            d = abs(s["mean"].iloc[0] - s["mean"].iloc[1])
            pooled = float(np.sqrt(0.5 * (s["std"].iloc[0] ** 2 + s["std"].iloc[1] ** 2)))
            print(f"  {split}: |Δ между бэкендами| = {d:.4f} МПа, "
                  f"σ по сидам = {pooled:.4f} → разница "
                  f"{'установлена' if d >= pooled else 'НЕ установлена'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--stage", default="main")
    ap.add_argument("--metric", default=METRIC)
    a = ap.parse_args()
    df = load(a.csv)
    df = df[df["stage"] == a.stage]
    pd.set_option("display.width", 200, "display.max_rows", 300,
                  "display.float_format", lambda v: f"{v:.4f}")

    if a.stage == "main":
        print("\n=== macro-RMSE, mean ± std по сидам (внутри сплита) ===")
        print(table_main(df, a.metric).to_string(index=False))
        pairs = [(s, "matched:" + s.split(":", 1)[1]) for s in df["split"].unique()
                 if s.startswith("extrap")]
        print("\n=== ΔRMSE = extrap − matched (контроль того же объёма) ===")
        print(table_delta(df, pairs, a.metric).to_string(index=False))
        print("\n=== ранговый анализ ВНУТРИ сплита, блоки — сиды ===")
        print(report_ranks(df, a.metric))
        print("\n=== порог содержательности разницы ===")
        print(report_gate(df, a.metric))
        print("\n=== физический аудит ===")
        print(table_physics(df).to_string(index=False))
    elif a.stage == "curves":
        print("\n=== кривые обучения ===")
        print(table_curves(df, a.metric).to_string(index=False))
    elif a.stage == "corrupt":
        print("\n=== деградация от порчи меток ===")
        print(table_corrupt(df, a.metric).to_string(index=False))
    elif a.stage == "backend":
        print("\n=== сравнение бэкендов при идентичном протоколе ===")
        print(table_backend(df, a.metric).to_string(index=False))


if __name__ == "__main__":
    main()
