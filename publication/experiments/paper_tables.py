"""
paper_tables.py — runs_v2.csv → four booktabs tables for the revised manuscript.

Every number in the tables is computed here from the raw run log; nothing is
copied from RESULTS.md or RESULTS_v1_vs_v2.md.  Alongside the .tex files the
script writes numbers_used.json, which lists every printed number together
with the exact filter on runs_v2.csv (or the code file) it came from, so the
manuscript can be audited number by number.

Tables
  T1  tab:main      macro-RMSE mean ± sd per split × family (stage `main`)
  T2  tab:paired    paired t-tests by seed, PINN − MLP and VPINN − MLP
  T3  tab:corrupt   label-corruption sweep + regression of the gap on the rate
  T4  tab:backend   PyTorch vs JAX: accuracy and cost, paired by seed

Statistics
  * mean ± sd over seeds, sd with ddof = 1 (pandas default);
  * paired t-test (scipy.stats.ttest_rel) with the seed as the block — the
    same test as publication/code/stats.py::significance_gate;
  * α = 0.05, two-sided;
  * regression of the seed-level gap PINN − MLP on the corruption fraction
    (scipy.stats.linregress on all seed × rate points).

Usage
  python publication/experiments/paper_tables.py \
      --csv publication/results/runs_v2.csv \
      --out publication/manuscript/tables
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
CSV_NAME = "runs_v2.csv"

FAMILY_LABEL = {"mlp": "MLP", "pinn": "PINN", "vpinn": "VPINN"}
FAMILIES = ["mlp", "pinn", "vpinn"]

# Order requested for Table 1: random, interp:*, extrap:*, extrap_joint:corner,
# matched:*.  Within a group the factor order follows the input vector
# [Q, k, α, μ, v]; μ has no held-out level (duplicates, ERRATA E-03/E-22).
SPLIT_ORDER = [
    "random",
    "interp:Q_mid", "interp:alpha_mid",
    "extrap:Q_high", "extrap:k_max", "extrap:alpha_max", "extrap:v_max",
    "extrap_joint:corner",
    "matched:Q_high", "matched:k_max", "matched:alpha_max", "matched:v_max",
    "matched:corner",
]

# Held-out regions, from publication/code/splits.py::REGIONS.  Units: Q is the
# reduction in DIAMETER (ERRATA E-22), α in degrees, v in m/min (the unit used
# in publication/manuscript/captions.md; the levels are 5/10/20/40/250).
REGION_TEXT = {
    "random": ("random", r"random 15\,\% of all cases"),
    "interp:Q_mid": ("interp", r"$Q = 0.10$ (interior level)"),
    "interp:alpha_mid": ("interp", r"$\alpha = 12^\circ$ (interior level)"),
    "extrap:Q_high": ("extrap", r"$Q = 0.20$ (highest level)"),
    "extrap:k_max": ("extrap", r"$k = 1.0$ (highest level)"),
    "extrap:alpha_max": ("extrap", r"$\alpha = 20^\circ$ (highest level)"),
    "extrap:v_max": ("extrap", r"$v = 250$ m/min (highest level)"),
    "extrap_joint:corner": ("extrap\\_joint", r"$\alpha = 20^\circ$ and $Q \ge 0.15$"),
    "matched:Q_high": ("matched", r"random, $n_\mathrm{test}$ of \texttt{extrap:Q\_high}"),
    "matched:k_max": ("matched", r"random, $n_\mathrm{test}$ of \texttt{extrap:k\_max}"),
    "matched:alpha_max": ("matched", r"random, $n_\mathrm{test}$ of \texttt{extrap:alpha\_max}"),
    "matched:v_max": ("matched", r"random, $n_\mathrm{test}$ of \texttt{extrap:v\_max}"),
    "matched:corner": ("matched", r"random, $n_\mathrm{test}$ of \texttt{extrap\_joint:corner}"),
}
REGION_SRC = "publication/code/splits.py: REGIONS"


# ────────────────────────────── helpers ───────────────────────────────────

class Ledger:
    """Collects every printed number with its provenance."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, str]] = []

    def add(self, value, meaning: str, source: str) -> str:
        v = value if isinstance(value, str) else str(value)
        self.rows.append({"value": v, "meaning": meaning, "source": source})
        return v

    def dump(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.rows, fh, ensure_ascii=False, indent=1)


def tt(s: str) -> str:
    """Split name in typewriter with underscores escaped."""
    return r"\texttt{" + s.replace("_", r"\_") + "}"


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "---"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.2g}"


def fmt_ms(mean: float, sd: float, nd: int = 2) -> str:
    return f"{mean:.{nd}f} $\\pm$ {sd:.{nd}f}"


def holm_adjust(p: np.ndarray) -> np.ndarray:
    """Holm–Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def paired(a: pd.Series, b: pd.Series) -> Tuple[float, float, float, float, int]:
    """Paired t-test of (a − b), aligned on the seed index.

    Returns mean Δ, sd Δ, t, p, n."""
    a = a.sort_index()
    b = b.sort_index()
    if not a.index.equals(b.index):
        raise ValueError(f"seed sets differ: {list(a.index)} vs {list(b.index)}")
    d = a - b
    res = stats.ttest_rel(a.to_numpy(), b.to_numpy())
    return float(d.mean()), float(d.std(ddof=1)), float(res.statistic), float(res.pvalue), int(len(d))


def by_seed(df: pd.DataFrame, col: str = "macro_rmse", **flt) -> pd.Series:
    sub = df
    for k, v in flt.items():
        sub = sub[sub[k] == v]
    s = sub.set_index("seed")[col]
    if s.index.has_duplicates:
        raise ValueError(f"duplicate seeds for {flt}")
    return s.sort_index()


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "error" in df.columns:
        bad = df["error"].notna() & (df["error"].astype(str).str.len() > 0)
        if bad.any():
            print(f"runs with an error field: {int(bad.sum())} — excluded")
            df = df.loc[~bad]
    return df.copy()


# ─────────────────────────────── T1 ──────────────────────────────────────

def table_main(df: pd.DataFrame, L: Ledger) -> str:
    d = df[df["stage"] == "main"]
    splits = [s for s in SPLIT_ORDER if s in set(d["split"])]
    missing = set(d["split"]) - set(splits)
    if missing:
        raise ValueError(f"splits not in SPLIT_ORDER: {missing}")

    n_seeds_all = set()
    body = []
    prev_kind = None
    for sp in splits:
        sub = d[d["split"] == sp]
        kind, region = REGION_TEXT[sp]
        n_test = int(sub["n_test"].iloc[0])
        n_train = int(sub["n_train"].iloc[0])
        assert sub["n_test"].nunique() == 1 and sub["n_train"].nunique() == 1
        L.add(n_train, f"n_train, {sp}", f"{CSV_NAME}: stage=main, split={sp} -> n_train")
        L.add(n_test, f"n_test, {sp}", f"{CSV_NAME}: stage=main, split={sp} -> n_test")
        cells = []
        seeds_by_family = []
        for f in FAMILIES:
            s = by_seed(sub, family=f)
            seeds_by_family.append(tuple(s.index))
            n_seeds_all.add(len(s))
            m, sd = float(s.mean()), float(s.std(ddof=1))
            src = f"{CSV_NAME}: stage=main, split={sp}, family={f} -> macro_rmse over {len(s)} seeds"
            L.add(f"{m:.2f}", f"macro-RMSE mean [MPa], {FAMILY_LABEL[f]}, {sp}", src + ", mean")
            L.add(f"{sd:.2f}", f"macro-RMSE sd [MPa], {FAMILY_LABEL[f]}, {sp}", src + ", sd(ddof=1)")
            cells.append(fmt_ms(m, sd))
        if len(set(seeds_by_family)) != 1:
            raise ValueError(f"{sp}: families trained on different seed sets")
        if prev_kind is not None and kind != prev_kind:
            body.append(r"\addlinespace")
        prev_kind = kind
        body.append(" & ".join([tt(sp), region, str(n_train), str(n_test)] + cells) + r" \\")
    if len(n_seeds_all) != 1:
        raise ValueError(f"unequal seed counts across cells: {n_seeds_all}")
    n_seeds = n_seeds_all.pop()
    L.add(n_seeds, "seeds per split x family, main track", f"{CSV_NAME}: stage=main -> count(seed) per split, family")
    L.add(len(splits), "number of splits", f"{CSV_NAME}: stage=main -> nunique(split)")

    # Region-definition numbers (caption / column 2) come from splits.py.
    for val, meaning in [("0.10", "Q level held out in interp:Q_mid"),
                         ("12", "alpha level (deg) held out in interp:alpha_mid"),
                         ("0.20", "Q level held out in extrap:Q_high"),
                         ("1.0", "k level held out in extrap:k_max"),
                         ("20", "alpha level (deg) held out in extrap:alpha_max and corner"),
                         ("0.15", "Q threshold of the joint corner region"),
                         ("250", "v level (m/min) held out in extrap:v_max; unit from publication/manuscript/captions.md"),
                         ("15", "random hold-out fraction, percent")]:
        L.add(val, meaning, REGION_SRC if val != "15" else "publication/code/splits.py: make_random_split(frac=0.15)")

    cap = (r"Macro-averaged RMSE of the four stress components on the held-out cases, "
           r"MPa, mean $\pm$ sd over " + str(n_seeds) + r" seeds, for the " + str(len(splits)) +
           r" splits of the main track (input $[Q,k,\alpha,\mu,v,r]$). "
           r"\texttt{random} is a random hold-out; \texttt{interp:*} removes an interior "
           r"factor level; \texttt{extrap:*} removes the highest level of one factor; "
           r"\texttt{extrap\_joint:corner} removes a corner of the joint $(\alpha,Q)$ space "
           r"(each axis separately stays inside the training range); \texttt{matched:*} is a "
           r"random hold-out of the same size as the corresponding \texttt{extrap} split, i.e. "
           r"the control that separates the cost of leaving the range from the cost of a "
           r"smaller training pool. The three families share the architecture, optimiser, "
           r"budget, early stopping and seeds and differ only in the composition of the loss.")
    head = (r"\begin{table}[htbp]" "\n"
            r"\centering" "\n"
            r"\caption{" + cap + "}\n"
            r"\label{tab:main}" "\n"
            r"\small" "\n"
            r"\begin{tabular}{@{}llrrrrr@{}}" "\n"
            r"\toprule" "\n"
            r"Split & Held-out region & $n_\mathrm{train}$ & $n_\mathrm{test}$ & MLP & PINN & VPINN \\" "\n"
            r"\midrule")
    tail = (r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{table}")
    src_note = (f"% T1 generated by publication/experiments/paper_tables.py from {CSV_NAME}\n"
                f"% src: {CSV_NAME}: stage==main; macro_rmse mean/sd(ddof=1) over seeds per split×family\n"
                f"% src: held-out regions — {REGION_SRC}\n")
    return src_note + "\n".join([head] + body + [tail])


# ─────────────────────────────── T2 ──────────────────────────────────────

def verdict(delta: float, p: float, phys: str) -> str:
    if not np.isfinite(p) or p >= ALPHA:
        return "n.s."
    return f"{phys} better" if delta < 0 else "MLP better"


def table_paired(df: pd.DataFrame, L: Ledger) -> str:
    d = df[df["stage"] == "main"]
    splits = [s for s in SPLIT_ORDER if s in set(d["split"])]
    body = []
    prev_kind = None
    n_seeds = None
    pvals: Dict[str, List[float]] = {"pinn": [], "vpinn": []}
    for sp in splits:
        sub = d[d["split"] == sp]
        kind = REGION_TEXT[sp][0]
        mlp = by_seed(sub, family="mlp")
        cells = []
        for f in ("pinn", "vpinn"):
            dm, dsd, t, p, n = paired(by_seed(sub, family=f), mlp)
            n_seeds = n if n_seeds is None else n_seeds
            assert n == n_seeds
            v = verdict(dm, p, FAMILY_LABEL[f])
            src = f"{CSV_NAME}: stage=main, split={sp} -> per-seed macro_rmse[{f}]-[mlp], ttest_rel, n={n}"
            L.add(f"{dm:+.2f}", f"Delta {FAMILY_LABEL[f]}-MLP [MPa], {sp}", src)
            L.add(f"{t:.2f}", f"paired t, {FAMILY_LABEL[f]}-MLP, {sp}", src)
            L.add(fmt_p(p), f"paired p, {FAMILY_LABEL[f]}-MLP, {sp}", src)
            cells += [f"${dm:+.2f}$", f"${t:.2f}$", fmt_p(p), v]
            pvals[f].append(p)
        if prev_kind is not None and kind != prev_kind:
            body.append(r"\addlinespace")
        prev_kind = kind
        body.append(" & ".join([tt(sp)] + cells) + r" \\")
    L.add(n_seeds, "pairs (seeds) per paired test", f"{CSV_NAME}: stage=main -> count(seed) per split, family")
    L.add("0.05", "significance level α", "protocol: publication/code/stats.py::significance_gate(alpha=0.05)")

    # Holm–Bonferroni across the splits: how many verdicts survive a family-wise
    # correction.  Reported in the caption so the uncorrected verdicts are not
    # over-read.
    holm = {}
    for f, ps in pvals.items():
        adj = holm_adjust(np.array(ps))
        holm[f] = int((adj < ALPHA).sum())
        L.add(holm[f], f"splits significant after Holm, {FAMILY_LABEL[f]}-MLP",
              f"{CSV_NAME}: stage=main -> Holm-Bonferroni over the {len(ps)} paired p-values, {FAMILY_LABEL[f]}-MLP")
        L.add(f"{adj.min():.3f}", f"smallest Holm-adjusted p, {FAMILY_LABEL[f]}-MLP",
              f"{CSV_NAME}: stage=main -> Holm-Bonferroni over the {len(ps)} paired p-values, {FAMILY_LABEL[f]}-MLP")
    holm_txt = (r" With a Holm correction over the " + str(len(splits)) + r" splits, "
                + str(holm["pinn"]) + r" of the PINN $-$ MLP and " + str(holm["vpinn"])
                + r" of the VPINN $-$ MLP differences remain significant (smallest adjusted $p = "
                + f"{holm_adjust(np.array(pvals['pinn'])).min():.3f}" + r"$ and $"
                + f"{holm_adjust(np.array(pvals['vpinn'])).min():.3f}" + r"$), so the verdicts should be "
                r"read as consistent direction rather than as isolated proofs.")

    cap = (r"Paired comparison of the physics-informed families with the data-driven baseline "
           r"on every split of Table~\ref{tab:main}. $\Delta$ is the mean over seeds of the "
           r"per-seed difference in macro-RMSE (MPa; negative favours the physics-informed model), "
           r"$t$ and $p$ are from a two-sided paired $t$-test with the seed as the block "
           r"($n = " + str(n_seeds) + r"$ pairs, 4 degrees of freedom), and the verdict applies "
           r"$\alpha = 0.05$ without correction for the number of splits. Seeds, not held-out "
           r"cases, are the replication unit: the test answers whether the advantage survives "
           r"retraining, not on how many cases one model beats the other." + holm_txt)
    head = (r"\begin{table}[htbp]" "\n"
            r"\centering" "\n"
            r"\caption{" + cap + "}\n"
            r"\label{tab:paired}" "\n"
            r"\small" "\n"
            r"\begin{tabular}{@{}lrrrlrrrl@{}}" "\n"
            r"\toprule" "\n"
            r" & \multicolumn{4}{c}{PINN $-$ MLP} & \multicolumn{4}{c}{VPINN $-$ MLP} \\" "\n"
            r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}" "\n"
            r"Split & $\Delta$, MPa & $t$ & $p$ & verdict & $\Delta$, MPa & $t$ & $p$ & verdict \\" "\n"
            r"\midrule")
    tail = (r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{table}")
    src_note = (f"% T2 generated by publication/experiments/paper_tables.py from {CSV_NAME}\n"
                f"% src: {CSV_NAME}: stage==main; per-seed macro_rmse differences, scipy.stats.ttest_rel, alpha=0.05\n")
    return src_note + "\n".join([head] + body + [tail])


# ─────────────────────────────── T3 ──────────────────────────────────────

def table_corrupt(df: pd.DataFrame, L: Ledger) -> str:
    d = df[df["stage"] == "corrupt"]
    splits = sorted(d["split"].unique())
    if len(splits) != 1:
        raise ValueError(f"corrupt stage spans several splits: {splits}")
    sp = splits[0]
    rates = sorted(d["corrupt_rate"].unique())
    pct = [f"{int(round(r * 100))}" for r in rates]
    for r, pc in zip(rates, pct):
        L.add(pc, "corruption rate, % of training cases", f"{CSV_NAME}: stage=corrupt -> corrupt_rate={r}")

    n_seeds = None
    body = []
    for f in FAMILIES:
        cells = []
        means = {}
        for r in rates:
            s = by_seed(d, family=f, corrupt_rate=r)
            n_seeds = len(s) if n_seeds is None else n_seeds
            assert len(s) == n_seeds
            m, sd = float(s.mean()), float(s.std(ddof=1))
            means[r] = m
            src = f"{CSV_NAME}: stage=corrupt, family={f}, corrupt_rate={r} -> macro_rmse over {len(s)} seeds"
            L.add(f"{m:.2f}", f"macro-RMSE mean [MPa], {FAMILY_LABEL[f]}, corruption {r:g}", src + ", mean")
            L.add(f"{sd:.2f}", f"macro-RMSE sd [MPa], {FAMILY_LABEL[f]}, corruption {r:g}", src + ", sd(ddof=1)")
            cells.append(fmt_ms(m, sd))
        ratio = means[max(rates)] / means[min(rates)]
        L.add(f"{ratio:.2f}", f"degradation factor {max(rates):g}/{min(rates):g}, {FAMILY_LABEL[f]}",
              f"{CSV_NAME}: stage=corrupt, family={f} -> mean(macro_rmse) at corrupt_rate={max(rates)} / at {min(rates)}")
        body.append(" & ".join([FAMILY_LABEL[f]] + cells + [f"$\\times${ratio:.2f}"]) + r" \\")

    body.append(r"\midrule")
    mlp = {r: by_seed(d, family="mlp", corrupt_rate=r) for r in rates}
    gap_points: List[Tuple[float, float]] = []
    for f in ("pinn", "vpinn"):
        dcells, pcells = [], []
        for r in rates:
            dm, dsd, t, p, n = paired(by_seed(d, family=f, corrupt_rate=r), mlp[r])
            src = f"{CSV_NAME}: stage=corrupt, corrupt_rate={r} -> per-seed macro_rmse[{f}]-[mlp], ttest_rel, n={n}"
            L.add(f"{dm:+.2f}", f"Delta {FAMILY_LABEL[f]}-MLP [MPa], corruption {r:g}", src)
            L.add(fmt_p(p), f"paired p, {FAMILY_LABEL[f]}-MLP, corruption {r:g}", src)
            dcells.append(f"${dm:+.2f}$")
            pcells.append(fmt_p(p))
            if f == "pinn":
                diff = (by_seed(d, family=f, corrupt_rate=r) - mlp[r])
                gap_points += [(float(r), float(v)) for v in diff.to_numpy()]
        body.append(" & ".join([f"$\\Delta$({FAMILY_LABEL[f]} $-$ MLP), MPa"] + dcells + ["---"]) + r" \\")
        body.append(" & ".join([f"$p$ (paired, {n_seeds} seeds)"] + pcells + ["---"]) + r" \\")

    x = np.array([g[0] for g in gap_points])
    y = np.array([g[1] for g in gap_points])
    lr = stats.linregress(x, y)
    r2 = float(lr.rvalue ** 2)
    n_pts = len(x)
    src_reg = (f"{CSV_NAME}: stage=corrupt -> seed-level gaps macro_rmse[pinn]-[mlp] at each corrupt_rate "
               f"({n_pts} points), linregress(gap ~ corrupt_rate)")
    L.add(f"{lr.slope:.2f}", "regression slope, gap PINN-MLP vs corruption fraction [MPa/unit]", src_reg)
    L.add(f"{lr.slope / 10:.2f}", "same slope per 10 percentage points [MPa]", src_reg)
    L.add(fmt_p(lr.pvalue), "regression p (slope = 0)", src_reg)
    L.add(f"{r2:.3f}", "regression R^2", src_reg)
    L.add(n_pts, "points in the regression", src_reg)
    L.add(n_seeds, "seeds per family x rate, corrupt stage", f"{CSV_NAME}: stage=corrupt -> count(seed) per family, corrupt_rate")

    body.append(r"\midrule")
    ncol = 2 + len(rates)
    body.append(r"\multicolumn{" + str(ncol) + r"}{@{}l@{}}{Regression of the seed-level gap "
                r"$\Delta$(PINN $-$ MLP) on the corruption fraction ($n = " + str(n_pts) + r"$ points): "
                r"slope $= " + f"{lr.slope:.2f}" + r"$ MPa per unit fraction "
                r"($" + f"{lr.slope / 10:.2f}" + r"$ MPa per 10 percentage points), "
                r"$p = " + fmt_p(lr.pvalue) + r"$, $R^2 = " + f"{r2:.3f}" + r"$.} \\")

    cap = (r"Robustness to label corruption on the \texttt{" + sp + r"} split: macro-RMSE on the clean "
           r"hold-out (MPa, mean $\pm$ sd over " + str(n_seeds) + r" seeds) when the given fraction of the "
           r"training cases carries a systematic offset of one stress component, for the three "
           r"families; the last column is the degradation factor between the highest and zero "
           r"corruption. Below the rule: the paired gap to the data-driven baseline at each rate "
           r"($\Delta < 0$ favours the physics-informed model; paired $t$-test over seeds), and the "
           r"linear regression of the seed-level gap PINN $-$ MLP on the corruption fraction.")
    colspec = "@{}l" + "r" * len(rates) + "r@{}"
    head = (r"\begin{table}[htbp]" "\n"
            r"\centering" "\n"
            r"\caption{" + cap + "}\n"
            r"\label{tab:corrupt}" "\n"
            r"\small" "\n"
            r"\begin{tabular}{" + colspec + "}\n"
            r"\toprule" "\n"
            r" & \multicolumn{" + str(len(rates)) + r"}{c}{Corrupted training labels, \%} & \\" "\n"
            r"\cmidrule(lr){2-" + str(1 + len(rates)) + "}" "\n"
            r"Family & " + " & ".join(pct) + r" & $\times$(" + pct[-1] + r"\,\%/" + pct[0] + r"\,\%) \\" "\n"
            r"\midrule")
    tail = (r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{table}")
    src_note = (f"% T3 generated by publication/experiments/paper_tables.py from {CSV_NAME}\n"
                f"% src: {CSV_NAME}: stage==corrupt & split=={sp}; macro_rmse mean/sd(ddof=1) over seeds per family×corrupt_rate;\n"
                f"%      paired ttest_rel by seed; linregress of seed-level PINN−MLP gap on corrupt_rate\n")
    return src_note + "\n".join([head] + body + [tail])


# ─────────────────────────────── T4 ──────────────────────────────────────

BACKEND_LABEL = {"torch": "PyTorch", "jax": "JAX"}
COST_COLS = [("macro_rmse", "macro-RMSE, MPa", 2), ("seconds", "wall time, s", 0),
             ("epochs", "epochs", 0), ("sec_per_epoch", "s/epoch", 3)]


def table_backend(df: pd.DataFrame, L: Ledger) -> str:
    d = df[df["stage"] == "backend"].copy()
    d["sec_per_epoch"] = d["seconds"] / d["epochs"]
    fams = sorted(d["family"].unique())
    if fams != ["pinn"]:
        raise ValueError(f"backend stage expected family pinn only, got {fams}")
    splits = [s for s in SPLIT_ORDER if s in set(d["split"])]
    backends = ["torch", "jax"]

    body = []
    n_seeds = None
    for sp in splits:
        sub = d[d["split"] == sp]
        per = {}
        for be in backends:
            cells = []
            for col, name, nd in COST_COLS:
                s = by_seed(sub, col=col, backend=be)
                n_seeds = len(s) if n_seeds is None else n_seeds
                assert len(s) == n_seeds
                per[(be, col)] = s
                m, sd = float(s.mean()), float(s.std(ddof=1))
                cname = col if col != 'sec_per_epoch' else 'seconds/epochs'
                src = f"{CSV_NAME}: stage=backend, split={sp}, backend={be} -> {cname} over {len(s)} seeds"
                L.add(f"{m:.{nd}f}", f"{name} mean, {BACKEND_LABEL[be]}, {sp}", src + ", mean")
                L.add(f"{sd:.{nd}f}", f"{name} sd, {BACKEND_LABEL[be]}, {sp}", src + ", sd(ddof=1)")
                cells.append(fmt_ms(m, sd, nd))
            body.append(" & ".join([tt(sp) if be == backends[0] else "", BACKEND_LABEL[be]] + cells) + r" \\")
        pcells = []
        for col, name, nd in COST_COLS:
            dm, dsd, t, p, n = paired(per[("jax", col)], per[("torch", col)])
            cname = col if col != 'sec_per_epoch' else 'seconds/epochs'
            src = f"{CSV_NAME}: stage=backend, split={sp} -> per-seed {cname}[jax]-[torch], ttest_rel, n={n}"
            L.add(fmt_p(p), f"paired p, JAX vs PyTorch, {name}, {sp}", src)
            pcells.append(f"$p = {fmt_p(p)}$")
        body.append(" & ".join(["", r"paired $p$ (JAX vs PyTorch)"] + pcells) + r" \\")
        if sp != splits[-1]:
            body.append(r"\addlinespace")

    # Pooled over both splits, paired by (split, seed).  Only the COST columns
    # are pooled: macro-RMSE on two different splits measures two different
    # tasks and must not be averaged (publication/code/stats.py::pool_guard).
    body.append(r"\midrule")
    pooled = {}
    for be in backends:
        cells = ["---"]
        for col, name, nd in COST_COLS[1:]:
            s = d[d["backend"] == be].set_index(["split", "seed"])[col].sort_index()
            pooled[(be, col)] = s
            m, sd = float(s.mean()), float(s.std(ddof=1))
            cname = col if col != 'sec_per_epoch' else 'seconds/epochs'
            src = f"{CSV_NAME}: stage=backend, backend={be}, both splits ({len(s)} runs) -> {cname}"
            L.add(f"{m:.{nd}f}", f"{name} mean, {BACKEND_LABEL[be]}, pooled", src + ", mean")
            L.add(f"{sd:.{nd}f}", f"{name} sd, {BACKEND_LABEL[be]}, pooled", src + ", sd(ddof=1)")
            cells.append(fmt_ms(m, sd, nd))
        body.append(" & ".join([r"both splits" if be == backends[0] else "", BACKEND_LABEL[be]] + cells) + r" \\")
    rcells, pcells = ["---"], ["---"]
    n_pool = None
    for col, name, nd in COST_COLS[1:]:
        sj, st = pooled[("jax", col)], pooled[("torch", col)]
        if not sj.index.equals(st.index):
            raise ValueError("pooled backend runs are not aligned on (split, seed)")
        ratio = float(sj.mean() / st.mean())
        res = stats.ttest_rel(sj.to_numpy(), st.to_numpy())
        n_pool = len(sj)
        cname = col if col != 'sec_per_epoch' else 'seconds/epochs'
        src = f"{CSV_NAME}: stage=backend, both splits -> {cname}: mean[jax]/mean[torch]; ttest_rel paired on (split, seed), n={n_pool}"
        L.add(f"{ratio:.2f}", f"ratio JAX/PyTorch, mean {name}, pooled", src)
        L.add(fmt_p(float(res.pvalue)), f"paired p, JAX vs PyTorch, {name}, pooled", src)
        rcells.append(f"$\\times${ratio:.2f}")
        pcells.append(f"$p = {fmt_p(float(res.pvalue))}$")
    body.append(" & ".join(["", "ratio JAX/PyTorch"] + rcells) + r" \\")
    body.append(" & ".join(["", f"paired $p$ ($n = {n_pool}$)"] + pcells) + r" \\")
    L.add(n_seeds, "seeds per backend per split", f"{CSV_NAME}: stage=backend -> count(seed) per split, backend")
    L.add(n_pool, "runs per backend, both splits", f"{CSV_NAME}: stage=backend -> count per backend")

    cap = (r"Two independent implementations of the same strong-form PINN (PyTorch and JAX; "
           r"identical architecture, data, splits, nominal hyper-parameters and early-stopping rule; "
           r"the JAX twin applies $\lambda_{\mathrm{phys}}=0.3$ directly, without the gradient-norm "
           r"calibration of the PyTorch run, and each draws its own initialisation and batch order) on the \texttt{random} and "
           r"\texttt{extrap:alpha\_max} splits, " + str(n_seeds) + r" seeds each. Accuracy is the "
           r"macro-RMSE on the held-out cases; cost is the wall time of the full training run, "
           r"the number of epochs to early stopping and their ratio (mean of the per-run ratio). "
           r"Mean $\pm$ sd over seeds; $p$ from a two-sided paired $t$-test with the seed as the "
           r"block. The bottom block pools the cost figures of both splits (" + str(n_pool) + r" runs per backend, "
           r"paired on split and seed); accuracy is not pooled because the two splits are different "
           r"tasks. Timings are for CPU, double precision; they are not "
           r"transferable to GPU or single precision, where JIT compilation may repay itself "
           r"differently.")
    head = (r"\begin{table}[htbp]" "\n"
            r"\centering" "\n"
            r"\caption{" + cap + "}\n"
            r"\label{tab:backend}" "\n"
            r"\small" "\n"
            r"\begin{tabular}{@{}llrrrr@{}}" "\n"
            r"\toprule" "\n"
            r"Split & Backend & macro-RMSE, MPa & Wall time, s & Epochs & s/epoch \\" "\n"
            r"\midrule")
    tail = (r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{table}")
    src_note = (f"% T4 generated by publication/experiments/paper_tables.py from {CSV_NAME}\n"
                f"% src: {CSV_NAME}: stage==backend (family pinn); macro_rmse, seconds, epochs, seconds/epochs;\n"
                f"%      mean/sd(ddof=1) over seeds; scipy.stats.ttest_rel paired by seed (per split) and by (split, seed) (pooled)\n")
    return src_note + "\n".join([head] + body + [tail])


# ─────────────────────────────── main ────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--csv", default=os.path.join(here, "..", "results", CSV_NAME))
    ap.add_argument("--out", default=os.path.join(here, "..", "manuscript", "tables"))
    a = ap.parse_args()

    df = load(a.csv)
    os.makedirs(a.out, exist_ok=True)
    L = Ledger()
    L.add(len(df), "runs in the log", f"{CSV_NAME}: row count (no run has an error field)")

    tables = {
        "tab_main.tex": table_main(df, L),
        "tab_paired.tex": table_paired(df, L),
        "tab_corrupt.tex": table_corrupt(df, L),
        "tab_backend.tex": table_backend(df, L),
    }
    for name, tex in tables.items():
        path = os.path.join(a.out, name)
        with open(path, "w") as fh:
            fh.write(tex + "\n")
        print(f"written: {path}")
    all_path = os.path.join(a.out, "all_tables.tex")
    with open(all_path, "w") as fh:
        fh.write("% Requires \\usepackage{booktabs}. Generated by publication/experiments/paper_tables.py\n"
                 f"% from publication/results/{CSV_NAME}; do not edit by hand — rerun the script.\n\n")
        fh.write("\n\n".join(tables.values()) + "\n")
    print(f"written: {all_path}")
    L.dump(os.path.join(a.out, "numbers_used.json"))
    print(f"numbers recorded: {len(L.rows)} → {os.path.join(a.out, 'numbers_used.json')}")


if __name__ == "__main__":
    main()
