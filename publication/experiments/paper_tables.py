"""
paper_tables.py — runs_v2.csv → six booktabs tables for the paper.

Every number in the tables is computed here from the raw run log; nothing is
copied by hand.  Alongside the .tex files the script writes numbers_used.json,
which lists every printed number together with the exact filter on
runs_v2.csv (or the code file) it came from, so the paper can be audited
number by number.

Tables
  T1  tab:main           macro-RMSE mean ± sd per split × family (stage `main`)
  T2  tab:paired         paired t-tests by seed, PINN − MLP and VPINN − MLP
  T3  tab:corrupt        label-corruption sweep + regression of the gap on the rate
  T4  tab:backend        PyTorch vs JAX: accuracy and cost, paired by seed
  T5  tab:curves         macro-RMSE against the fraction of the training pool
                         (stage `curves`), PINN − MLP paired by seed, slope of
                         the gap on log(fraction), Holm over the PINN − MLP tests
  T6  tab:admissibility  two-dimensional families (stage `twod`): accuracy,
                         full-form equilibrium residuals, traction-free
                         violations; optional FEM row of the same held-out cases

Statistics
  * mean ± sd over seeds, sd with ddof = 1 (pandas default);
  * paired t-test (scipy.stats.ttest_rel) with the seed as the block — the
    same test as code/stats.py::significance_gate;
  * α = 0.05, two-sided;
  * Holm step-down adjustment over the p-values of one comparison repeated
    on several splits, fractions or rates (holm_adjust);
  * regression of the seed-level gap PINN − MLP on the corruption fraction
    and on the natural logarithm of the training fraction
    (scipy.stats.linregress on all seed × level points).

FEM row of T6
  The finite-element data are not in runs_v2.csv.  The row is printed when it
  is supplied either as the json written by experiments/fem_row_2d.py --json
  (--fem-json) or as a Field2D .npz from which it is computed here with the
  functions of fem_row_2d.py (--field2d; a 1848-case set is reduced to the
  1756 cases by audit/removed_1848_to_1756.csv).  Without either, T6 has no
  FEM row and its caption does not mention one.

Usage
  python experiments/paper_tables.py \
      --csv results/runs_v2.csv \
      --out tables \
      [--fem-json fem_row.json | --field2d field2d.npz]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
CSV_NAME = "runs_v2.csv"

FAMILY_LABEL = {"mlp": "MLP", "pinn": "PINN", "vpinn": "VPINN"}
FAMILIES = ["mlp", "pinn", "vpinn"]

# Order of the rows of Table 1: random, interp:*, extrap:*, extrap_joint:corner,
# matched:*.  Within a group the factor order follows the input vector
# [Q, k, α, μ, v]; μ has no held-out level (duplicates).
SPLIT_ORDER = [
    "random",
    "interp:Q_mid", "interp:alpha_mid",
    "extrap:Q_high", "extrap:k_max", "extrap:alpha_max", "extrap:v_max",
    "extrap_joint:corner",
    "matched:Q_high", "matched:k_max", "matched:alpha_max", "matched:v_max",
    "matched:corner",
]

# Held-out regions, from code/splits.py::REGIONS.  Units: Q is the
# reduction in DIAMETER, α in degrees, v in m/min (levels 5/10/20/40/250).
REGION_TEXT = {
    "random": ("random", r"random 15\,\% of all cases"),
    "interp:Q_mid": ("interp", r"$Q = 10\,\%$ (interior level)"),
    "interp:alpha_mid": ("interp", r"$\alpha = 12^\circ$ (interior level)"),
    "extrap:Q_high": ("extrap", r"$Q = 20\,\%$ (highest level)"),
    "extrap:k_max": ("extrap", r"$k = 1.0$ (highest level)"),
    "extrap:alpha_max": ("extrap", r"$\alpha = 20^\circ$ (highest level)"),
    "extrap:v_max": ("extrap", r"$v = 250$ m/min (highest level)"),
    "extrap_joint:corner": ("extrap\\_joint", r"$\alpha = 20^\circ$ and $Q \ge 15\,\%$"),
    "matched:Q_high": ("matched", r"random, $n_\mathrm{test}$ of \texttt{extrap:Q\_high}"),
    "matched:k_max": ("matched", r"random, $n_\mathrm{test}$ of \texttt{extrap:k\_max}"),
    "matched:alpha_max": ("matched", r"random, $n_\mathrm{test}$ of \texttt{extrap:alpha\_max}"),
    "matched:v_max": ("matched", r"random, $n_\mathrm{test}$ of \texttt{extrap:v\_max}"),
    "matched:corner": ("matched", r"random, $n_\mathrm{test}$ of \texttt{extrap\_joint:alpha\_Q}"),
}
REGION_SRC = "code/splits.py: REGIONS"


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


# Names shown in the paper where they differ from the run log: the joint region
# is labelled after its two factors (alpha and Q) instead of "corner".
DISPLAY_NAME = {"extrap_joint:corner": "extrap_joint:alpha_Q", "matched:corner": "matched:alpha_Q"}


def tt(s: str) -> str:
    """Split name in typewriter with underscores escaped."""
    s = DISPLAY_NAME.get(s, s)
    return r"\texttt{" + s.replace("_", r"\_") + "}"


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "---"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.2g}"


def fmt_ms(mean: float, sd: float, nd: int = 2) -> str:
    return f"{mean:.{nd}f} $\\pm$ {sd:.{nd}f}"


def fmt_pm(mean: float, sd: float, nd: int = 2) -> str:
    """mean ± sd as one math group, the form used in tab:curves and tab:admissibility."""
    return f"${mean:.{nd}f}\\pm{sd:.{nd}f}$"


def fmt_p2(p: float) -> str:
    """p-value to two significant digits, trailing zero kept (0.070, 0.21)."""
    return f"{p:#.2g}" if np.isfinite(p) else "---"


NUM_WORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
            8: "eight", 9: "nine", 10: "ten"}


def word(n: int) -> str:
    return NUM_WORD.get(int(n), str(n))


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
    # Q is printed in percent of the diameter (splits.py keeps it as a fraction: 0.10, 0.20, 0.15).
    for val, meaning, src in [("10", "Q level (%) held out in interp:Q_mid", REGION_SRC),
                              ("12", "alpha level (deg) held out in interp:alpha_mid", REGION_SRC),
                              ("20", "Q level (%) held out in extrap:Q_high", REGION_SRC),
                              ("1.0", "k level held out in extrap:k_max", REGION_SRC),
                              ("20", "alpha level (deg) held out in extrap:alpha_max and corner", REGION_SRC),
                              ("15", "Q threshold (%) of the joint corner region", REGION_SRC),
                              ("250", "v level (m/min) held out in extrap:v_max", REGION_SRC),
                              ("15", "random hold-out fraction, percent", "code/splits.py: make_random_split(frac=0.15)")]:
        L.add(val, meaning, src)

    # Caption deliberately short: the definition of the split types and of the matched
    # controls is given in the body text.
    cap = (r"Macro-averaged RMSE on the held-out cases for the " + str(len(splits)) +
           r" splits of the main track, MPa, mean $\pm$ sd over " + str(n_seeds) +
           r" seeds. Split types are defined in Section~\ref{sec:fem-dataset}.")
    head = (r"\begin{table*}[t]" "\n"
            r"\centering" "\n"
            r"\caption[Macro-averaged RMSE on the 13 splits of the main track]{" + cap + "}\n"
            r"\label{tab:main}" "\n"
            r"\small" "\n"
            r"\begin{tabular}{@{}llrrrrr@{}}" "\n"
            r"\toprule" "\n"
            r"Split & Held-out region & $n_\mathrm{train}$ & $n_\mathrm{test}$ & MLP & PINN & VPINN \\" "\n"
            r"\midrule")
    tail = (r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{table*}")
    src_note = (f"% T1 generated by experiments/paper_tables.py from {CSV_NAME}\n"
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
    L.add("0.05", "significance level α", "protocol: code/stats.py::significance_gate(alpha=0.05)")

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
    # The consequence of the Holm correction is stated in the body text; the caption keeps
    # only the counts.
    holm_txt = (r" After a Holm correction over the " + str(len(splits)) + r" splits, "
                + str(holm["pinn"]) + r" PINN $-$ MLP and " + str(holm["vpinn"])
                + r" VPINN $-$ MLP differences remain significant.")

    cap = (r"Seed-paired comparison of the physics-informed families with the data-driven "
           r"baseline on every split of Table~\ref{tab:main}. $\Delta$ is the mean over "
           r"seeds of the per-seed difference in macro-RMSE, MPa. Negative favours the "
           r"physics-informed model. Two-sided paired $t$-test, $n = " + str(n_seeds) +
           r"$ pairs, $\alpha = 0.05$." + holm_txt)
    head = (r"\begin{table*}[t]" "\n"
            r"\centering" "\n"
            r"\caption[Seed-paired comparison with the data-driven baseline]{" + cap + "}\n"
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
            r"\end{table*}")
    src_note = (f"% T2 generated by experiments/paper_tables.py from {CSV_NAME}\n"
                f"% src: {CSV_NAME}: stage==main; per-seed macro_rmse differences, scipy.stats.ttest_rel, alpha=0.05\n")
    return src_note + "\n".join([head] + body + [tail])


# ─────────────────────────────── T3 ──────────────────────────────────────

def pct_list(rates: List[float]) -> str:
    """Corruption fractions as '5\\,\\%', '5 and 10\\,\\%', '1, 5 and 10\\,\\%'."""
    s = [f"{int(round(r * 100))}" for r in rates]
    head = ", ".join(s[:-1])
    return (head + " and " + s[-1] if head else s[-1]) + r"\,\%"


def corrupt_holm_txt(sig_raw: Dict[str, List[float]], sig_holm: Dict[str, List[float]]) -> str:
    """Caption sentence of tab:corrupt on what survives the Holm correction, from the p-values."""
    def clause(f: str) -> str:
        lab = FAMILY_LABEL[f] + r" $-$ MLP"
        raw, holm = sig_raw[f], sig_holm[f]
        if not raw:
            return "no " + lab + " gap is significant before or after it"
        if not holm:
            return "the " + lab + " gap is no longer significant at any fraction"
        only = "only " if len(holm) < len(raw) else ""
        return "the " + lab + " gap remains significant " + only + "at " + pct_list(holm)
    return " After the Holm correction " + clause("pinn") + ", and " + clause("vpinn") + "."


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
    # rates at which each gap is significant before and after Holm, for the caption
    sig_raw: Dict[str, List[float]] = {}
    sig_holm: Dict[str, List[float]] = {}
    for f in ("pinn", "vpinn"):
        dcells, pcells, praw = [], [], []
        for r in rates:
            dm, dsd, t, p, n = paired(by_seed(d, family=f, corrupt_rate=r), mlp[r])
            praw.append(p)
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
        # Holm over the fractions of this comparison (the clean reference included), as for tab:paired
        adj = holm_adjust(np.array(praw))
        for r, pa in zip(rates, adj):
            L.add(fmt_p(pa), f"Holm-adjusted p, {FAMILY_LABEL[f]}-MLP, corruption {r:g}",
                  f"{CSV_NAME}: stage=corrupt -> holm_adjust over the {len(rates)} paired p of {FAMILY_LABEL[f]}-MLP")
        body.append(" & ".join([r"$p_{\mathrm{Holm}}$ (" + str(len(rates)) + " fractions)"]
                               + [fmt_p(pa) for pa in adj] + ["---"]) + r" \\")
        sig_raw[f] = [r for r, p in zip(rates, praw) if p < ALPHA]
        sig_holm[f] = [r for r, pa in zip(rates, adj) if pa < ALPHA]
        src_sig = (f"{CSV_NAME}: stage=corrupt -> corrupt_rate with paired p < {ALPHA} "
                   f"(raw / holm_adjust over the {len(rates)} p) of {FAMILY_LABEL[f]}-MLP")
        L.add(",".join(f"{r:g}" for r in sig_raw[f]) or "none",
              f"corruption rates with significant {FAMILY_LABEL[f]}-MLP gap, before Holm", src_sig)
        L.add(",".join(f"{r:g}" for r in sig_holm[f]) or "none",
              f"corruption rates with significant {FAMILY_LABEL[f]}-MLP gap, after Holm", src_sig)

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
    body.append(r"\multicolumn{" + str(ncol) + r"}{@{}p{0.58\textwidth}@{}}{Regression of the seed-level gap "
                r"$\Delta$(PINN $-$ MLP) on the corruption fraction ($n = " + str(n_pts) + r"$ points): "
                r"slope $= " + f"{lr.slope:.2f}" + r"$ MPa per unit fraction "
                r"($" + f"{lr.slope / 10:.2f}" + r"$ MPa per 10 percentage points), "
                r"$p = " + fmt_p(lr.pvalue) + r"$, $R^2 = " + f"{r2:.3f}" + r"$.} \\")

    cap = (r"Effect of corrupted training labels on the \texttt{" + sp + r"} split. In the stated "
           r"fraction of the training cases a constant offset of $+250$~MPa is added to "
           r"$\sigma_{\theta\theta}$ over the whole profile. Macro-RMSE on the clean hold-out, MPa, "
           r"mean $\pm$ sd over " + str(n_seeds) + r" seeds. Below the rule: the seed-paired gap to "
           r"the data-driven baseline with its two-sided paired $p$-value and the value "
           r"$p_{\mathrm{Holm}}$ adjusted by the Holm procedure over the four fractions of the same "
           r"comparison, the clean reference included, and the regression of the gap on the "
           r"corrupted fraction, a single test that is not adjusted." + corrupt_holm_txt(sig_raw, sig_holm))
    colspec = "@{}l" + "r" * len(rates) + "r@{}"
    head = (r"\begin{table*}[t]" "\n"
            r"\centering" "\n"
            r"\caption[Effect of corrupted training labels]{" + cap + "}\n"
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
            r"\end{table*}")
    src_note = (f"% T3 generated by experiments/paper_tables.py from {CSV_NAME}\n"
                f"% src: {CSV_NAME}: stage==corrupt & split=={sp}; macro_rmse mean/sd(ddof=1) over seeds per family×corrupt_rate;\n"
                f"%      paired ttest_rel by seed; linregress of seed-level PINN−MLP gap on corrupt_rate\n")
    return src_note + "\n".join([head] + body + [tail])


# ─────────────────────────────── T4 ──────────────────────────────────────

BACKEND_LABEL = {"torch": "PyTorch", "jax": "JAX"}
# The JAX twin has no gradient-norm calibration g (phys_gain = 1.0), so its physics term
# carries lambda_phys instead of lambda_phys * g. The table reports that effective weight in
# place of the total wall time, which depends on the loss through the number of epochs.
COST_COLS = [("macro_rmse", "macro-RMSE, MPa", 2), ("w_phys", "effective physics weight", 3),
             ("epochs", "epochs", 0), ("sec_per_epoch", "s/epoch", 3)]
JAX_NO_G = r"0.3 (no $g$)"


def table_backend(df: pd.DataFrame, L: Ledger) -> str:
    d = df[df["stage"] == "backend"].copy()
    d["sec_per_epoch"] = d["seconds"] / d["epochs"]
    d["w_phys"] = d["lambda_physics"] * d["phys_gain"]
    if not (d.loc[d["backend"] == "jax", "phys_gain"] == 1.0).all():
        raise ValueError("JAX backend runs are expected without calibration (phys_gain = 1)")
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
                cells.append(JAX_NO_G if (col == "w_phys" and be == "jax") else fmt_ms(m, sd, nd))
            body.append(" & ".join([tt(sp) if be == backends[0] else "", BACKEND_LABEL[be]] + cells) + r" \\")
        pcells = []
        for col, name, nd in COST_COLS:
            dm, dsd, t, p, n = paired(per[("jax", col)], per[("torch", col)])
            cname = col if col != 'sec_per_epoch' else 'seconds/epochs'
            src = f"{CSV_NAME}: stage=backend, split={sp} -> per-seed {cname}[jax]-[torch], ttest_rel, n={n}"
            if col == "w_phys":
                pcells.append("---")
                continue
            L.add(fmt_p(p), f"paired p, JAX vs PyTorch, {name}, {sp}", src)
            pcells.append(f"$p = {fmt_p(p)}$")
        body.append(" & ".join(["", r"paired $p$ (JAX vs PyTorch)"] + pcells) + r" \\")
        if sp != splits[-1]:
            body.append(r"\addlinespace")

    # Pooled over both splits, paired by (split, seed).  Only the COST columns
    # are pooled: macro-RMSE on two different splits measures two different
    # tasks and must not be averaged (code/stats.py::pool_guard).
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
            cells.append(JAX_NO_G if (col == "w_phys" and be == "jax") else fmt_ms(m, sd, nd))
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
        if col == "w_phys":
            rcells.append(f"$\\times${ratio:.1f}")
            pcells.append("---")
            continue
        rcells.append(f"$\\times${ratio:.2f}")
        pcells.append(f"$p = {fmt_p(float(res.pvalue))}$")
    body.append(" & ".join(["", "ratio JAX/PyTorch"] + rcells) + r" \\")
    body.append(" & ".join(["", f"paired $p$ ($n = {n_pool}$)"] + pcells) + r" \\")
    L.add(n_seeds, "seeds per backend per split", f"{CSV_NAME}: stage=backend -> count(seed) per split, backend")
    L.add(n_pool, "runs per backend, both splits", f"{CSV_NAME}: stage=backend -> count per backend")

    cap = (r"The strong-form PINN in two implementations, PyTorch and JAX, on the "
           r"\texttt{random} and \texttt{extrap:alpha\_max} splits, " + str(n_seeds) +
           r" seeds each: accuracy (macro-RMSE, MPa), effective weight of the physics term and cost. "
           r"The JAX implementation applies $\lambda_{\mathrm{phys}}=0.3$ without the calibration $g$ "
           r"of Eq.~\eqref{eq:calib}, so its epoch counts are not comparable with those of PyTorch "
           r"and only the time per epoch is compared. Mean $\pm$ sd over seeds. The $p$-values are "
           r"from a seed-paired $t$-test. Timings are for CPU, double precision, four runs at a time.")
    head = (r"\begin{table*}[t]" "\n"
            r"\centering" "\n"
            r"\caption[The strong-form PINN in PyTorch and in JAX]{" + cap + "}\n"
            r"\label{tab:backend}" "\n"
            r"\small" "\n"
            r"\begin{tabular}{@{}llrrrr@{}}" "\n"
            r"\toprule" "\n"
            r"Split & Backend & macro-RMSE, MPa & $\lambda_{\mathrm{phys}}g$ & Epochs & s/epoch \\" "\n"
            r"\midrule")
    tail = (r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{table*}")
    src_note = (f"% T4 generated by experiments/paper_tables.py from {CSV_NAME}\n"
                f"% src: {CSV_NAME}: stage==backend (family pinn); macro_rmse, lambda_physics*phys_gain, epochs, seconds/epochs;\n"
                f"%      mean/sd(ddof=1) over seeds; scipy.stats.ttest_rel paired by seed (per split) and by (split, seed) (pooled)\n")
    return src_note + "\n".join([head] + body + [tail])


# ─────────────────────────────── T5 ──────────────────────────────────────

def table_curves(df: pd.DataFrame, L: Ledger) -> str:
    d = df[df["stage"] == "curves"]
    splits = [s for s in SPLIT_ORDER if s in set(d["split"])]
    missing = set(d["split"]) - set(splits)
    if missing:
        raise ValueError(f"splits not in SPLIT_ORDER: {missing}")
    fracs = sorted(d["frac"].unique())

    body = []
    n_seeds = None
    tests: List[Tuple[str, float, float]] = []      # (split, frac, p) of PINN − MLP
    slopes = {}
    for sp in splits:
        sub = d[d["split"] == sp]
        gap_x: List[float] = []
        gap_y: List[float] = []
        for i, fr in enumerate(fracs):
            cur = sub[sub["frac"] == fr]
            assert cur["n_train"].nunique() == 1
            n_train = int(cur["n_train"].iloc[0])
            pct = f"{int(round(fr * 100))}"
            where = f"stage=curves, split={sp}, frac={fr:g}"
            L.add(pct, f"fraction of the training pool, %, {sp}", f"{CSV_NAME}: {where}")
            L.add(n_train, f"n_train, {sp}, fraction {fr:g}", f"{CSV_NAME}: {where} -> n_train")
            cells = []
            seeds_by_family = []
            for f in FAMILIES:
                s = by_seed(cur, family=f)
                seeds_by_family.append(tuple(s.index))
                n_seeds = len(s) if n_seeds is None else n_seeds
                assert len(s) == n_seeds
                m, sd = float(s.mean()), float(s.std(ddof=1))
                src = f"{CSV_NAME}: {where}, family={f} -> macro_rmse over {len(s)} seeds"
                L.add(f"{m:.2f}", f"macro-RMSE mean [MPa], {FAMILY_LABEL[f]}, {sp}, fraction {fr:g}", src + ", mean")
                L.add(f"{sd:.2f}", f"macro-RMSE sd [MPa], {FAMILY_LABEL[f]}, {sp}, fraction {fr:g}", src + ", sd(ddof=1)")
                cells.append(fmt_pm(m, sd))
            if len(set(seeds_by_family)) != 1:
                raise ValueError(f"{sp}, fraction {fr}: families trained on different seed sets")
            pinn, mlp = by_seed(cur, family="pinn"), by_seed(cur, family="mlp")
            dm, dsd, t, p, n = paired(pinn, mlp)
            src = f"{CSV_NAME}: {where} -> per-seed macro_rmse[pinn]-[mlp], ttest_rel, n={n}"
            L.add(f"{dm:+.2f}", f"Delta PINN-MLP [MPa], {sp}, fraction {fr:g}", src)
            L.add(fmt_p(p), f"paired p, PINN-MLP, {sp}, fraction {fr:g}", src)
            tests.append((sp, fr, p))
            gap = (pinn - mlp).to_numpy()
            gap_x += [float(np.log(fr))] * len(gap)
            gap_y += [float(v) for v in gap]
            body.append(" & ".join([tt(sp) if i == 0 else "", pct + r"\,\%", str(n_train)]
                                   + cells + [f"${dm:+.2f}$", fmt_p(p)]) + r" \\")
        lr = stats.linregress(np.array(gap_x), np.array(gap_y))
        slopes[sp] = lr
        src_reg = (f"{CSV_NAME}: stage=curves, split={sp} -> seed-level gaps macro_rmse[pinn]-[mlp] "
                   f"({len(gap_x)} points), linregress(gap ~ ln(frac))")
        L.add(f"{lr.slope:+.2f}", f"slope of the gap PINN-MLP on ln(fraction) [MPa], {sp}", src_reg)
        L.add(fmt_p2(lr.pvalue), f"regression p (slope = 0), {sp}", src_reg)
        if sp != splits[-1]:
            body.append(r"\addlinespace")
    L.add(n_seeds, "seeds per split x fraction x family, curves stage",
          f"{CSV_NAME}: stage=curves -> count(seed) per split, frac, family")

    # Holm over all PINN − MLP tests of the table (both splits, all fractions).
    adj = holm_adjust(np.array([t[2] for t in tests]))
    src_holm = f"{CSV_NAME}: stage=curves -> holm_adjust over the {len(tests)} paired p of PINN-MLP"
    for (sp, fr, _), pa in zip(tests, adj):
        L.add(fmt_p(pa), f"Holm-adjusted p, PINN-MLP, {sp}, fraction {fr:g}", src_holm)
    n_sig = int((adj < ALPHA).sum())
    k = int(np.argmin(adj))
    L.add(len(tests), "PINN-MLP comparisons in the Holm family", src_holm)
    L.add(n_sig, "PINN-MLP comparisons significant after Holm", src_holm)
    L.add(fmt_p2(adj[k]), "smallest Holm-adjusted p, PINN-MLP", src_holm)

    slope_txt = ", ".join(f"${lr.slope:+.2f}$~MPa ($p={fmt_p2(lr.pvalue)}$) on {tt(sp)}"
                          for sp, lr in slopes.items())
    where_min = f"{tt(tests[k][0])}, {int(round(tests[k][1] * 100))}\\,\\%"
    if n_sig == 0:
        holm_txt = (r" After a Holm correction over the " + word(len(tests)) + r" PINN $-$ MLP "
                    r"comparisons of the table none remains significant, the smallest adjusted "
                    r"$p$ being " + fmt_p2(adj[k]) + " (" + where_min + ").")
    else:
        holm_txt = (r" After a Holm correction over the " + word(len(tests)) + r" PINN $-$ MLP "
                    r"comparisons of the table " + str(n_sig) + r" remain"
                    + ("s" if n_sig == 1 else "") + r" significant, the smallest adjusted "
                    r"$p$ being " + fmt_p2(adj[k]) + " (" + where_min + ").")
    cap = (r"Accuracy against the size of the training set: macro-RMSE (MPa) versus the "
           r"fraction of the training pool, " + word(n_seeds) + r" seeds. $\Delta$ is PINN minus "
           r"MLP with the seed-paired $p$-value. Slope of $\Delta$ on $\log(\text{fraction})$: "
           + slope_txt + "." + holm_txt)
    head = (r"\begin{table*}[t]" "\n"
            r"\centering" "\n"
            r"\caption[Accuracy against the size of the training set]{" + cap + "}\n"
            r"\label{tab:curves}" "\n"
            r"\small" "\n"
            r"\begin{tabular}{@{}lrrrrrrr@{}}" "\n"
            r"\toprule" "\n"
            r"Split & Fraction & $n_{\mathrm{train}}$ & MLP & PINN & VPINN & $\Delta$ & $p$ \\" "\n"
            r"\midrule")
    tail = (r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{table*}")
    src_note = (f"% T5 generated by experiments/paper_tables.py from {CSV_NAME}\n"
                f"% src: {CSV_NAME}: stage==curves; macro_rmse mean/sd(ddof=1) over seeds per split×frac×family;\n"
                f"%      PINN−MLP paired ttest_rel by seed; Holm over the {len(tests)} PINN−MLP p-values;\n"
                f"%      linregress of the seed-level PINN−MLP gap on ln(frac) per split\n")
    return src_note + "\n".join([head] + body + [tail])


# ─────────────────────────────── T6 ──────────────────────────────────────

FAMILIES_2D = ["mlp2d", "pinn2d_reduced", "pinn2d_full"]
FAMILY_2D_LABEL = {"mlp2d": "MLP", "pinn2d_reduced": "PINN, reduced form",
                   "pinn2d_full": "PINN, full form"}
# (column of runs_v2.csv, meaning, decimals, change against the MLP printed)
ADM_COLS = [("macro_rmse", "macro-RMSE [MPa]", 2, False),
            ("eq_res_median", "median |R_r|, full form [MPa/m]", 0, True),
            ("eq_z_full", "median |R_z|, full form [MPa/m]", 0, True),
            ("bc_sigma_rr", "mean |sigma_rr(r=1)| [MPa]", 2, False),
            ("bc_tau_rz", "mean |tau_rz(r=1)| [MPa]", 3, False)]
# FEM row: key of experiments/fem_row_2d.py::fem_row for each residual/violation column
FEM_KEYS = [("eq_r_full", 0), ("eq_z_full", 0), ("bc_sigma_rr", 2), ("bc_tau_rz", 3)]
# Grid of the two-dimensional set when no Field2D is at hand.
GRID_2D = (8, 20)
GRID_SRC = "code/dataset2d.py: build(n_z=8, n_r=20)"


def table_admissibility(df: pd.DataFrame, L: Ledger, fem: Optional[Dict] = None) -> str:
    """
    fem — None or {"rows": {split: {n_test, eq_r_full, eq_z_full, bc_sigma_rr,
    bc_tau_rz}}, "n_sets": int or None, "grid": [n_z, n_r] or None, "src": str}.
    """
    d = df[df["stage"] == "twod"]
    splits = [s for s in SPLIT_ORDER if s in set(d["split"])]
    missing = set(d["split"]) - set(splits)
    if missing:
        raise ValueError(f"splits not in SPLIT_ORDER: {missing}")
    fams = sorted(d["family"].unique())
    if fams != sorted(FAMILIES_2D):
        raise ValueError(f"twod stage expected families {FAMILIES_2D}, got {fams}")

    grid, grid_src = GRID_2D, GRID_SRC
    if fem is not None and fem.get("grid"):
        grid, grid_src = tuple(int(v) for v in fem["grid"]), fem["src"] + " -> grid"
    n_cases = None
    if "random" in splits:
        r = d[d["split"] == "random"]
        n_cases = int(r["n_train"].iloc[0] + r["n_test"].iloc[0])
        L.add(n_cases, "cases in the two-dimensional set",
              f"{CSV_NAME}: stage=twod, split=random -> n_train + n_test")
        if fem is not None and fem.get("n_sets") and int(fem["n_sets"]) != n_cases:
            raise ValueError(f"FEM row computed on {fem['n_sets']} cases, runs on {n_cases}")
    L.add(f"{grid[0]}x{grid[1]}", "grid n_z x n_r of the two-dimensional set", grid_src)

    body = []
    n_seeds = None
    for sp in splits:
        sub = d[d["split"] == sp]
        assert sub["n_test"].nunique() == 1 and sub["n_train"].nunique() == 1
        n_test = int(sub["n_test"].iloc[0])
        n_train = int(sub["n_train"].iloc[0])
        L.add(n_train, f"n_train, twod, {sp}", f"{CSV_NAME}: stage=twod, split={sp} -> n_train")
        L.add(n_test, f"n_test, twod, {sp}", f"{CSV_NAME}: stage=twod, split={sp} -> n_test")
        rows = []
        if fem is not None:
            fr = fem["rows"][sp]
            if "n_test" in fr and int(fr["n_test"]) != n_test:
                raise ValueError(f"{sp}: FEM row on {fr['n_test']} held-out cases, runs on {n_test}")
            cells = ["---"]
            for (key, nd), (_, name, _, _) in zip(FEM_KEYS, ADM_COLS[1:]):
                v = f"{float(fr[key]):.{nd}f}"
                L.add(v, f"{name}, FEM, {sp}", f"{fem['src']}: {sp} -> {key}")
                cells.append(v)
            rows.append(["FEM"] + cells)
        base: Dict[str, float] = {}
        seeds_by_family = []
        for f in FAMILIES_2D:
            cells = []
            for col, name, nd, rel in ADM_COLS:
                s = by_seed(sub, col=col, family=f)
                if col == ADM_COLS[0][0]:
                    seeds_by_family.append(tuple(s.index))
                n_seeds = len(s) if n_seeds is None else n_seeds
                assert len(s) == n_seeds
                m, sd = float(s.mean()), float(s.std(ddof=1))
                if f == "mlp2d":
                    base[col] = m
                src = f"{CSV_NAME}: stage=twod, split={sp}, family={f} -> {col} over {len(s)} seeds"
                L.add(f"{m:.{nd}f}", f"{name} mean, {FAMILY_2D_LABEL[f]}, {sp}", src + ", mean")
                L.add(f"{sd:.{nd}f}", f"{name} sd, {FAMILY_2D_LABEL[f]}, {sp}", src + ", sd(ddof=1)")
                cell = fmt_pm(m, sd, nd)
                if rel and f != "mlp2d":
                    pc = 100.0 * (m / base[col] - 1.0)
                    L.add(f"{pc:+.0f}", f"change of the mean {name} against the MLP, %, {FAMILY_2D_LABEL[f]}, {sp}",
                          f"{CSV_NAME}: stage=twod, split={sp} -> mean {col}[{f}] / mean {col}[mlp2d] - 1")
                    cell += f" (${pc:+.0f}\\,\\%$)"
                cells.append(cell)
            rows.append([FAMILY_2D_LABEL[f]] + cells)
        if len(set(seeds_by_family)) != 1:
            raise ValueError(f"{sp}: two-dimensional families trained on different seed sets")
        for i, row in enumerate(rows):
            body.append(" & ".join([tt(sp) if i == 0 else ""] + row) + r" \\")
        if sp != splits[-1]:
            body.append(r"\addlinespace")
    L.add(n_seeds, "seeds per split x family, twod stage",
          f"{CSV_NAME}: stage=twod -> count(seed) per split, family")

    n_fam = word(len(FAMILIES_2D))
    cap = (r"Accuracy and physical admissibility of the " + n_fam + r" two-dimensional families "
           r"($z$ as input, $" + f"{grid[0]}\\times{grid[1]}" + r"$ grid) on the held-out cases"
           + (f" of the {n_cases}-case set" if n_cases else "")
           + r", mean~$\pm$~sd over " + word(n_seeds) + r" seeds: macro-RMSE (mean of the four "
           r"component RMSEs), median full-form equilibrium residuals and mean traction-free "
           r"violations at $r/R=1$. In brackets, change of the mean residual relative to the MLP "
           r"on the same split. The " + n_fam + r" configurations differ only in the physics "
           r"terms of the loss.")
    if fem is not None:
        cap += r" FEM: the same quantities for the finite-element data of the same held-out cases."
    head = (r"\begin{table*}[t]" "\n"
            r"\centering" "\n"
            r"\caption[Accuracy and physical admissibility of the two-dimensional families]{" + cap + "}\n"
            r"\label{tab:admissibility}" "\n"
            r"\small" "\n"
            r"\setlength{\tabcolsep}{4pt}" "\n"
            r"\begin{tabular}{@{}llrrrrr@{}}" "\n"
            r"\toprule" "\n"
            r" & & & \multicolumn{2}{c}{Equilibrium residual, MPa/m} & "
            r"\multicolumn{2}{c}{Traction-free violation, MPa} \\" "\n"
            r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}" "\n"
            r"Split & Configuration & Macro-RMSE, MPa & $|R_r|$ & $|R_z|$ & "
            r"$|\sigma_{rr}(r{=}1)|$ & $|\tau_{rz}(r{=}1)|$ \\" "\n"
            r"\midrule")
    tail = (r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{table*}")
    src_note = (f"% T6 generated by experiments/paper_tables.py from {CSV_NAME}\n"
                f"% src: {CSV_NAME}: stage==twod; macro_rmse, eq_res_median (|R_r| full form), eq_z_full (|R_z| full form),\n"
                f"%      bc_sigma_rr, bc_tau_rz: mean/sd(ddof=1) over seeds; brackets: ratio of the seed means to mlp2d − 1\n"
                + (f"% src: FEM row — {fem['src']}\n" if fem is not None else ""))
    return src_note + "\n".join([head] + body + [tail])


def fem_from_json(path: str) -> Dict:
    """FEM row written by experiments/fem_row_2d.py --json."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return {"rows": d.get("splits", d), "n_sets": d.get("n_sets"), "grid": d.get("grid"),
            "src": f"experiments/fem_row_2d.py --json ({os.path.basename(path)})"}


def fem_from_field2d(path: str, removed_csv: str, runs_csv: str, splits: List[str]) -> Dict:
    """
    FEM row computed from a Field2D .npz with the functions of fem_row_2d.py.
    A set that still holds the cases of removed_csv (the 1848-case set) is
    reduced to the 1756 cases first; a set without any of them is taken as is.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import fem_row_2d as fr

    f = fr.Field2D.load(path)
    with open(removed_csv, encoding="utf-8") as fh:
        drop = {r["job"] for r in csv.DictReader(fh)}
    names = {str(n) for n in f.names}
    g = fr.clean_1756(f, removed_csv) if drop & names else f
    print(f"FEM row: {os.path.basename(path)}, cases {f.n_sets} -> {g.n_sets}")
    fr.fingerprint(g, runs_csv)
    return {"rows": fr.fem_row(g, splits), "n_sets": int(g.n_sets),
            "grid": [int(g.y.shape[1]), int(g.y.shape[2])],
            "src": f"experiments/fem_row_2d.py::fem_row on {os.path.basename(path)}"}


# ─────────────────────────────── main ────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--csv", default=os.path.join(here, "..", "results", CSV_NAME))
    ap.add_argument("--out", default="tables",
                    help="output directory, relative to the current directory (default: tables)")
    fem = ap.add_mutually_exclusive_group()
    fem.add_argument("--fem-json", default="",
                     help="FEM row of tab:admissibility, json of experiments/fem_row_2d.py --json")
    fem.add_argument("--field2d", default="",
                     help="Field2D .npz (1848 or 1756 cases): compute the FEM row of tab:admissibility")
    ap.add_argument("--removed", default=os.path.join(here, "..", "audit", "removed_1848_to_1756.csv"),
                    help="cases removed from the 1848-case set (used with --field2d)")
    a = ap.parse_args()

    df = load(a.csv)
    fem_data = None
    if a.fem_json:
        fem_data = fem_from_json(a.fem_json)
    elif a.field2d:
        twod = set(df.loc[df["stage"] == "twod", "split"])
        fem_data = fem_from_field2d(a.field2d, a.removed, a.csv,
                                    [s for s in SPLIT_ORDER if s in twod])
    os.makedirs(a.out, exist_ok=True)
    L = Ledger()
    L.add(len(df), "runs in the log", f"{CSV_NAME}: row count (no run has an error field)")

    tables = {
        "tab_main.tex": table_main(df, L),
        "tab_paired.tex": table_paired(df, L),
        "tab_corrupt.tex": table_corrupt(df, L),
        "tab_backend.tex": table_backend(df, L),
        "tab_curves.tex": table_curves(df, L),
        "tab_admissibility.tex": table_admissibility(df, L, fem_data),
    }
    for name, tex in tables.items():
        path = os.path.join(a.out, name)
        with open(path, "w") as fh:
            fh.write(tex + "\n")
        print(f"written: {path}")
    all_path = os.path.join(a.out, "all_tables.tex")
    with open(all_path, "w") as fh:
        fh.write("% Requires \\usepackage{booktabs}. Generated by experiments/paper_tables.py\n"
                 f"% from results/{CSV_NAME}; do not edit by hand — rerun the script.\n\n")
        fh.write("\n\n".join(tables.values()) + "\n")
    print(f"written: {all_path}")
    L.dump(os.path.join(a.out, "numbers_used.json"))
    print(f"numbers recorded: {len(L.rows)} → {os.path.join(a.out, 'numbers_used.json')}")


if __name__ == "__main__":
    main()
