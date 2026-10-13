# Physics-informed versus data-driven surrogates for residual stress after wire drawing — code

[![DOI](https://zenodo.org/badge/1417829495.svg)](https://doi.org/10.5281/zenodo.23352451)

This package holds the code, the run log and the verification scripts behind the paper

> D. Demin, G. Romanenko, *Physics-informed versus data-driven surrogates for residual stress
> after wire drawing: accuracy, physical admissibility and behaviour outside the training range*,
> submitted.

Starting from the raw finite-element exports, it rebuilds the two datasets of the paper
(1756 cases each), reruns the full experiment matrix (341 training runs), regenerates the result
tables, the field figures (Figs. 4–6) and the drawing-stress verification of Section 3.1.
Everything runs on a CPU; no GPU is needed.

Code comments, docstrings and console messages are in Russian. This README is the English entry
point, and every command below is run from the package root.

## Citation

If you use this code or the data, please cite the paper:

```bibtex
@unpublished{demin2026surrogates,
  author = {Demin, Dmitriy and Romanenko, Georgii},
  title  = {Physics-informed versus data-driven surrogates for residual stress after wire
            drawing: accuracy, physical admissibility and behaviour outside the training range},
  note   = {Submitted},
  year   = {2026}
}
```

and the Zenodo records of the code, DOI [10.5281/zenodo.23352452](https://doi.org/10.5281/zenodo.23352452)
for version 1.0.0 ([10.5281/zenodo.23352451](https://doi.org/10.5281/zenodo.23352451) for all
versions), and of the data, DOI `<DOI>`. Machine-readable metadata are in `CITATION.cff`.

## Contents

| Path | What it holds |
|---|---|
| `code/` | `build_datasets.py` (both datasets from the raw reports in one go), dataset classes (`dataset.py` 1D, `dataset2d.py` 2D), report reader (`raw_probe.py`), case filters, splits, PyTorch trainers (`trainer.py`, `trainer2d.py`), JAX twin of the strong-form PINN (`jax_twin.py`), label corruption, learning curves, statistics, physics checks, analytical drawing-stress relations |
| `experiments/` | `run_matrix.py` (the experiment matrix), `hp_1d.json` and `hp_2d.json` (hyperparameters of the paper), `paper_tables.py`, `make_results.py` and `analyze.py` (tables and report), `fem_row_2d.py` (FEM reference row of Table 13), `contours.py`, `dump_predictions.py`, `residual_maps.py` (field figures), `figures.py` (supplementary plots of the run log) |
| `audit/` | `run_raw_audit.py` (one pass over the raw reports; `build_datasets.py` uses it), `make_report.py` (Markdown report of that pass), `removed_1848_to_1756.csv` (the 92 runs removed from the 1848 filtered cases, with the reason; see step 2) |
| `verification/` | drawing-stress verification of Section 3.1 from the 112 history reports: `read_history.py`, `drawing_force.py`, `flow_stress.py`, `analytic_sigma_f.py`, `verification_recalc.py`, `sec31_numbers.py`; `results/` holds the outputs the paper quotes |
| `results/runs_v2.csv` | log of the 341 runs reported in the paper, one row per run: stage, family, backend, split, seed, training fraction, corruption rate, set sizes, epochs, run time, accuracy metrics, equilibrium and boundary-condition audits, hyperparameters |
| `tests/` | invariant tests: `test_pipeline.py` (numpy and scipy only), `test_training.py` (PyTorch and JAX) |

## Data

The finite-element data are not part of the repository tree. They are published as two archives
in a separate Zenodo record, DOI `<DOI>`:

* **`Raw_Data.zip`** (1.2 GB) — 2443 Abaqus field output reports (`.rpt`), one per
  simulation, in the folders `Vel_5`, `Vel_10`, `Vel_20`, `Vel_40` and `Vel_250` (drawing speed in
  m/min). The five `Vel_*` folders lie at the root of the archive, with no enclosing folder, and
  hold the `.rpt` files directly. The process parameters are encoded in the file name (`2a` twice
  the die semi-angle in degrees, `rd` the reduction in diameter × 10⁴, `cal` the calibration ratio
  × 100, `v` the drawing speed in m/min, `fric` or `f` a leading 0 followed by the friction
  coefficient × 1000) and are parsed by `code/raw_probe.py`.
* **`history_reports_112.zip`** (19 MB) — 112 Abaqus X-Y reports with time histories
  of the die reaction force, the nodal coordinates, the equivalent plastic strain and the
  temperature of a radial row of material points, used for the drawing-stress verification of
  Section 3.1. The reports lie in the folder `history_reports_112` at the root of the archive,
  together with `MANIFEST.sha256`; check it after unpacking with
  `cd history_reports_112 && sha256sum -c MANIFEST.sha256`.

The license of the data archives is stated on their record. Unpack both in a working folder of
your choice, `Raw_Data.zip` into a folder of its own since it has no top-level folder, and point
the commands below at them:

```bash
unzip Raw_Data.zip -d Raw_Data            # gives Raw_Data/Vel_5 ... Raw_Data/Vel_250
unzip history_reports_112.zip             # gives history_reports_112/
export RAW=$PWD/Raw_Data                  # the folder that contains Vel_5 ... Vel_250
export HIST=$PWD/history_reports_112      # the folder with the 112 reports
```

## Requirements

Python 3.11 and the packages in `requirements.txt` (tested versions are given there as comments):

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

A CPU build of PyTorch is sufficient (for example
`pip install torch --index-url https://download.pytorch.org/whl/cpu`). Building the datasets,
`fem_row_2d.py` and the verification scripts need numpy only; `tests/test_pipeline.py` needs numpy
and scipy; training needs PyTorch, and the `backend` stage and `tests/test_training.py` also JAX
and optax; tables, report and figures need pandas, scipy and matplotlib.

## Reproduction

Run the steps in this order. Step 3 is the only long one.

**1. Tests** (under a minute)

```bash
python tests/test_pipeline.py
python tests/test_training.py
```

**2. Datasets** (about 3 min)

```bash
python code/build_datasets.py --raw "$RAW" --out data --keep-1848
```

This runs the raw audit pass (`audit/run_raw_audit.py`, saved as `data/audit.npz`), builds the
1D set (20 radial bins) and the 2D set (8 × 20 grid in the axial window), applies the case filters
of Table 3 (2443 → 1848), removes the 92 runs of `audit/removed_1848_to_1756.csv` from both
(→ 1756) and writes `data/stress_1756.npz` and `data/field2d_1756.npz` (with `--keep-1848` also the
1848-case sets). It then prints a fingerprint of both sets: the standard deviation of each stress
component next to the value implied by `results/runs_v2.csv` (`rmse_c / nrmse_c_global`, 1D stages
`main`, `curves`, `corrupt`, `backend`; 2D stage `twod`). With the published data all eight values
agree to about 1e-16, and the last line reads `всё совпало` ("everything matched"); `--strict`
turns any mismatch into exit code 1.

Of the 92 runs in `audit/removed_1848_to_1756.csv` (column `group`), 89 are unfinished runs
detected by the window mask (spread of the surface radius in the axial window above 0.1 mm), and
3 are completed runs at Q = 1.5 %, v = 250 m/min that an earlier tail check (bulge of the wire
head, 17.82–17.88 mm) removed together with them.

**3. Experiment matrix** (about 20 run-hours; about 5 h wall-clock with `--workers 4`)

```bash
python experiments/run_matrix.py --stage main    --dataset data/stress_1756.npz  --hp experiments/hp_1d.json --seeds 0,1,2,3,4 --max-epochs 800 --workers 4 --out runs.csv
python experiments/run_matrix.py --stage curves  --dataset data/stress_1756.npz  --hp experiments/hp_1d.json --seeds 0,1,2     --max-epochs 800 --workers 4 --out runs.csv
python experiments/run_matrix.py --stage corrupt --dataset data/stress_1756.npz  --hp experiments/hp_1d.json --seeds 0,1,2     --max-epochs 800 --workers 4 --out runs.csv
python experiments/run_matrix.py --stage backend --dataset data/stress_1756.npz  --hp experiments/hp_1d.json --seeds 0,1,2,3,4 --max-epochs 800 --workers 4 --out runs.csv
python experiments/run_matrix.py --stage twod    --dataset data/field2d_1756.npz --hp experiments/hp_2d.json --seeds 0,1,2     --max-epochs 300 --workers 4 --out runs.csv
```

| Stage | Rows | Composition | Run-hours in the log |
|---|---:|---|---:|
| `main` | 195 | 13 splits × 3 families (MLP, PINN, VPINN) × 5 seeds | 10.6 |
| `curves` | 72 | 2 splits × 4 training fractions (10, 25, 50, 100 %) × 3 families × 3 seeds | 1.6 |
| `corrupt` | 36 | 4 corruption rates (0, 1, 5, 10 %) × 3 families × 3 seeds | 1.3 |
| `backend` | 20 | 2 splits × 2 implementations (PyTorch, JAX) of the strong-form PINN × 5 seeds | 2.3 |
| `twod` | 18 | 2 splits × 3 families with z as input × 3 seeds | 4.2 |
| total | 341 | the rows of `results/runs_v2.csv` | 19.9 |

All stages append to the same `runs.csv`, one row per finished run. A rerun of the same command
skips the rows already present, so an interrupted stage is resumed by repeating it. Each run uses
one PyTorch thread; `--max-epochs` is a cap, early stopping (patience 40 epochs in 1D, 30 in 2D)
ends most runs earlier. `hp_1d.json` and `hp_2d.json` hold the settings of the paper: 3 layers of
96 units, learning rate 3e-3 in 1D and 2e-3 in 2D, physics and boundary weights 0 for the MLP and
the defaults of `Config`/`Config2D` (0.3 and 0.5) for the physics-informed families.

**4. Tables** (seconds)

```bash
python experiments/paper_tables.py --csv results/runs_v2.csv --out tables --field2d data/field2d_1756.npz
python experiments/make_results.py --csv results/runs_v2.csv --dataset data/stress_1756.npz --out results_report.md
python experiments/fem_row_2d.py --field2d data/field2d_1848.npz --json data/fem_row.json
```

`paper_tables.py` writes Tables 8 and 10–14 as booktabs LaTeX (`tab_main`, `tab_paired`,
`tab_curves`, `tab_backend`, `tab_admissibility`, `tab_corrupt`) together with
`numbers_used.json`, which records the source of every printed number. `--field2d` adds the FEM
row of Table 13; `--fem-json data/fem_row.json` does the same from the output of
`fem_row_2d.py`. `make_results.py` writes a Markdown report (in Russian) that also contains the
numbers of Table 9 (cost of holding out a factor level); its `R²` and constant-predictor columns
need `--dataset` and are printed as dashes without it. `fem_row_2d.py` prints the FEM row of Table 13 on
its own, with the 2D fingerprint; it takes the 1848-case set and removes the 92 runs itself (or
use `--raw "$RAW"` instead of `--field2d`). Replace `results/runs_v2.csv` by `runs.csv` to
tabulate your own rerun of step 3. `paper_tables.py` needs the full matrix, all five stages: on
a partial `runs.csv` (a stage missing, or a stage interrupted so that the families of a split
were not run on the same seeds) it stops with an error and writes no tables. `make_results.py`
also reports on a partial log, comparing the families only on the seeds that all of them have.

**5. Fig. 4, FEM residual stress fields** (seconds)

```bash
python experiments/contours.py --mode raw --paper \
    --rpt "$RAW/Vel_10/1020_2a_24_rd_1000_cal_50_v_10_fric_050.rpt" --out fig4_fem_fields.png
```

The case is Q = 10 %, k = 0.5, α = 12°, μ = 0.05, v = 10 m/min, as in the paper.

**6. Figs. 5 and 6, model fields and equilibrium residuals** (about 30 min, one thread per run)

```bash
for fam in mlp2d pinn2d_reduced pinn2d_full; do
    python experiments/dump_predictions.py --dataset data/field2d_1756.npz --hp experiments/hp_2d.json \
        --family $fam --split random --seed 0 --out data/pred_$fam.npz
done
python experiments/contours.py --mode compare --paper \
    --npzs data/pred_mlp2d.npz,data/pred_pinn2d_reduced.npz,data/pred_pinn2d_full.npz \
    --labels "MLP,PINN reduced,PINN full" --out fig5_compare_fields.png
python experiments/residual_maps.py --paper --field2d data/field2d_1756.npz \
    --npzs data/pred_mlp2d.npz,data/pred_pinn2d_reduced.npz,data/pred_pinn2d_full.npz \
    --labels "MLP,PINN reduced,PINN full" --out fig6_residual_maps.png
```

`dump_predictions.py` trains one configuration exactly as the `twod` stage does and prints the
macro-RMSE of the same configuration from `results/runs_v2.csv` next to the fresh one. Both figures
show the test case with the median error of the last model in the list; `--which` (index of a test
case) or `--where` (for example `--where "alpha=8,v=20"`) select another one. See the
reproducibility notes below.

**7. Section 3.1, drawing-stress verification** (under a minute)

```bash
python verification/verification_recalc.py "$HIST" --json recalc.json --csv recalc.csv
python verification/sec31_numbers.py "$HIST" --json recalc.json > sec31_numbers.txt
```

`verification_recalc.py` computes for each of the 112 runs the drawing stress from the
simulation, the three analytical predictions with one flow stress per run, and the temperature
rise; `sec31_numbers.py` prints the ranges and summary numbers quoted in Section 3.1 and the rows
of Tables 6 and 7. The outputs are identical to `verification/results/recalc_2026-09-28.json`,
`recalc_2026-09-28.csv` and `sec31_numbers_2026-10-02.txt`. The scripts read the reports either as
`.rpt` or as `.rpt.gz`.

**Optional.** `python audit/make_report.py --npz data/audit.npz --out raw_audit_report.md` writes a
Markdown report of the raw audit pass (axial window, component order, factor levels, duplicates
under different friction labels, size of the equilibrium terms dropped by the reduced form).
`python experiments/figures.py --csv results/runs_v2.csv --dataset data/stress_1756.npz --outdir figs`
draws supplementary plots of the run log that are not in the paper.

## Where each result comes from

| Paper | Produced by |
|---|---|
| Table 3 (case filtering) | `code/build_datasets.py` |
| Tables 6 and 7, numbers of Section 3.1 on the drawing stress, flow stress and temperature | `verification/verification_recalc.py`, `verification/sec31_numbers.py` |
| Tables 8, 10, 11, 12, 13, 14 | `experiments/paper_tables.py` (FEM row of Table 13: `experiments/fem_row_2d.py`) |
| Table 9 | `experiments/make_results.py` |
| Fig. 4 | `experiments/contours.py --mode raw` |
| Figs. 5 and 6 | `experiments/dump_predictions.py`, `experiments/contours.py --mode compare`, `experiments/residual_maps.py` |
| Table 2, FEM self-check numbers in Section 3.1 | no script in this package |

Tables 1, 4 and 5 describe the design, the splits and the model families, and Figs. 1–3 are
drawings; the split sizes of Table 4 also appear in Table 8.

## Compute budget

| Step | Time |
|---|---|
| datasets (step 2) | about 3 min, single core |
| experiment matrix (step 3) | about 20 run-hours; about 5 h wall-clock with `--workers 4` |
| Figs. 5–6 (step 6) | about 30 min (4, 9 and 17 min for the three families) |
| tables, Fig. 4, verification | about a minute in total |

The run-hours are the sum of the `seconds` column of `results/runs_v2.csv` (one thread per run,
double precision, CPU).

## Reproducibility notes

* **2D runs reproduce bit for bit.** `data/field2d_1756.npz` built by step 2 is identical to the
  set used for the `twod` stage, and with the tested library versions a rerun of a `twod`
  configuration returns the logged numbers exactly.
* **1D runs reproduce statistically, within the seed spread.** The stress values of
  `data/stress_1756.npz` are identical to those used for the logged runs (the fingerprint of
  step 2), but the radii of the 20 radial bins, which enter the network as input and the
  equilibrium residual as coordinates, are rebuilt from the audit pass and are not bit-identical to
  those of the logged runs. A rerun of a 1D configuration therefore does not repeat its row of
  `runs_v2.csv` digit for digit; it lies within the spread over seeds (for example, MLP on the
  `random` split, seed 0: 8.08 MPa in a rerun against 8.48 MPa in the log, where the five logged
  seeds give 8.04 ± 0.33 MPa). Tables 8–14 generated from `results/runs_v2.csv` reproduce every
  number printed in the paper; generated from a fresh `runs.csv`, the 1D tables agree with them
  within the seed spread.
* **Figs. 5 and 6** of the paper were drawn from an earlier single-seed training of the three 2D
  families whose predictions were not saved (the 1848-case set cleaned by the window mask only,
  random split with 1498 training and 261 test cases). Step 6 redraws them from a fresh
  single-seed training on the 1756-case set used for all `twod` runs; the maps and the medians in
  the panel titles are close to the published ones but not identical, and the case shown in the
  paper is not in the new test split, so a different case is selected.
* **Table 2** (share of the variance along z) and the **FEM self-check numbers in Section 3.1**
  (the unloaded exported section and the internal consistency tests of the exported fields) have no
  script in this package.

## License

The code is released under the MIT License (`LICENSE`).
