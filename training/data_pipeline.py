import numpy as np


PLANE_ORDER_STRESS = [2, 0, 1, 3]
PLANE_ORDER_STRAIN = [1, 0, 2, 3]

STRESS_NAMES = ["sigma_rr", "sigma_tt", "sigma_zz", "tau_rz"]
STRAIN_NAMES = ["eps_rr", "eps_tt", "eps_zz", "gamma_rz"]

N_R = 20


def reorder_planes(y_full, plane_order):
    return np.stack([y_full[plane_order[i]] for i in range(4)], axis=0)


def make_grid_features(proc_params, n_r=N_R):
    n_sets = proc_params.shape[0]
    r_grid = np.linspace(0.0, 1.0, n_r, dtype=np.float32)
    proc_rep = np.repeat(proc_params, n_r, axis=0)
    r_rep = np.tile(r_grid, n_sets)[:, None]
    return np.hstack([proc_rep, r_rep]).astype(np.float32)


def stack_targets(y_full, plane_order):
    return np.stack(
        [y_full[plane_order[i]].reshape(-1) for i in range(4)],
        axis=1,
    ).astype(np.float32)


def holdout_split(n_sets, holdout_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    n_hold = max(1, int(round(n_sets * holdout_frac)))
    hold_sets = np.sort(rng.choice(np.arange(n_sets), size=n_hold, replace=False))
    trainval_sets = np.setdiff1d(np.arange(n_sets), hold_sets)
    return trainval_sets, hold_sets


def set_indices_to_rows(set_indices, n_r=N_R):
    return np.concatenate(
        [np.arange(s * n_r, (s + 1) * n_r) for s in set_indices]
    )


def build_stress_dataset(X_stress, y_stress, holdout_frac=0.15, seed=42):
    n_sets = X_stress.shape[1]
    proc = X_stress[0].astype(np.float32)
    X = make_grid_features(proc)
    y = stack_targets(y_stress, PLANE_ORDER_STRESS)

    trainval_sets, hold_sets = holdout_split(n_sets, holdout_frac, seed)
    trainval_rows = set_indices_to_rows(trainval_sets)
    hold_rows = set_indices_to_rows(hold_sets)

    return {
        "X_trainval": X[trainval_rows], "y_trainval": y[trainval_rows],
        "X_hold": X[hold_rows], "y_hold": y[hold_rows],
        "trainval_sets": trainval_sets, "hold_sets": hold_sets,
        "X_full": X, "y_full": y,
    }


def build_strain_dataset(X_strain, y_strain, holdout_frac=0.15, seed=42):
    n_sets = X_strain.shape[1]
    proc = X_strain[0].astype(np.float32)
    X = make_grid_features(proc)
    y = stack_targets(y_strain, PLANE_ORDER_STRAIN)

    trainval_sets, hold_sets = holdout_split(n_sets, holdout_frac, seed)
    trainval_rows = set_indices_to_rows(trainval_sets)
    hold_rows = set_indices_to_rows(hold_sets)

    return {
        "X_trainval": X[trainval_rows], "y_trainval": y[trainval_rows],
        "X_hold": X[hold_rows], "y_hold": y[hold_rows],
        "trainval_sets": trainval_sets, "hold_sets": hold_sets,
        "X_full": X, "y_full": y,
    }


def kfold_indices(set_indices, n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    sets = np.array(set_indices, copy=True)
    rng.shuffle(sets)
    folds = np.array_split(sets, n_folds)
    for k in range(n_folds):
        val_sets = np.sort(folds[k])
        train_sets = np.sort(np.concatenate([folds[i] for i in range(n_folds) if i != k]))
        yield train_sets, val_sets


def standardize_features(X_train):
    mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = X_train.std(axis=0, keepdims=True).astype(np.float32) + 1e-12
    return mean, std


def standardize_targets(y_train):
    mean = y_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = y_train.std(axis=0, keepdims=True).astype(np.float32) + 1e-12
    return mean, std
