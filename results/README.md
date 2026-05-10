# results/

## bundles/

Бандлы обученных моделей в формате `.pkl`. Загружаются numpy-only через `models.loader.load_bundle()` — без зависимости от torch / jax / deepxde.

```
results/bundles/
├── stress/    # 6 моделей для остаточных напряжений
│   ├── mlp_model.pkl              MLP-Optuna
│   ├── mlp_model_grid.pkl         MLP-grid
│   ├── pinn_model_torch.pkl       PINN-PyTorch
│   ├── pinn_model_dde.pkl         PINN-DeepXDE
│   ├── pinn_model_jax.pkl         PINN-JAX/Flax
│   └── vpinn_model.pkl            VPINN
└── strain/    # 6 моделей для пластических деформаций
    ├── mlp_strains_baseline.pkl
    ├── mlp_strains_optuna.pkl
    ├── pinn_strains_torch.pkl
    ├── pinn_strains_dde.pkl
    ├── pinn_strains_jax.pkl
    └── vpinn_strains.pkl
```

Все стресс-бандлы имеют одинаковый holdout split (305 наборов из 2035, seed=42) — это позволяет сопоставлять прогнозы разных архитектур на одном тестовом множестве.

## figures/ и tables/

Финальные рисунки для текста ВКР и таблицы метрик. Промежуточные/черновые версии в git не коммитятся (см. `.gitignore`).

## Структура одного бандла

```python
{
    'framework':         'deepxde_pytorch' | 'jax_flax' | 'pytorch_native' | ...,
    'activation':        'tanh' | 'softplus' | 'gelu',
    'mean_X':            np.ndarray (1, 6),  # для стандартизации входа
    'std_X':             np.ndarray (1, 6),
    'scaler_y_mean':     np.ndarray (1, 4),  # для де-стандартизации выхода
    'scaler_y_std':      np.ndarray (1, 4),
    'holdout_set_indices': np.ndarray (305,),
    'trainval_set_indices': np.ndarray (1730,),
    'holdout_metrics':   np.ndarray (11,),   # метрики на holdout
    'plane_order':       [2, 0, 1, 3],       # для stress / [1,0,2,3] для strain
    'torch_state_dict' | 'flax_params_numpy' | 'state_dict_np': веса
    ...
}
```
