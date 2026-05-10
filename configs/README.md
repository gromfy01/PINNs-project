# configs/

YAML-конфиги обучения. Запуск:

```
python training/train_stress.py --config configs/stress_pinn.yaml
python training/train_strain.py --config configs/strain_mlp.yaml
```

Структура конфига:

```yaml
version: stress_pinn_v1
model:        # архитектура и гиперпараметры сети
training:     # epochs, batch_size, lr, патиенс, веса физических потерь
data:         # holdout_frac, seed, n_folds
```

Гиперпараметры в текущих файлах — лучшие из Optuna runs, как они сохранены в соответствующих `.pkl` бандлах (`best_params` поле).
