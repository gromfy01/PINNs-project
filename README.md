# PINNs-project

Сравнительный анализ PINN, MLP и VPINN для прогноза остаточных напряжений и накопленных деформаций после волочения проволоки из стали AISI 1020 на данных МКЭ-моделирования в Abaqus. Работа выполнена в рамках ВКР на факультете компьютерных наук НИУ ВШЭ.

## Структура проекта

```
.
├── configs/           # YAML-конфигурации экспериментов
├── data/              # Данные МКЭ (не входят в репозиторий, см. ссылку ниже)
├── docs/              # Документация
├── evaluation/        # Метрики, физический аудит, классификация выбросов
├── models/            # PINN/MLP/VPINN архитектуры
├── notebooks/         # Демонстрационные ноутбуки
├── physics/           # Уравнения равновесия, совместности, Джонсон-Кук
├── preprocessing/     # Парсер .rpt → 4-компонентный pkl
├── results/           # Метрики, графики, обученные веса (см. Releases)
├── thesis/            # LaTeX-исходники разделов ВКР
├── training/          # Скрипты обучения и Optuna
├── visualization/     # Стили оформления, графики
├── .env.example       # Шаблон переменных окружения
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

## Инструкция по установке

### 1. Зависимости

Требуется Python 3.10+. Рекомендуется виртуальное окружение.

```bash
python -m venv venv
source venv/bin/activate    # или .\venv\Scripts\activate в Windows
pip install -r requirements.txt
```

Для DeepXDE-моделей backend ставится **до** импорта:

```python
import os
os.environ['DDE_BACKEND'] = 'pytorch'
import deepxde as dde
```

### 2. Данные

Исходные `.rpt`-отчёты Abaqus выложены как asset к релизу
[`raw-data-v1`](https://github.com/gromfy01/PINNs-project/releases/tag/raw-data-v1).
Скачать и распаковать в `data/Raw_Data/`:

```bash
curl -L -o Raw_Data.zip \
  https://github.com/gromfy01/PINNs-project/releases/download/raw-data-v1/Raw_Data.zip
unzip Raw_Data.zip -d data/Raw_Data/
```

Препроцессинг `.rpt` → `.pkl` запускается через `preprocessing/preproc.ipynb`,
результат — четыре массива в `data/processed/`:

```
data/processed/
├── X_stress.pkl   # (4, N_sets, 5)   — параметры процесса по плоскостям
├── y_stress.pkl   # (4, N_sets, 20)  — компоненты тензора напряжений
├── X_strain.pkl   # (4, N_sets, 5)
└── y_strain.pkl   # (4, N_sets, 20)  — компоненты тензора деформаций
```

Порядок компонент: `[σ_rr, σ_θθ, σ_zz, τ_rz]` для напряжений, `[ε_rr, ε_θθ, ε_zz, ε_rz]` для деформаций.

### 3. Обученные модели

Веса шести моделей (PINN-DeepXDE, PINN-JAX, PINN-PyTorch, MLP-grid, MLP-Optuna, VPINN) - в [Releases](https://github.com/gromfy01/PINNs-project/releases). Распаковать в `results/bundles/`.

### 4. Переменные окружения

```bash
cp .env.example .env
```

При пустых значениях применяются разумные дефолты (in-memory Optuna study, отключённый W&B, `cuda` если доступна).

### 5. Запуск

```bash
python training/train_stress.py --config configs/stress_pinn.yaml
```

Быстрый старт - `notebooks/00_quickstart.ipynb`.

## Замечания

- Физические потери считаются в нормированных координатах. В физических единицах градиенты несоразмерны, обучение коллапсирует.
- Разбиение train/val/test - group-aware: все 20 радиальных точек одного набора параметров остаются в одной части.
- При сравнении прогноза с эталоном применяется перестановка компонент `PLANE_ORDER = [2, 0, 1, 3]` к **обоим** массивам.
- Парсинг `.rpt`-отчётов Abaqus - в `preprocessing/`. Извлечение из `.odb` сделано отдельным Abaqus Python API скриптом, вне репозитория.

## Лицензия

MIT (см. `LICENSE`).

## Автор

Романенко Г., БКНАД222, факультет компьютерных наук НИУ ВШЭ. Научный руководитель: Дмитрий Дёмин.
