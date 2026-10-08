"""
protocol.py — ЕДИНЫЙ протокол обучения для всех шести моделей (пункт 4 плана).

Проблема, на которую указал Reviewer #1: в рукописи (§2.2.4) написано
«Adam followed by LBFGS for the DeepXDE backend, Adam with cosine schedule
for the others». При таком описании вывод «backend matters» сравнивает СХЕМЫ
ОБУЧЕНИЯ, а не движки автодифференцирования, и разница 0.4–0.8 МПа ничего
не говорит про autodiff.

Здесь протокол зафиксирован в одном месте и импортируется всеми шестью
launch-ноутбуками. Любое отклонение — только явным аргументом и с записью
в отчёт, молча разойтись нельзя: assert_conforms() падает.

Что уравнено:
    оптимизатор, планировщик LR, число эпох и терпение ранней остановки,
    размер батча, число коллокационных точек, пространство и число trials
    Optuna, сиды, критерий отбора лучшей конфигурации.

Что НЕ уравнивается (и не может быть):
    внутренняя механика вычисления производных — это и есть изучаемый фактор;
    VPINN использует слабую форму, поэтому у него есть свои λ и квадратура —
    он сравнивается как отдельное семейство, а не как «ещё один бэкенд».
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
SEEDS_EXTENDED: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

MODEL_FAMILIES: Dict[str, str] = {
    "mlp_grid":   "data-driven",
    "mlp_optuna": "data-driven",
    "pinn_torch": "physics-informed (strong form)",
    "pinn_dde":   "physics-informed (strong form)",
    "pinn_jax":   "physics-informed (strong form)",
    "vpinn":      "physics-informed (weak form)",
}

PHYSICS_MODELS: Tuple[str, ...] = ("pinn_torch", "pinn_dde", "pinn_jax", "vpinn")
BACKEND_MODELS: Tuple[str, ...] = ("pinn_torch", "pinn_dde", "pinn_jax")


@dataclass(frozen=True)
class TrainingProtocol:
    """Единственный источник правды по протоколу обучения."""

    optimizer: str = "AdamW"
    lr_schedule: str = "none"          # никакого cosine — его нет и в коде
    max_epochs: int = 400
    patience: int = 30
    batch_size: int = 64
    grad_clip: float = 1.0
    weight_decay: float = 1e-4

    n_r: int = 20                      # радиальных точек на набор
    n_collocation: int = 20            # коллокационные точки = точки данных
    n_quadrature: int = 32             # узлы Гаусса–Лежандра (только VPINN)

    n_trials: int = 20
    n_inner_splits: int = 3            # group-aware CV внутри train для Optuna
    tuning_objective: str = "worst-fold macro RMSE"

    n_folds: int = 5
    seeds: Tuple[int, ...] = SEEDS
    split_seed: int = 42

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_text(self) -> str:
        return "\n".join(f"  {k:<18} {v}" for k, v in self.as_dict().items())


PROTOCOL = TrainingProtocol()


# ─────────────────────── единое пространство Optuna ─────────────────────

#: границы поиска, одинаковые для всех шести моделей
SEARCH_SPACE: Dict[str, Any] = {
    "n_layers":       ("int", 3, 6),
    "n_units":        ("int", 32, 256),
    "activation":     ("cat", ("tanh", "selu", "softplus")),
    "learning_rate":  ("float_log", 5e-5, 2e-3),
    "batch_size":     ("cat", (16, 32, 64, 128)),
    "max_epochs":     ("int", 200, 600),
    "grad_clip":      ("float", 0.5, 5.0),
    "weight_decay":   ("float_log", 1e-5, 1e-2),
}

#: дополнительные веса, общие для всех physics-informed моделей
PHYSICS_SPACE: Dict[str, Any] = {
    "lambda_physics": ("float_log", 1e-4, 0.5),
    "lambda_bc":      ("float_log", 1e-3, 5.0),
    "lambda_profile": ("float_log", 1e-3, 1.0),
}


def suggest(trial, name: str, spec: Sequence[Any]):
    kind = spec[0]
    if kind == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    if kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    if kind == "float_log":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    if kind == "cat":
        return trial.suggest_categorical(name, list(spec[1]))
    raise ValueError(kind)


def suggest_config(trial, model: str) -> Dict[str, Any]:
    """
    Единая выборка гиперпараметров. Все шесть моделей обязаны вызывать
    именно её — так пространство поиска гарантированно совпадает.
    """
    if model not in MODEL_FAMILIES:
        raise KeyError(f"неизвестная модель: {model}")
    cfg: Dict[str, Any] = {}
    n_layers = suggest(trial, "n_layers", SEARCH_SPACE["n_layers"])
    cfg["n_layers"] = n_layers
    for i in range(n_layers):
        cfg[f"n_units_{i}"] = suggest(trial, f"n_units_{i}", SEARCH_SPACE["n_units"])
    for k in ("activation", "learning_rate", "batch_size", "max_epochs",
              "grad_clip", "weight_decay"):
        cfg[k] = suggest(trial, k, SEARCH_SPACE[k])
    cfg["optimizer"] = PROTOCOL.optimizer          # НЕ ищется — фиксирован
    if model in PHYSICS_MODELS:
        for k, spec in PHYSICS_SPACE.items():
            cfg[k] = suggest(trial, k, spec)
    return cfg


# ────────────────────────── проверка соответствия ───────────────────────

FORBIDDEN_KEYS = {"optimizer": ("LBFGS", "SGD", "RMSprop"),
                  "lr_schedule": ("cosine", "CosineAnnealingLR", "step")}


@dataclass
class ProtocolViolation:
    model: str
    key: str
    got: Any
    expected: Any

    def __str__(self) -> str:
        return f"{self.model}: {self.key} = {self.got!r}, протокол требует {self.expected!r}"


def assert_conforms(model: str, cfg: Dict[str, Any],
                    protocol: TrainingProtocol = PROTOCOL,
                    strict: bool = True) -> List[ProtocolViolation]:
    """
    Проверить конфиг обученной модели на соответствие протоколу.
    Возвращает список нарушений; при strict=True бросает AssertionError.
    """
    v: List[ProtocolViolation] = []
    if cfg.get("optimizer", protocol.optimizer) != protocol.optimizer:
        v.append(ProtocolViolation(model, "optimizer", cfg.get("optimizer"), protocol.optimizer))
    sch = cfg.get("lr_schedule", protocol.lr_schedule)
    if sch != protocol.lr_schedule:
        v.append(ProtocolViolation(model, "lr_schedule", sch, protocol.lr_schedule))
    for key, bad in FORBIDDEN_KEYS.items():
        if str(cfg.get(key, "")) in bad:
            v.append(ProtocolViolation(model, key, cfg.get(key), f"не из {bad}"))
    for key in ("n_trials", "n_inner_splits", "split_seed"):
        if key in cfg and cfg[key] != getattr(protocol, key):
            v.append(ProtocolViolation(model, key, cfg[key], getattr(protocol, key)))
    if strict and v:
        raise AssertionError("нарушения протокола:\n  " + "\n  ".join(map(str, v)))
    return v


#: таблица «что было / что стало» — идёт в статью как доказательство
#: контролируемости сравнения (пункт 4, критерий приёмки)
PROTOCOL_DIFF: List[Dict[str, str]] = [
    {"item": "оптимизатор",
     "before": "в рукописи §2.2.4: Adam+LBFGS для DeepXDE, Adam для остальных",
     "after": "AdamW для всех шести, LBFGS исключён из пространства поиска",
     "note": "в опубликованном коде LBFGS и раньше не выбирался Optuna — "
             "рукопись описывала протокол, которого в коде нет (ERRATA E-05)"},
    {"item": "планировщик LR",
     "before": "в рукописи: cosine schedule",
     "after": "без планировщика у всех шести",
     "note": "cosine отсутствует во всём репозитории (ERRATA E-05)"},
    {"item": "пространство Optuna",
     "before": "объявлено раздельно в каждом модуле",
     "after": "protocol.suggest_config() — один источник",
     "note": "границы совпадали и раньше; теперь это гарантировано кодом"},
    {"item": "число trials",
     "before": "20 (совпадало во всех ноутбуках, но не было зафиксировано)",
     "after": "PROTOCOL.n_trials = 20",
     "note": ""},
    {"item": "повторы",
     "before": "один прогон на модель",
     "after": "5 сидов, отчёт mean ± std (пункт 3)",
     "note": ""},
    {"item": "hold-out",
     "before": "случайные 15 % того же пространства параметров",
     "after": "экстраполяционные регионы + matched-контроли (пункт 1)",
     "note": ""},
]
