# Схемы обучения: исходники

`mlp_schematic.tex` и `pinn_schematic.tex` — исходники рисунков 2 и 3
(`pictures/mlp_schematic.pdf`, `pictures/pinn_schematic.pdf`). Собираются
отдельно, классом `standalone`:

    pdflatex mlp_schematic.tex
    pdflatex pinn_schematic.tex
    cp *.pdf ../../overleaf/pictures/

Прежние `mlp_optuna_schematic.png` и `pinn_optuna_schematic.png` удалены:
у них не было исходника, и они противоречили тексту статьи в трёх местах.

  - Показывали внешний цикл Optuna (TPE sampler + MedianPruner) и «Best trial»,
    тогда как в §2.3 сказано, что архитектура и гиперпараметры зафиксированы до
    того, как оценивалась хоть одна отложенная область, и никакого поиска не было.
    Optuna в коде этой работы не используется вовсе.
  - Подписывали оптимизатор как «Adam + cosine LR schedule». В тексте AdamW и
    расписания нет.
  - На схеме PINN стояло «Two-phase update: Phase 1 Adam, Phase 2 L-BFGS».
    L-BFGS в протоколе нет.

Новые схемы показывают то, что действительно происходит: один цикл обучения с
фиксированной конфигурацией, ранняя остановка по macro-RMSE на внутренней
подвыборке, восстановление лучшего состояния. Числа в рамках взяты из §2.3:
3 скрытых слоя по 96 tanh, признаки Фурье k ≤ 4, AdamW без расписания,
eta = 3e-3 (2e-3 в расширенном треке), weight decay 1e-4, обрезка градиента 1.0,
батч 64 (32), до 800 эпох (300), patience 40 (30), сиды 0–4 (0–2);
lambda_phys = 0.3, lambda_bc = 0.5 и калибровка g по нормам градиентов.
