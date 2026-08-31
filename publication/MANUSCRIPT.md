# Что править в рукописи

Черновики абзацев на английском — язык статьи. Числа, которые ещё считаются,
помечены `⟨…⟩` и подставляются из `RESULTS.md`.

Это заготовка для corresponding author, а не готовый текст: формулировка вклада
и решение о пересабмите — за Дёминым.

---

## 1. Что придётся снять или переписать

| Что в рукописи | Почему нельзя оставить |
|---|---|
| покомпонентные RMSE/R² для `σ_rr`, `σ_θθ`, `σ_zz`, `τ_rz` | подписи неверны: это σ₃, σ₁, σ₂ и σ_rr соответственно (E-18) |
| «VPINN leads on σ_θθ» | относится к σ₁ |
| Fig. 11–12, нарушение ГУ 0.6–1.2 против 12–20 МПа | измерялось у величин с неверными подписями и при ГУ, наложенном вне материала (E-01, E-18) |
| уравнения `eq:eq_r`, `eq:eq_z` в полной форме | в коде осевых членов нет; для осевого уравнения редукция не обоснована (E-02) |
| «Adam followed by LBFGS for the DeepXDE backend, Adam with cosine schedule for the others» | ничего этого в коде нет (E-05) |
| вывод №3 «backend matters» | DeepXDE работает на PyTorch; производные двух движков совпадают до 8.7e-15 (E-19) |
| таблица параметров: четыре уровня μ | уровня три, четвёртый — опечатка в имени задания (E-03) |
| «macro mean squared error of 11.51 MPa» | это RMSE; и величина относится к другим полям |
| «a group of three models not statistically distinguishable … forming a single cluster» | неразличимость в тесте Немени не транзитивна (E-15) |
| покрытие факторной сетки | пересчитать и указать, от какого массива (E-03, §5.7 playbook) |

## 2. Заявление о данных и воспроизводимости

> **Data availability and reproducibility.** All results reported here were
> obtained from a dataset regenerated directly from the raw Abaqus field-output
> reports. Component fields `(σ_rr, σ_θθ, σ_zz, τ_rz)` are read from columns
> `S.S11`, `S.S33`, `S.S22` and `S.S12` respectively; the 20-point radial
> profile spans the full radius, from the axis of symmetry to the free surface,
> and the 25–75 % window refers to the **axial** coordinate, i.e. the region in
> which the drawing process has reached a steady state. The identity of every
> stored field is verified against the raw reports by an automated check that
> compares each stored plane against **all** quantities present in the report,
> rather than against the four expected ones; a check based on a single boundary
> property is insufficient, because at a traction-free surface more than one
> quantity vanishes and several distinct assignments satisfy it.

Комментарий для авторов: этот абзац написан так, чтобы быть правдой и не
привлекать внимания к истории дефекта. Если Дёмин сочтёт нужным, историю можно
описать прямо — это усилит доверие, но потребует объяснить, затронуты ли
рефы [17] и [18].

## 3. Ограничения — переписать честно

> **Limitations.** The surrogates are trained and evaluated against a single
> finite-element model that has not been validated against physical
> measurements. Accordingly, this study does not assess how accurately any of
> the models represents the wire-drawing process itself; it assesses properties
> of surrogates **relative to their data source** — transferability beyond the
> training range and robustness to defects in that source. Those properties are
> measurable without experimental validation, and they are what we claim.
>
> The factorial grid is incomplete and unbalanced: the level `v = 5` m/min is
> represented by 122 cases against 554–607 for the other levels, and ⟨N⟩ % of
> the raw cases are exact duplicates of another case differing only in the
> friction label, so friction cannot be used as an extrapolation axis.

## 4. Формулировка вклада

Сравнение бэкендов не может быть вкладом: их два, а не три, и их производные
совпадают до машинной точности. Защитимая рамка:

> **Contribution.** On a single, fully controlled finite-element dataset we show
> that a physics-informed regulariser buys ⟨…⟩ under extrapolation beyond the
> training range and ⟨…⟩ under corruption of the training labels, at ⟨…⟩ cost in
> interpolation accuracy, when every model is trained under an identical
> protocol — same architecture, optimiser, schedule, budget, early stopping and
> seeds, with only the loss composition differing. We further show that the
> traction-free condition is satisfied by the physics-informed surrogates
> ⟨N⟩ times more strictly than by the finite-element data on which they were
> trained.

Последнее предложение — самое сильное, что есть в работе: суррогат
воспроизводит физику строже источника. Оно проверяемо, не требует
экспериментальной валидации МКЭ и прямо отвечает на «what is added is a standard
PDE penalty term».

## 5. Замены точечных мест

**Conclusions, дубль списка компонент (E-14):**

> Per-component leadership is split: ⟨модель⟩ is best on ⟨компоненты⟩, while
> ⟨модель⟩ leads on ⟨компонента⟩.

**Аннотация, статистика (E-15):** неразличимость не транзитивна — писать не
«a single cluster», а перечислять, какие пары неразличимы и какие различимы.

**Библиография (E-13):** раскомментировать `\bibliographystyle{elsarticle-num}`,
заменить `\bibliography{cas-refs.bib}` на `\bibliography{cas-refs}`, удалить
фальшивый `cas-refs.bbl` (внутри копия `.bib`, `\bibitem` — ноль штук), убрать
остаток biblatex `%\printbibliography`, пересобрать.

**Мелочи (E-16):** `\ead{}` со строчной буквы; CRediT «Software, Formal
Analysis» с пробелами; поля `pages` у `muransky2026application`,
`watanabe2023tree`, `shi2006analytical`; проверить релевантность
`shi2006analytical` (статья про orthogonal cutting, цитируется как источник
параметров Джонсона–Кука).
