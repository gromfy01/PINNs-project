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
> represented by 122 cases against 554–607 for the other levels, and 13.2 % of
> the raw cases are bit-identical duplicates of another case differing only in
> the friction label, so friction cannot be used as an extrapolation axis.
>
> The shear component `τ_rz` is essentially not predicted: its NRMSE is 0.58 on
> a random hold-out and 2.2–2.4 beyond the training range, against a physical
> scale of 0.67 MPa mean absolute value — two orders of magnitude below the
> normal components. Reported macro-averaged figures are therefore dominated by
> the normal components, and per-component figures must be given alongside them.

## 4. Формулировка вклада

Сравнение бэкендов не может быть вкладом: их два, а не три, и их производные
совпадают до машинной точности. Защитимая рамка:

> **Contribution.** Under a protocol in which every model shares architecture,
> optimiser, budget, early stopping and seeds — only the loss composition
> differing, and the weight of the physics term calibrated so that its gradient
> norm is a fixed fraction of the data term's — we report three findings.
>
> First, the physics-informed surrogates satisfy the traction-free condition
> **5.4 times more strictly than the finite-element data on which they were
> trained** (0.64 MPa against 3.43 MPa for the data and 4.50 MPa for the
> data-driven baseline), and halve the equilibrium residual (920 against
> 1936 MPa/m), at no measurable cost in accuracy (11.87 against 11.43 MPa
> macro-RMSE, p = 0.052). The advantage persists on a held-out region outside
> the training range (1.76 against 5.89 MPa, with 3.31 MPa for the data).
>
> Second, under systematic corruption of 10 % of the training labels the
> physics-informed surrogate degrades to 17.45 MPa against 19.47 MPa for the
> data-driven baseline (p = 0.001), while at zero corruption the two are
> indistinguishable.
>
> Third, transferability beyond the training range is **strongly
> axis-dependent**: holding out an entire factor level costs nothing along the
> land-length coefficient (×0.94), a factor of two along the die semi-angle
> (×1.94), a factor of 2.5 along the area reduction (×2.49), and a factor of
> 5.5 along drawing speed (×5.51), where the surrogate ceases to beat a
> constant predictor. The spread between axes exceeds the spread between model
> families, so a single "extrapolation error" figure is not informative.

Первое утверждение — самое сильное, что есть в работе: суррогат воспроизводит
физику строже источника. Оно проверяемо, не требует экспериментальной
валидации МКЭ и прямо отвечает на «what is added is a standard PDE penalty
term». Третье — прямой и полезный ответ на «interpolation within this
particular case».

### 4.1 Что заявлять НЕЛЬЗЯ

Проверено и не подтвердилось:

* **«Physics-informed выигрывает при дефиците данных».** При 10 % обучающего
  пула отрыв +1.21 МПа при p = 0.443 — не установлен; на случайном hold-out
  преимущества нет ни на одной доле.
* **«Выбор бэкенда важен».** Производные torch и JAX совпадают до 8.7e-15, а
  DeepXDE в работе — это PyTorch. Две независимые реализации одного метода
  расходятся на 0.6–3.4 МПа, и знак расхождения меняется от сплита к сплиту:
  ровно такой разрыв и был заявлен в отклонённой версии как эффект движка.
* **«Слабая форма лучше/хуже сильной».** При равной норме градиента физического
  члена сильная форма давит невязку равновесия вдвое сильнее (920 против 1605),
  но по точности семейства неразличимы; на разных сплитах лидируют разные.

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

---

## 6. Рисунки: что заменить

### 6.1 Схема архитектуры — заменить полностью

В отклонённой версии вход сети показан как `[Q, k, α, μ, v, r]`, без осевой
координаты. При этом в тексте (`eq:eq_r`, `eq:eq_z`) напечатана **полная**
форма уравнений равновесия с `∂τ_rz/∂z` и `∂σ_zz/∂z`. Reviewer #1 указал ровно
на это противоречие: без z во входе осевых производных не существует, и осевое
уравнение не может быть записано ни в каком виде.

**Замена:** `publication/figures/fig_architecture.png`. Показывает оба трека
рядом, с явной пометкой добавленной координаты, и то, какие члены становятся
вычислимыми. Три блока:

* `(a)` вход и архитектура обоих треков — 6 признаков против 7;
* `(b)` что даёт z: для основного трека `R_z` записать нельзя, для
  расширенного оба уравнения выписаны целиком, рядом — измеренная доля
  отброшенного (радиальное 1.3 %, осевое 68 %);
* `(c)` таблица состава функции потерь по трём семействам — они отличаются
  ТОЛЬКО им, архитектура и протокол одинаковы.

В подписи к рисунку в статье надо прямо сказать, что основной трек
соответствует постановке из предыдущей версии, а расширенный вводится в ответ
на замечание.

### 6.2 Остальные рисунки

| было | стало | почему |
|---|---|---|
| Fig. 11–12 (нарушение ГУ) | `fig_residual_maps.png` | прежние измеряли величины с неверными подписями (E-18) и при ГУ вне материала (E-01); новый даёт базовую линию самого МКЭ и обезразмеренную невязку |
| контуры по одному режиму | `fig_matrix_alpha_Q.png` | одиночная карта не показывает, что делает параметр |
| — | `fig_fields*.png`, `fig_errors*.png` | поля и невязки разнесены: у них разные шкалы и разные вопросы |
| — | `fig_compare_fields.png`, `fig_compare_errors.png` | сравнение трёх семейств на одном наборе |

Общее для всех: безразмерные координаты `r/R`, `z/L`; расходящаяся шкала с
нейтральной серединой (напряжения знакопеременные, радужная шкала создаёт
ложные границы); базовая линия самого МКЭ там, где сравнивается физика.
