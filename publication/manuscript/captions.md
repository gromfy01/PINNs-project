# Подписи к рисункам для новой версии статьи

Рисунки в режиме статьи лежат в `publication/figures/paper/`. По указанию
руководителя: подписи переведены, числа на осях чёрные, `R_0 → R`, общий
заголовок перенесён в подрисуночную подпись, пояснения из-под рисунка — в текст
статьи (см. §«Что переезжает в текст» внизу).

Нумерация — по расстановке в Overleaf (Рис. 2, 5, 6). Английский вариант —
основной; русский — как сформулировал руководитель, для сверки смысла.

Note: the area reductions in the captions are the measured values of the factor table (Section 2.2, tab:factors: Q = 0.10 → 20.1 %, Q = 0.025 → 5.4 %), not the identity 1 − (1 − Q)², for consistency with that table. <!-- src: sec22 tab:factors; ERRATA.md E-22 (0.2013, 0.0543) -->

---

## Fig. 2 — `paper/fig2_fem_fields.png`

<!-- REV К4: рисунок берётся из figures/paper/fig2_fem_fields.png — без общего
заголовка, подписи панелей и осей по-английски; текст, стоявший под цветовыми
шкалами (обозначение компоненты и единица измерения), перенесён в подпись. -->

**EN**

> **Fig. 2.** Distribution of residual stresses obtained from the FEM
> simulation at Q = 0.10 (10 % reduction in diameter, 20.1 % in area),
> k = 0.5, α = 12°, μ = 0.05, v = 10 m/min. Panels from left to right:
> the components σ_rr, σ_θθ, σ_zz and τ_rz over the full axial extent of the
> wire; the colour scale of each panel is graduated in MPa and is common to
> that panel only. r is normalised by the billet radius R = 18 mm, z by the
> model length L = 139.5 mm. Dashed lines mark the axial window 25–75 % of L
> from which the training data are taken.

**RU (по руководителю)**

> Рис. 2. Распределение остаточных напряжений, полученных в результате
> расчётов МКЭ, при параметрах Q = 10 % (по диаметру; 20,1 % по площади),
> k = 0,5, α = 12°, μ = 0,05, v = 10 м/мин. Компоненты σ_rr, σ_θθ, σ_zz и
> τ_rz по всей длине проволоки; r отнесено к радиусу заготовки R = 18 мм,
> z — к длине модели L = 139,5 мм. Пунктиром показано осевое окно 25–75 % L,
> из которого берутся обучающие данные.

Замечание для текста: на этом рисунке R — радиус **заготовки** (18 мм),
поэтому протянутая часть доходит до r/R = 0.9. На рис. 5 и 6 R — радиус
**поверхности в том же сечении**, и r/R = 1 всегда свободная поверхность.
Это надо сказать в тексте один раз, там, где вводится нормировка.

---

## Fig. 5 — `paper/fig5_compare_fields.png`

**EN**

> **Fig. 5.** Distribution of the residual stress tensor components over the
> investigated region for FEM, MLP, and PINN in the full and reduced
> formulations at Q = 0.025 (2.5 % in diameter, 5.4 % in area), k = 0.1,
> α = 8°, μ = 0.10, v = 20 m/min. Rows: FEM reference and the three model
> families; columns: σ_rr, σ_θθ, σ_zz, τ_rz. The colour scale is common to
> all rows within a component. r is normalised by the surface radius in the
> same cross-section (r/R = 1 is the free surface), z by the window length.
> The case shown has the median test error of the full PINN.

**RU (по руководителю)**

> Рис. 5. Результаты распределения компонент тензора остаточных напряжений
> по исследуемой области для МКЭ, MLP и PINN в полной и упрощённой
> постановке при параметрах процесса Q = 2,5 % (по диаметру; 5,4 % по
> площади), k = 0,1, α = 8°, μ = 0,10, v = 20 м/мин. Строки — МКЭ и три
> семейства моделей; столбцы — σ_rr, σ_θθ, σ_zz, τ_rz. Шкала общая для всех
> строк внутри компоненты. r отнесено к радиусу поверхности в том же сечении
> (r/R = 1 — свободная поверхность), z — к длине окна. Показан набор с
> медианной ошибкой полной PINN.

---

## Fig. 6 — `paper/fig6_residual_maps.png`

**EN**

> **Fig. 6.** Comparison of the equilibrium residual distributions R_r and
> R_z over the investigated region for FEM, MLP, and PINN in the full and
> reduced formulations at Q = 0.025, k = 0.1, α = 8°, μ = 0.10,
> v = 20 m/min (same case as Fig. 5). (a) radial residual R_r, (b) axial
> residual R_z, both made dimensionless by the characteristic magnitude of
> the terms entering the respective equation; the panel titles give the
> median |R| over all 261 test cases. (c) distribution over the test cases of
> the traction-free violation |σ_rr| at r/R = 1, mean marked. The FEM column
> is the reference: the source itself does not satisfy the equations exactly.

**RU (по руководителю)**

> Рис. 6. Сравнение распределения невязок R_z и R_r по исследуемой области
> для МКЭ, MLP и PINN в полной и упрощённой постановке при параметрах
> процесса Q = 2,5 %, k = 0,1, α = 8°, μ = 0,10, v = 20 м/мин (тот же набор,
> что на рис. 5). (a) радиальная невязка R_r, (b) осевая R_z, обе
> обезразмерены на характерную величину членов соответствующего уравнения;
> в заголовках панелей — медиана |R| по всем 261 тестовым наборам.
> (c) распределение по тестовым наборам нарушения traction-free |σ_rr| при
> r/R = 1, отмечено среднее. Столбец МКЭ — базовая линия: источник сам не
> удовлетворяет уравнениям точно.

---

## Что переезжает в текст (снято с рисунков)

Это пояснения, которые раньше стояли под рисунками. По указанию руководителя
они должны быть в тексте §3.2, а не в подписях. Английские формулировки — для
вставки.

**Нормировка координат** (один раз, при первом рисунке с полями):

> Coordinates are dimensionless throughout. In Fig. 2 the radius is
> normalised by the billet radius R = 18 mm, so the drawn portion of the wire
> ends at r/R = 0.9; in Figs. 5–6 it is normalised by the surface radius of
> the same cross-section, so that r/R = 1 coincides with the free surface at
> every axial position. The axial coordinate is normalised by the model
> length (Fig. 2) or by the window length (Figs. 5–6). The radial axis is
> stretched relative to the axial one: at L/R ≈ 3.5–7.7 an equal-aspect
> panel degenerates into a strip.

**Колорбары и сравнимость** (при рис. 5):

> Within each component the colour scale is shared by the FEM reference and
> all model families, so that panels in one column are directly comparable.
> The displayed case is the one with the median test error of the full PINN
> rather than the best one, to show typical rather than favourable behaviour.

**Обезразмеривание невязок** (при рис. 6):

> The equilibrium residuals R_r and R_z are divided by the characteristic
> magnitude of the terms of the respective equation — the same scaling used
> in the loss function: scale_r = (sd[σ_rr] + sd[σ_θθ])/R and
> scale_z = sd[σ_zz]/L + sd[τ_rz]/R, computed once from the FEM data and
> applied identically to all families. A value of 0.1 therefore means that
> the equation fails to close by 10 % of the characteristic magnitude of its
> own terms. Medians are taken over interior grid nodes, where the central
> difference is second-order; at the boundary it degenerates to one-sided.

**Базовая линия МКЭ** (при рис. 6):

> The FEM solution is included as the reference for every physics-based
> metric. Because of nodal extrapolation and averaging, the source itself
> does not satisfy the equilibrium equations or the traction-free condition
> exactly (median |R_r| = 0.041, |σ_rr(r=1)| = 3.54 MPa); a surrogate that
> violates them by less than the source is, in this respect, more physically
> admissible than the data it was trained on.

**Что различает семейства** (при рис. 5–6, один раз):

> The three families share the architecture, the split, the seed, the number
> of epochs and the optimiser; they differ only in the composition of the
> loss function.
