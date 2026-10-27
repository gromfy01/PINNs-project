# Вычитка отправленной рукописи (EwC, 14.10.2026)

Проверена версия из архива `EwC_submission_2026-10-13.zip` (46 страниц, номера страниц ниже по этому PDF). Если Дмитрий правил текст перед отправкой, часть пунктов может быть уже исправлена.

Проверка: шесть независимых проходов (язык, согласованность чисел, числа против журнала прогонов, перекрёстные ссылки и обозначения, литература, термины и технические утверждения). Каждое замечание отдельно перепроверено с попыткой опровергнуть. После объединения дублей осталось 12 замечаний по существу, 20 опечаток и терминов и 17 мелочей оформления. Ключевые числа (табл. 2, счёт случаев при v = 5 м/мин, множители g) пересчитаны по данным заново.

## A. Неточности по существу

**A1. Введение, предпоследний абзац, с. 5. Описание теста Фридмана противоречит методике.**
Сейчас: «a Friedman test being used only as an auxiliary check across splits».
Раздел 2.3 «Statistical treatment» и раздел 3.2.1 применяют тест внутри каждого сплита, блоки — сиды, сплиты не объединяются.
Исправить: «a Friedman test over seeds within each split being used only as an auxiliary check».

**A2. Табл. 2 (с. 8) и доля потери τrz «79 %» (с. 8, с. 30, вывод 5). Числа посчитаны на наборе 1848, а не на итоговых 1756.**
Средние |σ| в табл. 2 (47.9 / 56.6 / 140.2 / 3.22 и 46.6 / 54.4 / 137.6 / 0.68) совпадают до цифры с набором 1848.
На наборе 1756 получается 48.0 / 56.2 / 141.7 / 1.92 для поля и 47.4 / 55.2 / 141.4 / 0.53 для профиля, то есть τrz теряет при свёртке 72 %, а не 79 %. Нормальные компоненты теряют 0.2–1.9 %.
Популяция столбца «Share of the variance along z» не установлена.
Вывод (τrz в одномерной постановке непредсказуем) не меняется.
Исправить: при доработке пересчитать табл. 2 на 1756 и заменить 79 % на 72 % во всех трёх местах. Минимально: указать в подписи, что таблица посчитана на 1848 случаях до снятия 92 расчётов.
Связанное (с. 9 и с. 34): «a mean |σ| of about 79 MPa for the normal components against 0.67 MPa for τrz» — это тоже значения набора 1848. На 1756: 81 МПа и 0.54 МПа.

**A3. Раздел 2.3 «Network and inputs», с. 12. Утверждение про Optuna не подтверждается.**
Сейчас: «This configuration was selected with Optuna [2] at the initial design stage of the study and then held fixed.»
В ноутбуках Optuna (май 2026, 20 испытаний, прежняя версия набора) оптимумы — 4–6 слоёв по 76–253 нейрона, lr 1.0·10⁻³–1.9·10⁻³. Пространство поиска по lr ограничено 2·10⁻³. Конфигурация статьи 3 × 96, lr 3·10⁻³ среди оптимумов отсутствует и лежит вне пространства поиска.
Нужно подтверждение Дмитрия. Предлагаемая формулировка: «This configuration was fixed a priori, before any held-out region was evaluated, and was not tuned afterwards. An earlier Optuna search [2] on a previous version of the dataset favoured deeper networks with a smaller learning rate.»

**A4. Счёт случаев при v = 5 м/мин: с. 15 («(122 cases)») и с. 34 («represented by 122 cases against 554–607 for the other levels»).**
Это счёт по 2443 сырым отчётам. В анализируемом наборе 1756: v = 5 м/мин — 94 случая, остальные уровни — 250 / 436 / 475 / 501. В обучающих пулах сплитов random и extrap:alpha_max — 83 и 87.
Исправить на с. 15: «(94 of the 1756 cases)». На с. 34: «in the 1756 cases the level v = 5 m/min is represented by 94 cases against 250–501 for the other levels».

**A5. Раздел 3.2.1, абзац после табл. 9, с. 26.**
Сейчас: «In absolute terms the advantage is small, 0.3–1.9 MPa at an error level of 8–16 MPa».
Четыре установленные разности PINN − MLP по табл. 10: 0.29, 0.58, 1.06, 1.16 МПа при ошибках 8.3–16.6 МПа. Значение 1.9 МПа взято с extrap:Q_high, где разность незначима (p = 0.062), а ошибки 23–26 МПа.
Исправить: «0.3–1.2 MPa on the four splits where it is established, at an error level of 8–17 MPa».

**A6. Раздел 2.2 «Group-aware splits», с. 10.**
Сейчас: «Each extrapolation split is accompanied by two controls.»
По табл. 4 внутренний контроль есть только у сплитов по α и Q. У k, v и совместного (α, Q) — только согласованный (matched).
Исправить: «Each extrapolation split is accompanied by a matched control, and the splits along α and Q also by an interior control.»

**A7. Раздел 2.2, с. 10 (и раздел 3.2.1, с. 25). «Joint departure from the ranges» противоречит следующему предложению.**
Сейчас: «is defined by a joint departure from the ranges of two factors, the die semi-angle α and the reduction Q». Тут же сказано, что по каждой оси тестовые случаи остаются внутри обучающего диапазона.
Исправить: «is defined by a gap in the joint range of two factors» — так же, как в табл. 4, аннотации и выводах.
На с. 25: «the joint departure from the range in semi-angle and reduction» → «the gap in the joint semi-angle–reduction range».

**A8. Раздел 2.3 «Loss function», с. 14. «Damped roughly two-hundred-fold» не согласуется ни с приведёнными там же процентами, ни с измеренными множителями.**
Отношение 37 % / 0.36 % ≈ 100. По журналу прогонов калибровочный множитель g (уравнение 9) в среднем 4.97 для VPINN и 0.165 для PINN, то есть отношение около 30. Проценты взяты из ранних замеров на прежней версии.
Исправить: «… and is damped by more than an order of magnitude: the calibration multiplier of Eq. (9) is g ≈ 5.0 for the weak form against 0.165 for the strong form, and without the calibration "strong versus weak" would degenerate into "physics on versus off".»

**A9. Раздел 3.1, проверки внутренней согласованности, с. 22. Сравнение несопоставимых величин.**
Сейчас: «the maximum von Mises stress over the retained cases is 232.6 MPa, against the 314–331 MPa that the constitutive equation returns along the histories of the 112 dedicated runs».
314–331 МПа — напряжение течения при скорости волочения, со скоростным множителем Джонсона–Кука 1.35–1.46. Остаточное поле сравнивать нужно с квазистатическим пределом A + Bεⁿ.
Исправить: убрать сравнение или сравнить с квазистатическим пределом текучести при комнатной температуре.

**A10. Введение, с. 3. Ссылки [21] и [36] описаны неверно.**
Сейчас: «Convolutional neural networks prove effective in indentation analysis, simultaneously predicting non-equiaxial residual stresses and plastic flow properties [21, 36]».
[36] — CNN, неравнодвухосные (non-equibiaxial) напряжения. [21] — ANN, равнодвухосные (equi-biaxial). «Equiaxial» относится к форме зерна, а не к напряжённому состоянию.
Исправить: «Neural networks prove effective in indentation analysis, predicting plastic flow properties together with equi-biaxial residual stresses [21] and, with a convolutional network, non-equibiaxial residual stresses [36]».

**A11. Аннотация. Формулировка про малые выборки сильнее, чем в тексте.**
Сейчас: «and with less training data no advantage is established».
В разделе 3.2.2 два из восьми сравнений значимы (p = 0.048 и 0.033), с противоположными знаками. После поправки Холма ни одно не значимо.
Исправить при доработке: «no consistent advantage is established». Это +1 слово к 250.

**A12. Порядок первых ссылок на рисунки: 1, 4, 5, 6, 2, 3.**
Рис. 4 (поля МКЭ), рис. 5 и 6 впервые упомянуты в разделе 2.2 (с. 8), а схемы, рис. 2 и 3, — только на с. 12. Springer требует ссылаться на рисунки по порядку.
Исправить при доработке: поменять нумерацию (поля МКЭ сделать рис. 2) или убрать ранние ссылки на рис. 5–6 в разделе 2.2.

## B. Опечатки и термины

1. **«billet» в 7 местах** (нарушает правило «wire»):
   - с. 8: «normalised by the billet radius» → «by the initial radius of the wire»;
   - с. 9: «(the billet adhered to the die …)» → «(the wire adhered to the die …)»;
   - с. 9: «their tail retains the billet radius R0» → «the initial radius of the wire R0»;
   - с. 17, подпись рис. 4: «normalised by the billet radius R = 18 mm» → «by the initial radius of the wire, R = R0 = 18 mm»;
   - с. 21: «in contact with the tail of the billet» → «tail of the wire»;
   - с. 39, условные обозначения: «R = billet radius …» → «initial radius of the wire …»;
   - с. 39: «R0, Rf = radius of the billet and of the drawn wire» → «initial radius of the wire and radius of the drawn wire».
2. с. 3: «physically informed machine learning» → «physics-informed machine learning». «the full potential of physical informed neural networks (PINNs)» → «the full potential of PINNs» (сокращение уже введено выше).
3. с. 2: «Finite element analysis (FEM)» → «The finite element method (FEM)» (ниже FEA определено как finite element analysis).
4. с. 3: «additive manufactured steels» → «additively manufactured steels».
5. с. 3 и ссылка [56]: «Incepu-Net» → «IncepU-Net». В references.bib название сети взять в фигурные скобки, иначе стиль печатает «incepu-net».
6. с. 2: «Nb3Sn» → «Nb₃Sn» (Nb$_3$Sn).
7. с. 5: «S1 - is the axis of symmetry of the wire; S2 - is the outlet surface» → «S1 is the axis of symmetry of the wire, S2 is the outlet surface …».
8. Введение, с. 2–3: американское написание при британском во всей статье: behavior, optimization, penalized, generalization, regularization → behaviour, optimisation, penalised, generalisation, regularisation.
9. Заголовки разделов в разном регистре: «Preparing Machine Learning models», «ML models results» и др. Привести к одному стилю.
10. с. 26: «The VPINN is established better than the MLP» и «The strong form is established better than the weak form» → «established as better».
11. с. 26: «with the Nemenyi post-hoc separating only …» → «with the Nemenyi post hoc test separating only …».
12. Подпись табл. 10, с. 27: «0 PINN − MLP and 0 VPINN − MLP differences remain significant» → «no PINN − MLP or VPINN − MLP difference remains significant».
13. с. 18–19 и приложение A, с. 40: «In the writing of Vega et al.» → «In the notation of Vega et al.».
14. с. 15: «beside the same quantity computed on the FEM data» → «the same quantities».
15. с. 15 и с. 30: «R² is reported with a single denominator, the standard deviation of each component over the whole dataset» → «the variance» (знаменатель R² — дисперсия).
16. с. 2: «By training artificial neural networks (ANNs) on representative datasets generated through FEA, these networks can …» (висячий оборот) → «Once trained on representative datasets generated through FEA, artificial neural networks (ANNs) can …».
17. Литература [6]: «p 1–46» → «pp 1–46».
18. Литература [12] и [42]: один автор напечатан как «Del Rey» и «del Rey». Привести к одному виду.
19. Литература [13]: «International Journal of Advanced Manufacturing Technology» → «The International Journal …», как в [14] и [57].
20. Литература [14]: добавить выпуск 3–4, то есть 141(3–4):1847–1858. Литература [24]: выпуск 1–2 и полные инициалы (Li SY). Литература [46]: нет страниц.

## C. Мелочи оформления (при доработке)

- Индекс k использован трижды: коэффициент калибровки k, частоты Фурье (k = 1…4) и пробные функции (k ≤ 8). Для частот и пробных функций лучше j.
- Уравнение 12: r = 1 − (Rf/R0)² — обозначение степени деформации совпадает с радиальной координатой. Лучше ψ или r_A.
- Уравнение 8: общий символ невязки R совпадает с радиусом R. Лучше 𝓡.
- Подписи осей рис. 4–6: z/L, где L — длина модели (рис. 4) или длина окна (рис. 5–6). Пояснить в подписях или обозначить L_m и L_w.
- Условные обозначения: ε̇ описана как «mean strain rate over the pass», а в уравнении (1) это мгновенная скорость деформации.
- Подписи табл. 6 и 7 не говорят, какие формулы стоят в столбцах и что такое ΔT. Подпись табл. 9 «holding out a whole factor level» не охватывает совместную область (α, Q).
- с. 22: «|tr LE|» — пояснить, что LE — логарифмическая деформация в Abaqus.
- с. 6: усреднение напряжения течения записано как «∫ σ dεp / ∫ dεp», а словами сказано «weighted by plastic work». Уточнить, что вес — приращение пластической деформации.
- с. 6: «the analytical solutions …, which are energy methods» — Зибель и Авицур не являются чисто энергетическими методами. Лучше убрать оборот.
- с. 8: «at a window length of 3.5–7.7 radii» — 7.7 относится ко всей модели (рис. 4), а не к окну.
- Подпись табл. 10 и раздел 2.3: «α = 0.05» — α уже обозначает угол. Писать «significance level 0.05».
- Раздел 3.2: «mean ± standard deviation over five seeds» — в этапах размера выборки, искажения меток и в двумерном треке сидов три.
- Введение, с. 5: «n = 5 pairs in the one-dimensional track» — n = 3 также в этапах размера выборки и искажения меток.
- Ссылка [55] (Wang, Yu, Perdikaris 2022, NTK) для балансировки по нормам градиентов. Точнее подходит Wang, Teng, Perdikaris 2021 (SIAM J Sci Comput 43(5):A3055) — источник не однозначен.
- с. 26: «A Friedman test over seeds rejects equality on three of the four splits where the paired test does» → уточнить «where the paired PINN–MLP test does».
- Литература: 13 записей содержат «and others» в .bib. Стиль всё равно печатает «et al», но полные списки авторов надёжнее.
- Единицы: «J kg⁻¹K⁻¹» → «J kg⁻¹ K⁻¹».
