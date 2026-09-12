# Замечания Дмитрия к скомпилированному PDF (08.09.2026)

## К1 (стр. 3)
**Комментарий:** Предложение до следующего подраздела можно так явно не указывать. В самих файлах эти поля есть.

**К тексту:** Not exported are PEEQ, temperature, displacements, the die reaction force, contact pressure and shear (CPRESS, CSHEAR), and inter- mediate frames. Hence the drawing force cannot be recovered from the dataset (in the retained runs the exported section is unloaded: median | ∫ on 𝜎𝑧 𝑧𝑧 d𝐴|/𝐴= 5.3 MPa against an rms 𝜎𝑧 𝑧𝑧 of 167 MPa), the friction term cannot be checked through contact stresses, and p

## К2 (стр. 3)
**Комментарий:** Мысль правильная, но стоит переформулировать само предложение

**К тексту:** The choice was checked by measuring, per component, the share of the in-window variance that lies along 𝑧and the loss from collapsing the field along 𝑧(Table 2): 10–12 % and 2–4 % for the normal components, but 40 % and 79 % for 𝜏𝑟 𝑟𝑧 , which changes sign along 𝑧and cancels under averaging. The shear component is thus intrinsically unpredictable from a one-dimensional representation, a property of

## К3 (стр. 3)
**Комментарий:** Про начальный радиус, как и длину заготовки внесу данные в раздел 2.1 . Тут дублировать не нужно

**К тексту:** 𝑅= 18 mm,

## К4 (стр. 4)
**Комментарий:** Поменять обозначения на рисунке: убрать тайтл текст под шкалами перенести в подрисуночную подпись перевести на англ.

**К тексту:** Fig. 2 Residual stress components from one finite-element run over the wire cross-section; the dashed lines mark

## К5 (стр. 4)
**Комментарий:** Не уверен, что это устоявшееся выражение для описания данной ситуации. Скорее стоит пояснить немного более длинно - в виду численных эффектов связанных с применением МКЭ при больших деформациях имеет место выход отдельных узлов за ось симметрии

**К тексту:** Fig. 2 Residual stress components from one finite-element run over the wire cross-section; the dashed lines mark

## К6 (стр. 4)
**Комментарий:** Нужно переписать само предложение: При дальнейшей подготовке данных проводился отсев части результатов, который был необходим по нескольким причинам.

**К тексту:** Nodal noise yields

## К7 (стр. 4)
**Комментарий:** Надо подумать, оставлять ли эту информацию.

**К тексту:** One run with a misspelled friction token is dropped. Cases whose profile is bit-identical to another case differing only in the friction label are dropped, keeping one per group (323 of the 2443 raw cases, 13.2 %; 319 of them survive the two preceding filters and are counted in Table 3).

## К8 (стр. 4)
**Комментарий:** Про 160 точек писать не стоит, но, да, сказать про то, что при выполнении расчётов согласно таблице 1, получаются уникальные/несовпадающие в рассматриваемом окне конфигурации точек

**К тексту:** All partitions are made at the level of cases, so the 20 radial points (or 160 grid points) of a case never straddle training and test.

## К9 (стр. 4)
**Комментарий:** Нужно добавить какое то вводное предложение, типа: Для последующего процесса обучения было рассмотрено несколько сценариев разбиения начального набора данных

**К тексту:** All partitions are made at the level of cases, so the 20 radial points (or 160 grid points) of a case never straddle training and test.

## К10 (стр. 4)
**Комментарий:** Скорее стоит назвать это разбиением/разделением по параметру \alpha или по углу наглона деформирующей зону. Называть это параметр просто углом - жаргон, который не стоит использовать в такого рода текстах

**К тексту:** corner split

## К11 (стр. 4)
**Комментарий:** В целом, мысль в самом абзаце правильная, но стоит сделать еще вариант самого текста с более академичными оборотами

**К тексту:** corner split

## К12 (стр. 4)
**Комментарий:** Это устоявшийся термин? Думается, что его лучше раскрыть более подробно

**К тексту:** training budget;

## К13 (стр. 4)
**Комментарий:** Опять таки, стоит добавить вводные конструкции: При подготовке самих моделей машинного обучения с целью проведения дальнейшего сравнительного анализа использовалась (...)

**К тексту:** training budget;

## К14 (стр. 4)
**Комментарий:** Архитектура и наборы остальных гиперпараметров определялись за счет прогонов Optuna. Поэтому, это утверждение стоит перенести в более позднюю часть текста

**К тексту:** (привязка к абзацу, без выделения)

## К15 (стр. 5)
**Комментарий:** Стоит несколько поменять предложение: Для формирования PINN были использованы уравнения равновесия, которые представлены ниже. В них входят параметры зависящие от координат r,z. Однако, при подготовке моделей согласно работам [17,18] и использовании соотношений (1)-(2), они претерпевают ряд изменений

**К тексту:** (привязка к абзацу, без выделения)

## К16 (стр. 7)
**Комментарий:** Стоит добавить вводную конструкцию. Как упоминалось ранее, для корректного сравнения подготовленных моделей необходимо на этапе обучения подготовить равные условия, для чего был спроектирован следующий протокол.

**К тексту:** he previous version trained each model once and applied Friedman–Nemenyi across the 304 hold-out cases;

## К17 (стр. 7)
**Комментарий:** Поскольку это другой журнал и издательство, то вероятность того, что рецензенты были знакомы с предыдущей версией текста крайне мала. Для стороннего читателя - еще меньше. Поэтому. Можно сказать , что также проводился отдельны прогон тестов, который показывает следующее...

**К тексту:** he previous version trained each model once and applied Friedman–Nemenyi across the 304 hold-out cases;

## К18 (стр. 7)
**Комментарий:** Писать "вместо эксперимента" - совсем не нужно. На таких вещах могут развернуть саму публикацию. В целом этот раздел еще будет подвергаться корректировке с учетом выгрузки усилий и не только. Но в целом, тут текст должен начинаться несколько иначе: Для верификации получаемых результатов имитационного моделирования было проведено сравнение с известными аналитическими соотношениями (Зибель , Авитзур) , которые показывают ...

**К тексту:** In place of experimental validation, five internal consistency tests were run on the 1756 retained cases. The exported cross-section is self-equilibrated to 3.2 %: the median | ∫ qu 𝜎𝑧 𝑧𝑧 d𝐴|/𝐴is 5.3 MPa against an rms 𝜎𝑧 𝑧𝑧 of 167 MPa. Local equilibrium closes to 2.1 % of the sum of the moduli of its terms. At the free surface |𝜎𝑟 𝑟𝑟 | is 4.0 MPa against 54.0 MPa in the bulk, i.e. 7.4 % (a nodal 

## К19 (стр. 8)
**Комментарий:** Это правда, но думается, что стоит написать еще более аккуратно

**К тексту:** The finite-element model is the reference throughout; no physical mea- surement enters, so the section characterises the surrogates relative to their data source, not to the process (see the closing paragraph of this section).

## К20 (стр. 8)
**Комментарий:** Надо подумать над интерпретацией и самой формулировкой

**К тексту:** The highest speed is where the surrogate breaks down. With 𝑣= 250 m/min withheld the error is 48–56 MPa, still below the 2 71 MPa of the constant predictor, but 𝑅n 𝑅n 2 (global denominator) falls to 0.23–0.45, the shear component is predicted worse than a con- stant (NRMSE 1.6–1.9, held-out standard deviation as denomina- tor) and the MLP becomes unstable across seeds (±19 MPa). This axis carries 

## К21 (стр. 8)
**Комментарий:** Возможно стоит вставить ссылку

**К тексту:** Holm correction

## К22 (стр. 9)
**Комментарий:** Стоит написать более аккуратно

**К тексту:** The strong form is estab- lished better than the weak form on extrap:k_max (−0.70 MPa, 𝑝= 0.018) and, by a margin smaller than the seed spread, on matched:v_max (−0.11 MPa, 𝑝= 0.046); on the remaining eleven splits the two physics-informed forms are indistinguishable. A Friedman test over seeds [40,41] rejects equality on three of the four splits where the paired test does (extrap:alpha_max, 𝑝= 0.

## К23 (стр. 9)
**Комментарий:** Комментарий аналогичный тому, что был ранее про предыдущие версии текста

**К тексту:** The previous version attributedf

## К24 (стр. 9)
**Комментарий:** Комментарий аналогичный тому, что был ранее про предыдущие версии текста

**К тексту:** Figs. 11–12

## К25 (стр. 10)
**Комментарий:** Нужен вводный абзац прежде чем переходить к выводам: В представленной работе была проведена серия расчетов по волочению проволок , подготовлены разлинчые модели машинного обучения, а также несколько различных вариантов разделения предобработанных выборок для дальнейшей оценки. По результатам проведенного исследования можно сделать следующие выводы (этот текст/пример абзаца можно расширить и уточнить)

**К тексту:** The surrogates are trained and evaluated against a single finite-element model that has not been validated against physical measurements (Section 2.2). This study therefore does not assess how accurately any of the models represents the wire- drawing process; it assesses properties of the surrogates relative to their data source, transferability beyond the training range and robustness to defects 

## К26 (стр. 10)
**Комментарий:** В целом, да. Но, нужно подумать над тем, все ли перечисленное в итоге оставить.

**К тексту:** The surrogates are trained and evaluated against a single finite-element model that has not been validated against physical measurements (Section 2.2). This study therefore does not assess how accurately any of the models represents the wire- drawing process; it assesses properties of the surrogates relative to their data source, transferability beyond the training range and robustness to defects 

## К27 (стр. 11)
**Комментарий:** Не думаю, что стоит отдельно дублировать в такой формулировке этот вывод. Лучше будет провести сравнение усилий волочения и уже относительно этого результата говорить что либо про адекватность моделей

**К тексту:** The finite-element model was not validated against physical measurements. These conclusions therefore concern the sur-i rogates relative to their data source—accuracy on the finite-i element field, transferability beyond the training range, and robustness to defects of that source—and not the accuracy of the drawing simulation itself.

## К28 (стр. 11)
**Комментарий:** Комментарий аналогичный тому, что был ранее про предыдущие версии текста

**К тексту:** The finite-element model was not validated against physical measurements. These conclusions therefore concern the sur-i rogates relative to their data source—accuracy on the finite-i element field, transferability beyond the training range, and robustness to defects of that source—and not the accuracy of the drawing simulation itself.

## К29 (стр. 11)
**Комментарий:** Про кривые обучения упомянуто несколько раз, но нигде нет самих рисунков или материалов, на которые идет отсылка

**К тексту:** :corner