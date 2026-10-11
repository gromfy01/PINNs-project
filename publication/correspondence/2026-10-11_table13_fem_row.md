# Таблица 13: строка «источник МКЭ» для двумерного трека (11.10.2026)

Дмитрий, таблица 13 (точность и допустимость двумерных моделей) пересчитана, как вы просили, на наборе
1756 случаев с тремя сидами — по 18 прогонам стадии twod (MLP и PINN в сокращённой и полной форме,
сплиты random 1493/263 и extrap:alpha_max 1377/379). Переобучение для этого не понадобилось.

Не хватает одной строки — тех же величин для самих данных МКЭ на тех же тестовых случаях (263 и 379):
медианных невязок равновесия |R_r|, |R_z| (МПа/м) и нарушения граничного условия |σ_rr(r=1)|, |τ_rz(r=1)|.
Без неё сравнение «модели строже самого МКЭ» в 2D остаётся по одному сиду (прежний прогон на 1498/261).

**Вопрос.** Можете прислать двумерный набор, на котором шла стадия twod (файл Field2D в формате .npz, тот,
что передавался в `run_matrix.py --stage twod --dataset …`), или архив сырых отчётов (2443 .rpt, релиз
raw-data-v1)? По нему строка считается за секунды, без обучения.

Если удобнее посчитать у себя — из папки `publication/`:

```python
import sys, numpy as np
sys.path[:0] = ['code']
from dataset2d import Field2D
from splits import build_split_suite
from trainer2d import equilibrium_audit_2d
f = Field2D.load('ПУТЬ_К_NPZ')
suite = build_split_suite(f.proc)
for name in ('random', 'extrap:alpha_max'):
    te = suite[name].test_sets
    eq = equilibrium_audit_2d(f.y[te], f.r_phys[te], f.z_phys[te])
    print(name, len(te), round(eq['eq_r_full']), round(eq['eq_z_full']),
          round(float(np.abs(f.y[te][:, :, -1, 0]).mean()), 2),
          round(float(np.abs(f.y[te][:, :, -1, 3]).mean()), 3))
```

Должно получиться 263 и 379 случаев; четыре числа в каждой строке — |R_r|, |R_z|, |σ_rr(r=1)|, |τ_rz(r=1)|.
