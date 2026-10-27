# -*- coding: utf-8 -*-
"""Правки по итогам сквозной проверки. Числа не меняются, меняются формулировки."""
import sys, os
ROOT = '/home/user/PINNs-project'
FIXES = [
    # (файлы, старое, новое, пояснение)
    (['publication/manuscript/sec32_results.tex', 'publication/manuscript/overleaf/original.tex',
      'publication/manuscript/overleaf/sections/sec32_body.tex'],
     "the MLP satisfies the traction-free condition no better than the source (2.90 against 3.54~MPa)",
     "the MLP satisfies the traction-free condition at the level of the source (2.90 against 3.54~MPa)",
     "слова противоречили числам: 2.90 меньше 3.54"),

    (['publication/manuscript/sec4_conclusions.tex'],
     "sharing the architecture, the optimiser, the training budget, the early-stopping rule and the seeds",
     "sharing the architecture, the optimiser, the epoch limit, the early-stopping rule, the mini-batch size and the seeds",
     "термин training budget раскрыт так же, как в разделе 2.3"),

    (['publication/manuscript/sec4_conclusions.tex'],
     "predicts this component with $R^2 = 0.94$ (held-out denominator)",
     "predicts this component with $R^2 = 0.93$--$0.94$ (held-out denominator)",
     "в таблице допустимости 0.932-0.935, к 0.94 округляется только полная форма"),

    (['publication/manuscript/sec4_conclusions.tex'],
     "by 14.2~MPa per unit corrupted fraction ($p=0.0004$)",
     "by 14.2~MPa per unit corrupted fraction ($p=0.00036$)",
     "одно округление вероятности по всей статье"),

    (['publication/manuscript/sec32_results.tex', 'publication/manuscript/overleaf/original.tex',
      'publication/manuscript/overleaf/sections/sec32_body.tex'],
     "the joint corner ($-1.06$~MPa, $p=0.011$)",
     "the joint departure from the range in semi-angle and reduction ($-1.06$~MPa, $p=0.011$)",
     "жаргон corner в бегущем тексте"),

    (['publication/manuscript/sec22_fem_dataset.tex'],
     "and no temperature field exists: the analysis is not thermo-mechanically coupled.",
     ("and the temperature field was not among the exported quantities, so its magnitude is not "
      "established directly; the estimate of the adiabatic heating below places it below the "
      "resolution of the exported fields."),
     "сетка построена термосвязанными элементами CAX4RT, отсутствие поля в выгрузке не есть отсутствие связки"),

    (['publication/manuscript/overleaf/original.tex'],
     "The geometry of the die was described by three parameters: the cross-section reduction ($Q$)",
     "The geometry of the die was described by three parameters: the reduction of diameter ($Q$)",
     "Q задано через радиусы и в разделе 2.2 показано, что это обжатие по диаметру"),
]

def main() -> int:
    write = "--write" in sys.argv
    total = 0
    for files, old, new, why in FIXES:
        hits = []
        for rel in files:
            p = os.path.join(ROOT, rel)
            if not os.path.exists(p):
                continue
            s = open(p, encoding="utf-8").read()
            if old not in s:
                continue
            hits.append(rel)
            if write:
                open(p, "w", encoding="utf-8").write(s.replace(old, new))
        total += len(hits)
        print(f"{'применено' if write else 'найдено':9s} в {len(hits)}: {why}")
        for h in hits:
            print(f"            {h}")
    print(f"\nвсего замен: {total}" + ("" if write else " — запустить с --write"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
