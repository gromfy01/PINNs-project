"""
make_release.py — сборка публичного пакета кода по белому списку.

Копирует из репозитория в выходной каталог ровно файлы из WHITELIST и ничего
больше, раскладывая их так:

    code/            модули (сборка наборов, фильтры, разбиения, тренеры, ...)
    experiments/     прогон матрицы, таблицы, отчёт, рисунки, hp_1d/hp_2d.json
    audit/           проход аудита по сырым .rpt, его отчёт, список 92 снятых заданий
    verification/    сверка напряжения волочения по 112 историям (в репозитории — reexport/)
    verification/results/  результаты этой сверки, на которые опирается раздел 3.1
    results/         журнал прогонов runs_v2.csv
    tests/           тесты инвариантов
    README.md, requirements.txt, LICENSE, CITATION.cff, .gitignore  (из release/)

Правила:
  * копируется только белый список, без масок и рекурсивного обхода каталогов,
    поэтому __pycache__, черновики и внутренние документы попасть в пакет не
    могут; --only принимает подмножество белого списка, любой другой путь —
    отказ;
  * выходной каталог должен быть пустым или новым и лежать вне репозитория;
    --force разрешает перезапись, только если в нём лежат исключительно
    файлы белого списка;
  * отсутствующий источник или символическая ссылка — отказ до копирования;
  * после копирования: sha256 каждого файла (сверяется с источником) в формате
    sha256sum (stdout; остальное — в stderr), проверка, что в каталоге нет
    ничего сверх скопированного, компиляция всех .py, поиск в текстах
    внутренних пометок (ссылок на рабочие документы, прежнюю подачу,
    абсолютных путей). Пометки печатаются как предупреждения; с --strict они
    дают код возврата 1.

Запуск:
    python release/make_release.py --out <каталог пакета>
    python release/make_release.py --out <каталог> > SHA256SUMS   # хеши в файл
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
from typing import Dict, List, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))

CODE = ["analytical.py", "build_datasets.py", "corruption.py", "dataset.py", "dataset2d.py",
        "filters.py", "jax_twin.py", "learning_curves.py", "physics_check.py", "protocol.py",
        "raw_probe.py", "splits.py", "stats.py", "trainer.py", "trainer2d.py",
        "vega_avitzur.py"]
EXPERIMENTS = ["analyze.py", "contours.py", "dump_predictions.py", "fem_row_2d.py",
               "figures.py", "make_results.py", "paper_tables.py", "residual_maps.py",
               "run_matrix.py", "hp_1d.json", "hp_2d.json"]
AUDIT = ["run_raw_audit.py", "make_report.py", "removed_1848_to_1756.csv"]
VERIFICATION = ["analytic_sigma_f.py", "drawing_force.py", "flow_stress.py",
                "read_history.py", "sec31_numbers.py", "verification_recalc.py"]
VERIFICATION_RESULTS = ["recalc_2026-09-28.json", "recalc_2026-09-28.csv",
                        "sec31_numbers_2026-10-02.txt"]
TESTS = ["test_pipeline.py", "test_training.py"]
TOP = ["README.md", "requirements.txt", "LICENSE", "CITATION.cff", ".gitignore"]

# путь в пакете → путь в репозитории (относительно REPO)
WHITELIST: List[Tuple[str, str]] = (
    [(f"code/{n}", f"code/{n}") for n in CODE]
    + [(f"experiments/{n}", f"experiments/{n}") for n in EXPERIMENTS]
    + [(f"audit/{n}", f"audit/{n}") for n in AUDIT]
    + [(f"verification/{n}", f"reexport/{n}") for n in VERIFICATION]
    + [(f"verification/results/{n}", f"reexport/results/{n}") for n in VERIFICATION_RESULTS]
    + [("results/runs_v2.csv", "results/runs_v2.csv")]
    + [(f"tests/{n}", f"tests/{n}") for n in TESTS]
    + [(n, f"release/{n}") for n in TOP]
)

TEXT_EXT = (".py", ".md", ".txt", ".csv", ".json", ".cff", ".gitignore")

# Пометки, которых не должно быть в публичном пакете: ссылки на рабочие
# документы и прежнюю подачу, обращения к людям, пути рабочей копии.
MARKERS: List[Tuple[str, str]] = [
    (r"ERRATA|PLAYBOOK|QA_DATA_AND_Z|VALIDATION\.md|RESULTS_v1_vs_v2|RAW_AUDIT\.md|"
     r"captions\.md|CHECKLIST\.md|MANUSCRIPT\.md|FIGURE_STYLE\.md", "внутренний документ"),
    (r"\bE-\d{2}\b", "номер записи реестра ошибок"),
    (r"CIRPJ|\bR[123]-c|Reviewer\s*#|[Рр]ецензент|отклон[её]нн|отклонили", "прежняя подача"),
    (r"[Рр]уководител|handover|correspondence|переписк", "люди и переписка"),
    (r"пункт \d+ плана|защитимее|уточнено \d", "рабочие пометки"),
    (r"(?<![\w.])publication/|(?<![\w.])reexport/|(?<![\w.])docs/|\.ipynb", "путь рабочей копии"),
    (r"/home/|/Users/|(?<![\w])[A-Z]:[\\/]", "абсолютный путь рабочей машины"),
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


def existing_files(root: str) -> List[str]:
    """Все файлы и символические ссылки под root (пути через '/')."""
    out = []
    for dp, dns, fns in os.walk(root):
        for n in fns + [d for d in dns if os.path.islink(os.path.join(dp, d))]:
            out.append(os.path.relpath(os.path.join(dp, n), root).replace(os.sep, "/"))
    return sorted(out)


def scan_markers(path: str) -> List[Tuple[int, str, str]]:
    hits = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            for rx, what in MARKERS:
                m = re.search(rx, line)
                if m:
                    hits.append((i, what, line.strip()[:120]))
    return hits


def select(only: Sequence[str]) -> List[Tuple[str, str]]:
    """Подмножество белого списка; путь вне белого списка — отказ (код 2)."""
    if not only:
        return list(WHITELIST)
    allowed: Dict[str, str] = dict(WHITELIST)
    bad = [p for p in only if p not in allowed]
    if bad:
        log("ОТКАЗ: не в белом списке:")
        for p in bad:
            log(f"  {p}")
        sys.exit(2)
    return [(p, allowed[p]) for p in dict.fromkeys(only)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="", help="выходной каталог пакета (обязателен, кроме --list)")
    ap.add_argument("--repo", default=REPO, help="корень репозитория (по умолчанию — над release/)")
    ap.add_argument("--only", nargs="*", default=[],
                    help="скопировать только эти пути пакета (из белого списка)")
    ap.add_argument("--force", action="store_true",
                    help="перезаписать каталог, если в нём только файлы белого списка")
    ap.add_argument("--strict", action="store_true",
                    help="код возврата 1 при найденных внутренних пометках")
    ap.add_argument("--list", action="store_true", help="напечатать белый список и выйти")
    a = ap.parse_args()

    if a.list:
        for dst, src in WHITELIST:
            print(f"{dst:45s} ← {src}")
        return 0
    if not a.out:
        ap.error("нужен --out")

    plan = select(a.only)
    repo = os.path.abspath(a.repo)
    out = os.path.abspath(a.out)
    if out == repo or out.startswith(repo + os.sep):
        # пакет внутри дерева-источника смешался бы с рабочими файлами
        log(f"ОТКАЗ: выходной каталог внутри репозитория ({out})")
        return 2

    # источники: все на месте, обычные файлы, не ссылки
    problems = []
    for dst, src in plan:
        p = os.path.join(repo, src)
        if os.path.islink(p):
            problems.append(f"символическая ссылка: {src}")
        elif not os.path.isfile(p):
            problems.append(f"нет файла: {src}")
    if problems:
        log("ОТКАЗ: источники не готовы, ничего не скопировано:")
        for m in problems:
            log(f"  {m}")
        return 1

    # выходной каталог: новый, пустой или (с --force) только файлы белого списка
    whitelist_dst = {d for d, _ in WHITELIST}
    if os.path.exists(out):
        if not os.path.isdir(out):
            log(f"ОТКАЗ: {out} существует и это не каталог")
            return 2
        present = existing_files(out)
        if present and not a.force:
            log(f"ОТКАЗ: {out} не пуст ({len(present)} файлов); --force перезапишет файлы белого списка")
            return 2
        stray = [p for p in present if p not in whitelist_dst]
        if stray:
            log(f"ОТКАЗ: в {out} есть файлы вне белого списка, перезаписывать не буду:")
            for p in stray[:20]:
                log(f"  {p}")
            return 2

    # копирование
    log(f"репозиторий: {repo}")
    log(f"пакет:       {out}")
    log(f"файлов:      {len(plan)}")
    for dst, src in plan:
        s = os.path.join(repo, src)
        d = os.path.join(out, dst)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(s, d)
        hs, hd = sha256(s), sha256(d)
        if hs != hd:
            log(f"ОШИБКА: копия {dst} не совпала с источником")
            return 1
        print(f"{hd}  {dst}", flush=True)

    # в каталоге не должно быть ничего сверх скопированного
    copied = {d for d, _ in plan}
    extra = sorted(set(existing_files(out)) - copied)
    if extra:
        log("ОШИБКА: в пакете есть файлы вне белого списка:")
        for p in extra:
            log(f"  {p}")
        return 1

    # синтаксис .py (без записи __pycache__)
    bad_py = []
    for dst, _ in plan:
        if dst.endswith(".py"):
            try:
                with open(os.path.join(out, dst), encoding="utf-8") as fh:
                    compile(fh.read(), dst, "exec")
            except SyntaxError as exc:
                bad_py.append(f"{dst}: {exc}")
    if bad_py:
        log("ОШИБКА: не компилируются:")
        for m in bad_py:
            log(f"  {m}")
        return 1

    # внутренние пометки в текстах
    n_hits = 0
    for dst, _ in plan:
        name = os.path.basename(dst)
        if not (dst.endswith(TEXT_EXT) or name in ("LICENSE", ".gitignore")):
            continue
        for line, what, text in scan_markers(os.path.join(out, dst)):
            if n_hits == 0:
                log("\nвнутренние пометки (проверить перед публикацией):")
            n_hits += 1
            log(f"  {dst}:{line}  [{what}]  {text}")

    log(f"\nскопировано {len(plan)} файлов; внутренних пометок: {n_hits}")
    return 1 if (n_hits and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
