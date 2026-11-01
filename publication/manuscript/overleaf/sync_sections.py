# -*- coding: utf-8 -*-
"""
sync_sections.py — перенести содержимое sections/*.tex в original.tex.

original.tex — это файл статьи, как он лежит в Overleaf. Куски, которые
собираются из репозитория (разделы и таблицы), помечены в нём парой строк

    % >>> inlined sections/<имя>.tex
    ...
    % <<< end sections/<имя>.tex

Скрипт заменяет то, что между маркерами, содержимым соответствующего файла из
sections/. Таблицы туда попадают из ../tables/, которые строит
publication/experiments/paper_tables.py, поэтому подписи и числа не разъезжаются.

    python publication/manuscript/overleaf/sync_sections.py            # проверить
    python publication/manuscript/overleaf/sync_sections.py --write    # записать
"""
from __future__ import annotations

import argparse
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL = os.path.join(HERE, "original.tex")
SECTIONS = os.path.join(HERE, "sections")
TABLES = os.path.abspath(os.path.join(HERE, "..", "tables"))

# имя куска -> откуда брать содержимое
SOURCES = {
    "tab_main": os.path.join(TABLES, "tab_main.tex"),
    "tab_paired": os.path.join(TABLES, "tab_paired.tex"),
    "tab_backend": os.path.join(TABLES, "tab_backend.tex"),
    "tab_corrupt": os.path.join(TABLES, "tab_corrupt.tex"),
}

# Блоки прозы, которые извлекаются из статьи в sections/ (--extract). Выводы живут
# в manuscript/sec4_conclusions.tex, поэтому их копия кладётся туда же.
PROSE_ELSEWHERE = {
    "sec4_conclusions": os.path.abspath(os.path.join(HERE, "..", "sec4_conclusions.tex")),
}


def blocks(text: str) -> dict:
    """{имя: (начало содержимого, конец содержимого)} по маркерам."""
    out = {}
    for m in re.finditer(r"% >>> inlined sections/(\S+)\.tex\n", text):
        name = m.group(1)
        end = re.search(r"% <<< end sections/" + re.escape(name) + r"\.tex", text[m.end():])
        if end:
            out[name] = (m.end(), m.end() + end.start())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="втянуть таблицы из ../tables/ в original.tex")
    ap.add_argument("--extract", action="store_true",
                    help="выгрузить блоки прозы из original.tex в sections/ (обратное направление)")
    a = ap.parse_args()

    text = open(ORIGINAL, encoding="utf-8").read()
    found = blocks(text)

    if a.extract:
        # original.tex — источник истины для прозы: статья правится в нём
        for name, (lo, hi) in found.items():
            if name in SOURCES:
                continue
            dst = PROSE_ELSEWHERE.get(name, os.path.join(SECTIONS, name + ".tex"))
            body = text[lo:hi].strip() + "\n"
            if os.path.exists(dst) and open(dst, encoding="utf-8").read() == body:
                print(f"  {name}: без изменений")
                continue
            open(dst, "w", encoding="utf-8").write(body)
            print(f"  {name}: выгружен в {os.path.relpath(dst, HERE)}")
        return

    changed = []
    # с конца, чтобы смещения не поехали
    for name in sorted(found, key=lambda n: -found[n][0]):
        if name not in SOURCES:
            continue          # проза правится в original.tex; сюда её не втягиваем
        src = SOURCES[name]
        if not os.path.exists(src):
            print(f"  {name}: источник не найден ({src}) — пропуск")
            continue
        new = open(src, encoding="utf-8").read().strip() + "\n"
        lo, hi = found[name]
        if text[lo:hi].strip() == new.strip():
            print(f"  {name}: без изменений")
            continue
        changed.append(name)
        text = text[:lo] + new + text[hi:]
        print(f"  {name}: обновлён из {os.path.relpath(src, HERE)}")

    if not changed:
        print("\nвсё уже синхронно")
        return
    if a.write:
        shutil.copyfile(ORIGINAL, ORIGINAL + ".bak")
        open(ORIGINAL, "w", encoding="utf-8").write(text)
        print(f"\nзаписано в original.tex (копия прежнего — original.tex.bak); обновлено: {', '.join(changed)}")
    else:
        print(f"\nбудет обновлено: {', '.join(changed)} — запустить с --write")


if __name__ == "__main__":
    main()
