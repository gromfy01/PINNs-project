# -*- coding: utf-8 -*-
"""
check_odb_alive.py — за минуту узнать, какие .odb ещё существуют.

Обычный Python, Abaqus не нужен. Запускать на машине, где смонтированы
диски E:, H:, V:.

    python check_odb_alive.py reexport_list.csv          # только отобранные 30
    python check_odb_alive.py all_odb_paths.csv           # все 2443

Печатает сводку по партиям и пишет alive.csv рядом.
"""
import csv
import os
import sys
from collections import Counter

lst = sys.argv[1] if len(sys.argv) > 1 else "reexport_list.csv"
rows = list(csv.DictReader(open(lst, encoding="utf-8")))
alive, dead = Counter(), Counter()
with open("alive.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["job", "odb_path", "exists", "size_MB"])
    for r in rows:
        p = r["odb_path"]
        ok = os.path.exists(p)
        (alive if ok else dead)[os.path.dirname(p)] += 1
        w.writerow([r["job"], p, int(ok), round(os.path.getsize(p) / 1e6, 1) if ok else ""])
print("партия                                         жив  мёртв")
for d in sorted(set(alive) | set(dead)):
    print("%-45s %4d  %4d" % (d, alive[d], dead[d]))
print("\nитого: жив %d, нет %d из %d" % (sum(alive.values()), sum(dead.values()), len(rows)))
