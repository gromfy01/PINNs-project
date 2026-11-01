"""
pack_histories.py — архив 112 X-Y отчётов Abaqus (истории по времени), по
которым verification/verification_recalc.py и verification/sec31_numbers.py
считают напряжение волочения и предел текучести для раздела 3.1.

Берёт из каталога SRC все *.rpt и *.rpt.gz (без обхода подкаталогов) и пишет
zip, в котором они лежат в одной папке вместе с MANIFEST.sha256 — sha256
каждого отчёта в том виде, в каком он лежит в архиве, в формате sha256sum:

    history_reports_112/
        MANIFEST.sha256
        1020_2a_16_rd_1000_cal_0_v_10_f_0100.rpt
        ...

После распаковки проверка — `sha256sum -c MANIFEST.sha256` внутри папки, а сама
папка подаётся verification_recalc.py и sec31_numbers.py как DIR.

Проверки до записи (любая неудача — отказ, архив не пишется):
  * отчётов ровно --expect (по умолчанию 112; 0 — не проверять);
  * одно задание не лежит одновременно как .rpt и .rpt.gz;
  * у каждого файла в первых строках заголовок X-Y отчёта истории
    (COORD:COOR1 и RF), а не Field Output Report основного набора.

Архив воспроизводим: порядок по имени, фиксированные дата и права у всех
записей; .rpt сжимаются (deflate), .rpt.gz кладутся без повторного сжатия.
После записи архив перечитывается и каждая запись сверяется с манифестом.

Запуск:
    python release/pack_histories.py SRC --out history_reports_112.zip
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import sys
import tempfile
import zipfile
from typing import List, Tuple

EXPECT = 112
MANIFEST = "MANIFEST.sha256"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
HEADER_LINES = 5


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def job_of(name: str) -> str:
    return name[:-7] if name.endswith(".rpt.gz") else name[:-4]


def looks_like_history(path: str) -> bool:
    """Заголовок X-Y отчёта истории: подписи COORD:COOR1 и RF в первых строках."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="latin-1", errors="replace") as fh:
        head = "".join(fh.readline() for _ in range(HEADER_LINES))
    return "COORD:COOR1" in head and "RF" in head


def collect(src: str) -> Tuple[List[str], List[str]]:
    """(отчёты, прочие файлы) в каталоге src, по имени."""
    reports, other = [], []
    for n in sorted(os.listdir(src)):
        p = os.path.join(src, n)
        if not os.path.isfile(p):
            continue
        (reports if n.endswith((".rpt", ".rpt.gz")) else other).append(n)
    return reports, other


def entry(arcname: str, compress: int) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(arcname, date_time=ZIP_DATE)
    zi.compress_type = compress
    zi.create_system = 3                    # unix: права берутся из external_attr
    zi.external_attr = 0o100644 << 16
    return zi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="каталог с X-Y отчётами (*.rpt или *.rpt.gz)")
    ap.add_argument("--out", default="history_reports_112.zip", help="путь к zip")
    ap.add_argument("--folder", default="",
                    help="папка внутри архива (по умолчанию — имя zip без .zip)")
    ap.add_argument("--expect", type=int, default=EXPECT,
                    help=f"ожидаемое число отчётов (по умолчанию {EXPECT}; 0 — не проверять)")
    ap.add_argument("--force", action="store_true", help="перезаписать существующий zip")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    if not os.path.isdir(src):
        print(f"ОТКАЗ: нет каталога {src}", file=sys.stderr)
        return 1
    out = os.path.abspath(a.out)
    if os.path.exists(out) and not a.force:
        print(f"ОТКАЗ: {out} уже есть (--force перезапишет)", file=sys.stderr)
        return 1
    base = os.path.basename(out)
    folder = a.folder or (base[:-4] if base.endswith(".zip") else base)
    if not folder or "/" in folder or "\\" in folder or folder in (".", ".."):
        print(f"ОТКАЗ: недопустимое имя папки в архиве: {folder!r}", file=sys.stderr)
        return 1

    reports, other = collect(src)
    problems = []
    if a.expect and len(reports) != a.expect:
        problems.append(f"отчётов {len(reports)}, ожидалось {a.expect}")
    jobs = [job_of(n) for n in reports]
    dup = sorted({j for j in jobs if jobs.count(j) > 1})
    if dup:
        problems.append(f"задания и как .rpt, и как .rpt.gz: {dup[:5]}")
    not_hist = [n for n in reports if not looks_like_history(os.path.join(src, n))]
    if not_hist:
        problems.append(f"не похожи на X-Y отчёт истории ({len(not_hist)}): {not_hist[:5]}")
    if problems:
        print("ОТКАЗ, архив не записан:", file=sys.stderr)
        for m in problems:
            print(f"  {m}", file=sys.stderr)
        return 1
    if other:
        print(f"пропущены файлы, не являющиеся отчётами ({len(other)}): {other[:5]}")

    # манифест и архив; запись во временный файл рядом, затем переименование
    digests = {n: sha256_file(os.path.join(src, n)) for n in reports}
    manifest = "".join(f"{digests[n]}  {n}\n" for n in reports).encode("ascii")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pack_", suffix=".zip", dir=os.path.dirname(out) or ".")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", allowZip64=True) as zf:
            zf.writestr(entry(f"{folder}/{MANIFEST}", zipfile.ZIP_DEFLATED), manifest)
            for i, n in enumerate(reports, 1):
                with open(os.path.join(src, n), "rb") as fh:
                    data = fh.read()
                if sha256_bytes(data) != digests[n]:
                    raise RuntimeError(f"{n} изменился во время упаковки")
                comp = zipfile.ZIP_STORED if n.endswith(".gz") else zipfile.ZIP_DEFLATED
                zf.writestr(entry(f"{folder}/{n}", comp), data)
                if i % 28 == 0:
                    print(f"  {i}/{len(reports)}", flush=True)

        # перечитать и сверить с манифестом
        with zipfile.ZipFile(tmp) as zf:
            names = zf.namelist()
            want = [f"{folder}/{MANIFEST}"] + [f"{folder}/{n}" for n in reports]
            if names != want:
                raise RuntimeError("состав архива не совпал с ожидаемым")
            lines = zf.read(f"{folder}/{MANIFEST}").decode("ascii").splitlines()
            for line in lines:
                h, n = line.split("  ", 1)
                if sha256_bytes(zf.read(f"{folder}/{n}")) != h:
                    raise RuntimeError(f"запись {n} не совпала с манифестом")
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    print(f"записано: {out}")
    print(f"  отчётов {len(reports)} (.rpt {sum(not n.endswith('.gz') for n in reports)}, "
          f".rpt.gz {sum(n.endswith('.gz') for n in reports)}), папка в архиве {folder}/")
    print(f"  размер {os.path.getsize(out) / 1e6:.1f} МБ")
    print(f"  sha256 манифеста {sha256_bytes(manifest)}")
    print(f"  sha256 архива    {sha256_file(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
