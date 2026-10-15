# -*- coding: utf-8 -*-
"""
reexport_fields.py — повторная выгрузка полей из .odb для валидации.

ЗАПУСКАТЬ ПОД ABAQUS PYTHON, не под обычным:
    abaqus python reexport_fields.py reexport_list.csv out_dir

Что выгружается и зачем (ничего из этого нет в имеющихся .rpt):
    RF      — реакция на опорной точке волоки = усилие волочения.
              Прямое сравнение с Зибелем и Авитцуром НА ДАННЫХ СТАТЬИ.
    PEEQ    — накопленная пластическая деформация. Фактор избыточной
              работы Φ количественно; сейчас доступна только нижняя оценка.
    CPRESS, CSHEAR — контактные давление и сдвиг на поверхности проволоки.
              Проверка члена трения через контактные напряжения.
    S, LE, COORD — то же, что уже есть, для сверки с прежней выгрузкой.
    TEMP / NT11 — если есть; ожидается, что нет (расчёт не термомеханический).

Кадры: ПОСЛЕДНИЙ (для сверки с имеющимся) и ещё до 4 равномерно по шагу —
чтобы видеть напряжение волочения, пока проволока в волоке, а не только
остаточное после выхода. Именно отсутствие промежуточных кадров сделало
усилие волочения невосстановимым на 1756 наборах датасета.

Формат вывода — CSV на кадр, с одной строкой на узел проволоки, плюс
отдельный файл с историей RF по кадрам. Читается pandas без Abaqus.

Скрипт терпим к отсутствию поля: пишет, чего нет, и идёт дальше. Первый
прогон стоит сделать на 2-3 файлах и посмотреть missing.txt.
"""
from __future__ import print_function
import csv
import os
import sys

try:
    from odbAccess import openOdb
    from abaqusConstants import NODAL, INTEGRATION_POINT
except ImportError:
    sys.stderr.write("Нужен Abaqus Python: abaqus python reexport_fields.py ...\n")
    sys.exit(2)

WANT_NODAL = ["S", "LE", "COORD", "PEEQ", "TEMP", "NT11"]
WANT_CONTACT = ["CPRESS", "CSHEAR", "CSHEAR1"]
N_EXTRA_FRAMES = 4


def pick_frames(step):
    n = len(step.frames)
    if n <= 1:
        return [n - 1]
    idx = sorted(set([int(round(i * (n - 1) / float(N_EXTRA_FRAMES)))
                      for i in range(N_EXTRA_FRAMES + 1)]))
    return idx


def wire_instance(odb):
    for name, inst in odb.rootAssembly.instances.items():
        if "WIRE" in name.upper():
            return name, inst
    # запасной вариант: инстанс с наибольшим числом узлов
    name = max(odb.rootAssembly.instances.keys(),
               key=lambda k: len(odb.rootAssembly.instances[k].nodes))
    return name, odb.rootAssembly.instances[name]


def die_reference_nodes(odb):
    """Опорные точки волоки: там сидит реакция RF."""
    out = []
    for name, inst in odb.rootAssembly.instances.items():
        if "DIE" in name.upper():
            for n in inst.nodes:
                out.append((name, n.label))
    return out


def export_one(odb_path, out_dir, missing):
    job = os.path.splitext(os.path.basename(odb_path))[0]
    if not os.path.exists(odb_path):
        missing.append((job, "ODB_NOT_FOUND", odb_path))
        return
    try:
        odb = openOdb(odb_path, readOnly=True)
    except Exception as e:
        missing.append((job, "ODB_OPEN_FAILED", str(e)))
        return
    try:
        step = odb.steps[odb.steps.keys()[-1]]
        wname, wire = wire_instance(odb)
        die_refs = die_reference_nodes(odb)
        frames = pick_frames(step)

        # история RF по всем кадрам — это и есть усилие волочения
        with open(os.path.join(out_dir, job + "_RF.csv"), "w") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "step_time", "instance", "node", "RF1", "RF2", "RF_mag"])
            for fi, fr in enumerate(step.frames):
                if "RF" not in fr.fieldOutputs:
                    if fi == 0:
                        missing.append((job, "NO_RF", ""))
                    break
                rf = fr.fieldOutputs["RF"]
                for iname, lbl in die_refs:
                    inst = odb.rootAssembly.instances[iname]
                    sub = rf.getSubset(region=inst.getNodeFromLabel(lbl))
                    for v in sub.values:
                        d = list(v.data) + [0.0, 0.0]
                        w.writerow([fi, fr.frameValue, iname, lbl, d[0], d[1], v.magnitude])

        # поля по узлам проволоки на выбранных кадрах
        for fi in frames:
            fr = step.frames[fi]
            cols, data = ["node", "step_time"], {}
            for n in wire.nodes:
                data[n.label] = [n.label, fr.frameValue]
            for fname in WANT_NODAL:
                if fname not in fr.fieldOutputs:
                    missing.append((job, "NO_" + fname, "frame %d" % fi))
                    continue
                fo = fr.fieldOutputs[fname]
                sub = fo.getSubset(region=wire, position=NODAL)
                if not sub.values:
                    # PEEQ живёт в точках интегрирования — экстраполируем к узлам
                    sub = fo.getSubset(region=wire, position=INTEGRATION_POINT)
                    if not sub.values:
                        missing.append((job, "EMPTY_" + fname, "frame %d" % fi))
                        continue
                    # грубо: среднее по элементам, инцидентным узлу
                    acc = {}
                    for v in sub.values:
                        el = wire.getElementFromLabel(v.elementLabel)
                        for nl in el.connectivity:
                            acc.setdefault(nl, []).append(
                                v.data if hasattr(v.data, "__len__") else [v.data])
                    lab = sub.values[0].componentLabels or [fname]
                    cols += ["%s.%s" % (fname, c) for c in lab]
                    for nl, rows in data.items():
                        vals = acc.get(nl)
                        if vals:
                            m = [sum(x[i] for x in vals) / float(len(vals))
                                 for i in range(len(vals[0]))]
                        else:
                            m = [float("nan")] * len(lab)
                        rows += m
                    continue
                lab = sub.values[0].componentLabels or [fname]
                cols += ["%s.%s" % (fname, c) for c in lab]
                got = {}
                for v in sub.values:
                    got[v.nodeLabel] = list(v.data) if hasattr(v.data, "__len__") else [v.data]
                for nl, rows in data.items():
                    rows += got.get(nl, [float("nan")] * len(lab))
            # контактные величины — на поверхностных узлах, остальным NaN
            for fname in WANT_CONTACT:
                if fname not in fr.fieldOutputs:
                    if fi == frames[-1]:
                        missing.append((job, "NO_" + fname, ""))
                    continue
                sub = fr.fieldOutputs[fname].getSubset(region=wire)
                got = {v.nodeLabel: (v.data if not hasattr(v.data, "__len__") else v.data[0])
                       for v in sub.values}
                cols.append(fname)
                for nl, rows in data.items():
                    rows.append(got.get(nl, float("nan")))
            with open(os.path.join(out_dir, "%s_frame%03d.csv" % (job, fi)), "w") as fh:
                w = csv.writer(fh)
                w.writerow(cols)
                for nl in sorted(data):
                    w.writerow(data[nl])
    finally:
        odb.close()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    lst, out_dir = sys.argv[1], sys.argv[2]
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    missing = []
    with open(lst) as fh:
        rows = list(csv.DictReader(fh))
    for i, row in enumerate(rows):
        print("[%d/%d] %s" % (i + 1, len(rows), row["job"]))
        export_one(row["odb_path"], out_dir, missing)
    with open(os.path.join(out_dir, "missing.txt"), "w") as fh:
        for m in missing:
            fh.write("\t".join(m) + "\n")
    print("готово; пропусков: %d (см. missing.txt)" % len(missing))


if __name__ == "__main__":
    main()
