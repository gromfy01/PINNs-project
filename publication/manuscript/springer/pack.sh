#!/bin/bash
# Пакет для подачи в журнал Springer: один каталог без подпапок (так требуют IJAMT и
# Engineering with Computers) — .tex, класс, стиль ссылок, references.bib, рисунки, PDF, и zip.
# Запуск из publication/manuscript/springer/:  ./pack.sh
set -eu
cd "$(dirname "$0")"
OUT=submission
rm -rf $OUT && mkdir $OUT
python3 convert.py ../overleaf/original.tex $OUT/manuscript.tex --flat
cp sn-jnl.cls sn-basic.bst sn-mathphys-num.bst $OUT/
grep -v "^[[:space:]]*%" ../overleaf/asmejour-sample.bib > $OUT/references.bib  # без рабочих пометок
for f in $(grep -o '\\includegraphics\(\[[^]]*\]\)\?{[^}]*}' $OUT/manuscript.tex | sed 's/.*{\(.*\)}/\1/'); do
  cp ../overleaf/pictures/$f $OUT/
done
cd $OUT
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error manuscript.tex > /dev/null
bibtex manuscript > /dev/null
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error manuscript.tex > /dev/null
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error manuscript.tex > /dev/null
grep -E "undefined|Error" manuscript.log | head -5 || true
rm -f manuscript.aux manuscript.blg manuscript.log manuscript.out
cd .. && rm -f submission.zip && python3 -c "
import zipfile, os
with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir('submission')): z.write(os.path.join('submission', f), f)
print('submission.zip:', len(z.namelist()) if False else '', sorted(os.listdir('submission')))"
