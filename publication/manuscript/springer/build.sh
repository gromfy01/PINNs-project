#!/bin/bash
# Сборка вёрстки Springer Nature: convert.py, затем pdflatex, bibtex, pdflatex x2 (без shell escape).
# Запуск из publication/manuscript/springer/:  ./build.sh
set -eu
cd "$(dirname "$0")"
python3 convert.py "$@"
J=manuscript_sn
rm -f $J.aux $J.bbl $J.blg $J.out
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error $J.tex > $J.run.log 2>&1 || { tail -30 $J.log; exit 1; }
bibtex $J > $J.bib.log 2>&1 || { cat $J.bib.log; exit 1; }
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error $J.tex > $J.run.log 2>&1
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error $J.tex > $J.run.log 2>&1
grep -E "undefined|Error" $J.log | head -20 || true
python3 -c "import pymupdf,sys; print('pages:', pymupdf.open(sys.argv[1]).page_count)" $J.pdf 2>/dev/null || true
