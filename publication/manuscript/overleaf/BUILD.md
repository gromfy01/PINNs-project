# Локальная сборка статьи

На Overleaf `original.tex` собирается как есть. Ниже — как собрать его же
в контейнере, где стоит TeX Live 2023 из репозиториев Ubuntu.

    apt-get install -y --no-install-recommends \
        texlive-latex-base texlive-latex-recommended texlive-latex-extra \
        texlive-fonts-recommended texlive-fonts-extra texlive-science \
        texlive-bibtex-extra texlive-plain-generic texlive-lang-european tex-gyre

    pdflatex original && bibtex original && pdflatex original && pdflatex original

## Три расхождения с Overleaf

Правятся в рабочей копии, в репозиторий не вносятся: на Overleaf (TeX Live 2024)
всё это работает, а `asmejour.cls` должен оставаться эталонным v1.18.

1. `\RequirePackage[hyperref,\MyColorOption,...]{xcolor}` (строка 324 класса).
   Ядро LaTeX 2023-11-01 не раскрывает `\MyColorOption` в списке опций, и xcolor
   видит имя макроса как опцию. Локально достаточно
   `\RequirePackage[dvipsnames,svgnames,x11names]{xcolor}`.

2. Класс грузит `hyperxmp` перед `hyperref` (строки 780 и 782), а hyperxmp 5.12
   требует обратного порядка. Связано с опцией `pdf-a` в `\documentclass`:
   она передаёт hyperref ключи `pdfapart`, `pdfaconformance`, которые определяет
   именно hyperxmp. Локально проще снять `pdf-a` — на вёрстку это не влияет,
   меняется только метаданные PDF/A.

3. Шрифты TeX Gyre Heros Condensed (`qhvc`) нужны для заголовка и блока авторов;
   без пакета `tex-gyre` они подменяются, и титул выглядит иначе.

## Сборка на macOS без прав администратора

TinyTeX ставится в домашнюю папку, sudo не нужен:

    curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh
    export PATH=$PATH:$HOME/Library/TinyTeX/bin/universal-darwin
    tlmgr install extsizes collection-latexrecommended collection-latexextra \
                  collection-fontsrecommended collection-mathscience

На TeX Live 2026 ни одна из трёх правок выше не нужна: класс собирается как есть,
включая pdf-a и исходный порядок hyperxmp/hyperref. Расхождения касаются только
TeX Live 2023.

## Что должно получиться

19 страниц, ноль ошибок, ноль неразрешённых ссылок и цитат.
Полный цикл (pdflatex, bibtex, pdflatex, pdflatex) занимает около 4 секунд,
так что таймаут сборки на стороне Overleaf этим документом не объясняется.
