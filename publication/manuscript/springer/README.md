# Вёрстка Springer Nature (IJAMT, Engineering with Computers)

Текст статьи один — `../overleaf/original.tex`. Отсюда генерируется вёрстка шаблона
Springer Nature (`sn-jnl.cls`, версия 3, октябрь 2023): тело статьи переносится дословно,
меняются только титульный блок, окружения рисунков и таблиц и приложения. Рисунки и
`.bib` берутся из `../overleaf/`.

    ./build.sh            # convert.py → manuscript_sn.tex → manuscript_sn.pdf
    ./pack.sh             # пакет для подачи: submission/ и submission.zip

Что делает `convert.py` (каждая замена проверяется по ожидаемому числу вхождений):

- `figure*`/`table*` → одноколоночные; у таблиц шрифт класса (8 pt) вместо `\small`;
- четыре широкие таблицы (main, paired, backend, admissibility) повёрнуты на отдельные
  страницы, у трёх (families, curves, corrupt) уменьшен межколоночный отступ;
- снят `\clearpage\clearpage` перед выводами (он нужен только двухколоночной ASME);
- номенклатура — своим окружением (в sn-jnl её нет), приложения — `appendices`
  с нумерацией формул A1, B1…;
- добавлен раздел Declarations с заглушками (Funding, Competing interests,
  Data availability, Author contributions) — текст заполнить перед подачей;
- все комментарии TeX вырезаются: в исходнике рабочие пометки, в файл для журнала
  они попасть не должны. `--keep-comments` оставляет их.

`pack.sh` собирает пакет для подачи одним каталогом без подпапок (IJAMT прямо требует
«do not use subfolders»): `manuscript.tex`, класс, стили, `references.bib` без
комментариев, рисунки, скомпилированный PDF и `.bbl`.

Стиль ссылок `sn-basic` (нумерация по алфавиту, названия в sentence case — поэтому
аббревиатуры и марки сталей в `.bib` защищены фигурными скобками).

Файлы класса взяты с зеркала шаблона на GitHub
(`DanySK/template-latex-springer-nature-sn-jnl`, 12.10.2026); sha256:

    82046223888ead55b3177866b6a58d0aa488c7b81a866b1456b1b31275f717ca  sn-jnl.cls
    1e881d925dfe17dfd86b1f9338fd80fe4440f4dbe016ef8e9b58bde5f6b5c6b4  sn-basic.bst
    9f54bd1e93035022a5a6cfe0d5e0a2b0aedcb762079d3e487befa2bf2ecd1db8  sn-mathphys-num.bst

Перед подачей сверить с актуальной версией шаблона на сайте Springer (на Overleaf
сейчас v3.1, декабрь 2024).
