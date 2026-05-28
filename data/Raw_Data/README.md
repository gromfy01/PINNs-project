# data/Raw_Data/

Исходные отчёты численного моделирования (Abaqus, формат `.rpt`).
Используются как вход для препроцессинга в `preprocessing/preproc.ipynb`
и служат внешним тест-сетом для валидации модели деформаций.

## Скачивание

Архив (≈ 1.2 ГБ) выложен как asset к релизу:

> **TBD — replace with release URL once published:**
> `https://github.com/gromfy01/PINNs-project/releases/latest`

После скачивания распаковать в эту же папку — должна получиться
структура `data/Raw_Data/Vel_*/*.rpt`.

## Как добавить в Release (для автора)

```bash
gh release create raw-data-v1 \
  --title "Raw FEM data (Abaqus .rpt)" \
  --notes "Test set for the strain pipeline." \
  Raw_Data.zip
```

После создания релиза заменить URL выше на постоянную ссылку
вида `…/releases/download/raw-data-v1/Raw_Data.zip`.
