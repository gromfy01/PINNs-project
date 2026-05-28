# data/Raw_Data/

Исходные отчёты численного моделирования (Abaqus, формат `.rpt`).
Используются как вход для препроцессинга в `preprocessing/preproc.ipynb`
и служат внешним тест-сетом для валидации модели деформаций.

## Скачивание

Архив (≈ 1.2 ГБ) выложен как asset к релизу
[`raw-data-v1`](https://github.com/gromfy01/PINNs-project/releases/tag/raw-data-v1).

Прямая ссылка на zip:

> https://github.com/gromfy01/PINNs-project/releases/download/raw-data-v1/Raw_Data.zip

Скачать из терминала:

```bash
curl -L -o Raw_Data.zip \
  https://github.com/gromfy01/PINNs-project/releases/download/raw-data-v1/Raw_Data.zip
unzip Raw_Data.zip -d .
```

После распаковки структура: `data/Raw_Data/Vel_*/*.rpt` — этого
достаточно, чтобы запустить `preprocessing/preproc.ipynb`.
