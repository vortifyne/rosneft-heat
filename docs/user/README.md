# Запуск и просмотр результатов

Для работы нужны готовая программа `basin_forward_solver` с её библиотеками и инструменты
из каталога `tools/`. Подготовка данных и просмотр используют Python; зависимости устанавливает
`uv` по файлам `pyproject.toml` и `uv.lock`. Команды выполняются из папки программы.

## Расчёт

Поместите подготовленные таблицы в `input/`: `configurations.csv`, `layers.csv`,
`boundaries.csv`, `lithotypes.csv`, `kinetics.csv`. Запустите:

```bash
./basin_forward_solver input result
```

Для готовой сборки репозитория в Docker вместо этой команды используется:

```bash
tools/run_basin_forward.sh input result
```

В `result/` появятся снимки `snapshot_*.cgrd` и итоговый файл `summary.txt` с параметрами
и числом шагов и итераций. Для нового расчёта выбирайте новый каталог результатов.

## Просмотр

```bash
uv run --frozen python tools/view_basin_snapshots.py result
```

Ползунок выбирает сохранённое состояние. Для отображения зрелости витринита добавьте
`--field reflectance`, для слоёв — `--field layer`.

## Подготовка исходных таблиц

Если данные даны в XLSX, положите их в `data/` и перед расчётом выполните:

```bash
uv run --frozen python tools/prepare_basin_input.py data input
```
