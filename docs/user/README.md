# Пользовательская документация

Раздел предназначен для описания запуска приложения, входных данных, параметров командной строки и выходных результатов.

## Требования

Проект поддерживает два способа сборки: **в изолированном Docker-контейнере** (рекомендуется для воспроизводимости) и **локально на хост-системе**.

### Вариант 1: Сборка через Docker (Рекомендуемый)

- **Docker Engine** (версия `>= 20.10`)
- **GNU Make**

### Вариант 2: Локальная сборка

- **Компилятор C++20**: GCC **>= 13** или Clang **>= 16**
- **CMake**: **>= 3.28**
- **Ninja**
- **vcpkg**: переменная `$VCPKG_ROOT` должна указывать на установленный vcpkg (добавить в `.bashrc` или `.zshrc`)
- **Системные зависимости** (для сборки SuiteSparse / BLAS):
  - _Ubuntu/Debian:_ `sudo apt install build-essential gfortran liblapack-dev libblas-dev pkg-config`

## Сборка через Docker

Все зависимости (CMake, компиляторы, vcpkg) уже изолированы в образе.

### 1. Собрать Docker-образ (выполняется 1 раз)

```bash
make docker-image
```

### 2. Сборка проекта

- **Release-сборка:**

  ```bash
  make docker-release
  ```

- **Debug-сборка:**

  ```bash
  make docker-debug
  ```

### 3. Запуск тестов

Запуск набора юнит-тестов (Google Test / CTest):

```bash
make docker-test
```

### 4. Запуск программы и бенчмарков

- **Запуск солвера:**

  ```bash
  make docker-run
  ```

- **Запуск Google Benchmark:**

  ```bash
  make docker-bench
  ```

## Локальная сборка (без Docker)

Если у вас настроен `vcpkg` на хост-машине:

### Через Makefile

```bash
# 1. Сборка Release
make local-release

# 2. Запуск тестов
make local-test

# 3. Запуск солвера
make local-run

# 4. Запуск бенчмарков
make local-bench
```

### Напрямую через CMake CLI (CMake Presets)

```bash
# Конфигурация и сборка Release
cmake --preset release -DVCPKG_INSTALLED_DIR=./vcpkg_installed
cmake --build --preset release

# Запуск тестов
ctest --test-dir build/release --output-on-failure

# Запуск бинарника
./build/release/heat_solver
```

## Очистка артефактов сборки

Удаление директорий сборки (`build/`), установленных зависимостей (`vcpkg_installed/`) и временных файлов:

```bash
make clean
```

## Структура проекта

- `src/` — исходный код солвера (дискретизация, линейные/нелинейные методы, временные схемы).
- `tests/` — модульные тесты на базе Google Test.
- `benchmarks/` — замеры производительности на базе Google Benchmark.
- `docs/` — архитектурная и пользовательская документация.
- `CMakePresets.json` — конфигурации сборки (Debug, Release).
- `vcpkg.json` — манифест внешних зависимостей (Eigen3, fmt, SuiteSparse UMFPACK, GTest, Benchmark).
