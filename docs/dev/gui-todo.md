# GUI доработки после изменения движка

- В `desktop/inverse_pipeline.py` добавить `regularization` в `INVERSE_PARAMETERS`. Значение по умолчанию приходит из `--print-defaults` решателя и равно `0`.
- В `validate_options` принимать конечные числа ≥ 0. Ноль отключает штраф.
- В `tests/tools/test_inverse_desktop.py` проверить приём `0` и `0.25`, отказ для `-1`, `nan` и `inf`.
