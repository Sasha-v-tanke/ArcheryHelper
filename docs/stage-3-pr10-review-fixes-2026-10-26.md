# PR #10 — исправления ревью от 26.10.2026

Проверенный ревьюером HEAD: `6514665d75a4ef80d3c424dd497aa550fa596dce`.
Исправления внесены в тот же [Draft PR #10](https://github.com/Sasha-v-tanke/ArcheryHelper/pull/10).
Merge и переход к Stage 4 не выполнялись.

| Замечание | Исправление | Проверка |
|---|---|---|
| P1: evaluation обходила выбор пользовательской группы | CLI и matrix runner принимают `--selections`; evaluator использует тот же SelectedTargetGeometryPipeline: candidates → подтверждённый выбор → ROI/foreign masking → geometry/quality → calibration → normalization | В кадре 9 faces обрабатывается подтверждённый средний блок; смена GT не меняет geometry; выбор чужого блока даёт большую ошибку |
| P2: tracking терялся после batch | VISUAL_TRACKING_FROM_USER восстанавливается при совпадении fingerprint; IDs/count/layout/config перепроверяются; ownership и physical face order остаются false | Реальные ORB/RANSAC correspondences на fixture: confirm → track → save → batch; AUTO_CONFIDENT сохранён, ownership не повышен |
| P2: неудачный replay портил предыдущие результаты | Пересчёт выполняется во временном соседнем каталоге; сравнение предшествует изменению исходного output | После MISMATCH, исключения обработки и CLI с невалидным GT весь прежний каталог побайтово совпадает, временный каталог удалён |

Численные пороги не менялись. Draft policy revision 4 требует selected-target pipeline.
Legacy `full-frame` сохранён для development-диагностики, но не проходит текущий final gate.

## Привязка выбора к независимой evaluation

`target-selections.json` — map `original_sha256 → saved selection`, отдельный от GT.
Пример запуска:

```sh
python -m archery_ml.geometry --manifest data/mobile-work/manifests/dataset_manifest.json \
  --split mobile_dev --format TRIPLE --triple-layout VERTICAL \
  --minimum-scoring-zone 6 --face-diameter-mm 200 \
  --selections data/mobile-selection-review/target-selections.json \
  --board-template dataset_tools/boards/vertical-v1.json \
  --policy dataset_tools/stage3_geometry_policy.json --output data/reports/selected-dev.json
```

В manifest используются upright RGB/проверенные EXIF derivatives. Кандидаты обнаруживаются
повторно; stale fingerprint, невалидные IDs/config и неподтверждённые предложения дают
GROUP_SELECTION_REQUIRED. Такие фото остаются в denominator. Чужая группа не выбирается
автоматически вместо неподтверждённой. Подтверждение пользователя само по себе не
доказывает правильность выбора: ошибка проявляется при сравнении с независимым GT.

Per-image `selection` содержит proposals, источник решения, candidate IDs, причины отказа
и результат выбранного pipeline с transforms. `selection_coverage` отдельно считает
подтверждённые/неподтверждённые/невалидные выборы. Provenance фиксирует версию pipeline
и SHA256 ровно тех selections-байтов, которые были загружены. Aggregation отклоняет
смешанные версии или разные selections hashes. Final gate отклоняет legacy pipeline
и READY без подтверждённого выбора. GT-landmarks не передаются в selection/geometry.

## Сохранность replay

При MISMATCH старые inventory, detections, geometry-results, metrics, selections,
run-manifest, previews и crops не изменяются. Диагностика сохраняется рядом как
`<output-name>-replay-mismatch.json`. При исключении старые файлы также остаются целыми.
Board/GT из CLI сначала попадают во временный каталог. При успешной проверке обновляется
только `reproducibility.json`; исходные результаты и run-manifest сохраняются.

## Проверки и данные

- Все Python unit tests: **171 PASS**, без локальных skips.
- Android: **8 PASS**, `testDebugUnitTest assembleDebug --rerun-tasks` успешен;
  выполнены все 46 задач. Runtime, старая модель и Room schema не изменялись.
- Новые регрессии: 6 сценариев в `test_stage3_review_fixes.py` и 2 проверки final gate
  в `test_stage3_validation.py`; CI обнаруживает оба файла через `test_stage3_*.py`.
- Development evaluation проверена на прежних 58 фото с сохранёнными выборами:
  **51 READY**, **6 quality rejects**, **1 GROUP_SELECTION_REQUIRED — №23**.
  57 USER_CONFIRMED, 1 AUTO_CONFIDENT. Предыдущие 52 pilot READY включали preview
  неподтверждённой группы №23; это объясняет различие, а не ухудшение geometry.
- Независимого GT нет. Median/P95 accuracy, false acceptance/rejection не измерялись.
  Suitability остаётся unknown; эти результаты не являются независимой аттестацией.
- Локальные артефакты: `data/reports/stage3-review-fixes/`; исходный pilot не перезаписывался.
- Реальный development run выполнен на чистом `3a89bbf3c07611b9f1106b4df0f1b387ce26ad23`.
  Для всех 57 подтверждённых выборов статусы и homographies совпали с pilot
  (absolute tolerance `1e-9`); SHA256 оригиналов совпали 58/58.
- Python/Android CI реализации прошли для push и PR:
  https://github.com/Sasha-v-tanke/ArcheryHelper/actions/runs/40281245649.
  В CI два ожидаемых external-v9 skips; локально эти проверки прошли.
- Сводка без фотографий: `docs/reports/stage-3-pr10-review-fixes-2026-10-26.json`.

## Коммиты исправлений

| Коммит | Содержание |
|---|---|
| `dd531bb` | Выбранная группа в independent evaluation, CLI, matrix и final gate |
| `24ca11d` | Восстановление tracking без подтверждения ownership |
| `3f11436` | Изолированный replay |
| `fa94661` | Регрессии трёх замечаний и проверка передачи selections через все 18 configs |
| `3a89bbf` | Документация, CLI replay-input regression и защита provenance полей |

Исправления готовы к повторному review инженерной инфраструктуры. Отсутствие новых
размеченных фото само по себе не препятствует её merge после review. Приёмка полного
Stage 3 остаётся **BLOCKED: INDEPENDENT REAL-DATA VALIDATION**: нужны development calibration,
frozen policy и независимый holdout с reviewed GT, физической ориентацией и полным coverage.
