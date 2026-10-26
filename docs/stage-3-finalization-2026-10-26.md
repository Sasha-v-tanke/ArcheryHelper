# Stage 3 — результат проверки 26.10.2026

**Engineering completion: READY FOR REVIEW.** Доступные инструменты интегрированы,
автоматизированы и проверены; качество алгоритма не аттестовано.
**Stage 3 acceptance: BLOCKED: INDEPENDENT REAL-DATA VALIDATION.**
Stage 4 не начат. Merge не выполнялся.

PR: https://github.com/Sasha-v-tanke/ArcheryHelper/pull/10 — Draft.
План: https://app.notion.com/p/3f099ff957ed8108b097f02f508c7ffd.

## GitHub до изменений

| Артефакт | Проверенный результат |
|---|---|
| `origin/main` | `9212f88c5e38eb89b547bed8089d2d94bb3a3bd5` |
| PR #8 | Смёржен в geometry-ветку 22.10; исправления затем вошли в PR #9 |
| PR #9 | Смёржен squash-коммитом `9212f88` 25.10.2026 |
| `feature/geometry-validation` | `3eb2677`; обратный diff в основном удаляет новые policy/provenance/GT safeguards; не переносился |
| `feature/stage3-completion` | `a95b672`; 37 файловых изменений относительно актуального main отсутствовали |
| Новая ветка | `feature/stage3-finalization`, создана от `origin/main`; слепого merge нет |

Сравнение выполнено по файловым diff, публичным API и поведению тестов, с учётом squash.
PR #9 уже содержал равный вес READY-изображений, per-face gate, 18 конфигураций,
проверку GT digest/revision и отказ при незакреплённой policy. Повторно они не переносились.

## Изменения и коммиты

| Коммит | Содержание |
|---|---|
| `cef416b` | Недостающий pilot, candidate/group selection, previews, EXIF inventory, board calibration, независимый mobile workflow/editor и тесты |
| `1016e29` | Отдельно восстановлены 8 удалённых документов/исторических отчётов из Git |
| `a1f3d5e` | Единая проверка offline/interactive selection; защита метрик от дубликатов; audit полного manifest до фильтрации; калибровка pilot; проверяемый replay; 8 регрессий |
| `e050d22` | Раздельная диагностика ошибок центра/масштаба/поворота и остатка affine fit; дополнительная регрессия |

Модули: `geometry/{selection,selected_geometry,pilot,pilot_inventory,pilot_viewer,
pilot_metrics,boundary,calibration,calibration_tools,detector,evaluate,validation,cli,audit}`;
`data/{mobile_workflow,annotation_editor,geometry_provenance,importers/mobile_geometry}`;
`contracts.ImageGeometryAnnotation.landmark_ids`; `dataset_tools/{mobile,boards}`;
новые `test_geometry_*` и `test_stage3_*`. Точный список — Files changed PR.
`landmark_ids` — необязательное обратно совместимое поле для привязки actual GT IDs;
сигнатуры трёх основных geometry API сохранены.

Исходные фото, crops и отчёты pilot находятся только в игнорируемом `data/`.
Android runtime, старая модель, Room schema и ML training не изменялись.
Восстановленные старые документы — исторический контекст; текущий статус указан здесь.

## Реальный pilot: 58 оригиналов

Набор найден локально в `/Users/alex/Downloads/Мишени`. Использован именно он.
Сохранённые пользовательские выборы взяты из существующего `mobile-pilot-58`.
Это `legacy_exploratory/mobile_dev`, уже использованный при разработке, не holdout.

| Показатель | Результат |
|---|---:|
| Прочитано / уникальных / точных дубликатов | 58 / 58 / 0 |
| Найдено candidates | 262 |
| Кадров с >3 candidates | 32 |
| Кадров с несколькими triple-группами | 20 |
| Сохранённых USER_CONFIRMED выборов | 57 |
| Требует подтверждения владельца | 1 — №23, AUTO_CONFIDENT |
| Geometry READY / rejected | 52 / 6 |
| READY с подтверждённым владельцем | 51 |
| Selected canonical crops | 156, по 512×512 |
| Physical orientation | 58 UNANCHORED, 0 ANCHORED |
| Независимо измеренных GT-фото | 0 |
| Устройства | 57 Galaxy S25 Ultra, 1 неизвестное |
| EXIF | 57 orientation=6; №58 orientation=0, явно помечена invalid |

Количество READY и необходимость подтверждения — разные измерения: №23 входит в 52,
но принадлежность мишеней на нём не подтверждена. 57 конфигураций обрабатывались как
TRIPLE VERTICAL, зоны 10–6; №58 — SINGLE, зоны 10–1. Физический диаметр и scoring mode
не выдумываются. Настоящий TRIANGULAR этим набором не покрыт.

| Отказ алгоритма | Кадры | Число |
|---|---|---:|
| GEOMETRY_UNCERTAIN | №4 | 1 |
| TOO_BLURRY | №22, №43, №47 | 3 |
| TARGET_CLIPPED | №34, №58 | 2 |

Это причины quality gate, а не независимо подтверждённая suitability. False rejection
rate по ним вычислить нельзя. Проблемы сгруппированы в `failure-cases.json` и
`problem-ranking.json`. Обработка не содержит условий по именам этих фотографий.

Сложные общие случаи: несколько соседних листов, перекрытия стрелами, разрывы синего
кольца, синий фон, перспектива, размытие, неполные внешние границы. Перенесённый detector
использует concentric evidence и внешний angular envelope, инициализация учитывает
ellipse shear. Перенесённый display mask не меняет transforms или сырые canonical crops.
50 из 156 display-границ на 26 фото имеют BOUNDARY_UNCERTAIN: требуется проверка края.
Полный исходный crop доступен рядом с display preview.

Старые visual ratings сохранены: 25 Correct, 32 Wrong geometry, 1 Uncertain.
**Все 58 относятся к прежней geometry revision и требуют перепроверки.** Они не
засчитаны как подтверждение новых результатов. Просмотр диагностической подборки
не является pixel-level review всех 58 снимков или independent GT.

Найденные чужие faces маскируются в производной копии до нормализации. Ненайденные
чужие faces этим не исключаются: принадлежность подтверждается отдельно. Стрелы
не используются для определения владельца. Неоднозначность сохраняется как AMBIGUOUS
(UNCERTAIN в пользовательском смысле), а не превращается в USER_CONFIRMED.

## Что измерено

| Данные / проверка | Значение | Значение для приёмки |
|---|---|---|
| Реальные 58: original SHA256 после обработки | Совпадают 58/58 | Оригиналы сохранены |
| Реальные 58: original↔canonical round-trip max | `2.2737367544323206e-12 px` | Согласованность матриц; не accuracy |
| Реальные 58: median/P95 2D, scale/center/orientation error | Не измерены: нет independent GT | Не доказаны |
| Реальные 58: false accept/reject | Не измерены: нет независимой suitability | Не доказаны |
| 3 синтетических board-примера, SINGLE/VERTICAL/TRIANGULAR | READY 3/3, ANCHORED 3/3 | Только development smoke |
| Те же 3 примера: median 2D | ≈0.011938 полного радиуса | **Выше draft 0.0075** |
| Те же 3 примера: P95 / max 2D | ≤0.0120473 / 0.0120473 | Ниже 0.015 / 0.035 только на этих fixtures |

Синтетический renderer здесь 600×800, radius=100 px; измеренные ошибки нельзя переносить
на реальные фото. Median failure сохранён, пороги не ослаблялись. Достижимость policy
по этим данным не доказана: требуется development calibration, включая влияние масштаба
и raster/кольцевых границ, затем независимая аттестация. Красивый preview и fit residual
не заменяют независимо измеренные 2D ошибки.

Дополнительная проверка: одинаковый board при radius=100/200/400 px для всех трёх
layouts дал median ≈0.011938/0.005947/0.002938. При 100 px affine scale bias ≈+2.805%.
Это свидетельство зависимости от разрешения в данном renderer, а не универсальная
поправка масштаба. Ошибка не компенсировалась данными GT. Все 9 development-результатов
сохранены в JSON-сводке; реальные требования к разрешению ещё нужно откалибровать.

При наличии GT отчёт раздельно хранит physical-center error, affine scale bias,
orientation error, affine center offset, anisotropy и affine residual. Отдельно —
наблюдаемая ellipse perspective ratio и coverage. Affine residual является диагностикой
нелинейной ошибки, а не прямым измерением перспективы камеры. Affine fit никогда не
исправляет прогноз и не используется для улучшения acceptance-метрик.

## Проверки

| Проверка | Результат |
|---|---|
| Все Python unit tests, `discover -p 'test_*.py'` | 163 PASS, без skips локально |
| Geometry discovery | `test_geometry_*.py`, 33 теста |
| Stage 3 discovery | `test_stage3_*.py`, 87 тестов в текущей версии |
| Новые finalization-регрессии | 9: malformed/stale selection, duplicate GT, leakage, calibration, replay, downstream rejection, error decomposition |
| Android `testDebugUnitTest assembleDebug --rerun-tasks` | 8 тестов PASS; BUILD SUCCESSFUL, 46 задач выполнены |
| CI Python + Android на `e050d22` | Обе push/PR проверки success; CI обнаружил новый finalization-файл |
| CI external-v9 tests | 2 ожидаемых skips: внешнего v9 export в CI нет; локально прошли |
| Synthetic example → rebuild → snapshot → matrix | 3 отчёта, 15 конфигураций явно отсутствуют; BLOCKED, exit 1 |
| `validate_all --final` с draft policy | BLOCKED до holdout evaluation, exit 1 |
| Реальный повторный pilot на `e050d22` | VERIFIED: SHA исходников/входов/результатов и dependencies совпали; durations исключены |

Локальный Python 3.12.7; CI использует `.python-version` 3.11.
NumPy 2.2.6, OpenCV 4.12.0, Pillow 11.3.0. Новых dependencies нет.
Тесты покрывают single/triple, поворот/перспективу/scale/crop, EXIF 1–8, invalid EXIF,
absence/blur/exposure, wrong face count, отказ normalization, несколько листов,
сохранение выбора, provenance tampering, calibration/evaluation separation и leakage.

## Таблица требований Stage 3

| Требование | Инженерная проверка | Реальная приёмка |
|---|---|---|
| TargetGeometryDetector / TargetNormalizer / ImageQualityGate | Реализованы, регрессии проходят | Точность не аттестована |
| SINGLE / VERTICAL / TRIANGULAR | Synthetic tests; физические marker axes и face IDs | TRIANGULAR real отсутствует |
| Candidate detection → group selection → config → geometry | CLI + offline/loopback review + сохранённый выбор | №23 требует подтверждения |
| Wrong face count / rejected quality | Отказ, без downstream READY normalization | Suitability review ещё нужен |
| original↔canonical, EXIF, неизменность originals | Матрицы + hashes + unit/real проверки | Выполнено технически |
| Orientation calibration | Только наблюдаемые markers; GT не передаётся детектору | На 58 фото UNANCHORED |
| Preview / diagnostics / hard-case selection | JSON, HTML, crops, overlays, ranking | Новые visual ratings отсутствуют |
| Independent GT import / review / provenance | Version/hash/reviewer/session/device, отдельные IDs, anti-duplication | Реальных GT нет |
| Equal-image / per-face / per-config metrics | Tests + controlled gate fixtures | Нет реальной матрицы 18 configs |
| Train/dev/holdout isolation | Import safeguards + audit полного supplied manifest | Внешние неуказанные наборы audit не сертифицирует |
| Frozen policy и numerical thresholds | Пороги сохранены, draft fail-closed | Не заморожена; dev median выше draft |
| Python / Android / CI | Проверены; PR готов к review | Не равны accuracy acceptance |
| Stage 4 gate | Переход не выполнялся | BLOCKED |

## Минимальные данные и точная процедура закрытия

1. Собрать отдельный development batch с измеренным щитом/ориентиром, включая обе triple
   layouts, разные масштабы и сложные условия. Измерить компоненты ошибки, uncertainty
   и пригодность. Разобрать median failure; обосновать policy и утвердить её отдельной
   frozen revision после development calibration. Текущие числа не менять ради PASS.
2. До новой съёмки создать `mobile_workflow plan`, назначить реальные session/device/split,
   проверить и зафиксировать `approved-before-capture`. Для текущей draft policy минимум
   holdout — **200 уникальных оригиналов, 6 сессий, 3 устройства, 18 конфигураций**;
   каждая ≥10 оригиналов из ≥2 сессий, ≥8 READY и usable acceptance ≥80%.
   18×10=180, значит нужны ещё ≥20 оригиналов сверх подгрупповых минимумов.
   Это минимум для вычисления policy, не гарантия PASS. Dev и резервный holdout — отдельно.
3. Рекомендуемый capture plan: 15 usable + 5 unusable на конфигурацию = 360 holdout фото.
   Включить реальные отрицательные случаи, все отказы сохранять в denominator. Этот объём
   не доказывает статистическую верхнюю границу false acceptance ≤1%: при нуле ошибок
   для односторонней 95% границы нужны ≥299 независимых отрицательных фото.
4. Физически измерить print template/face centers/radii и marker positions, сохранить
   revision и review. Минимум 3 видимых непротиворечивых ArUco-маркера. Для usable face:
   один измеренный центр и ≥4 уникальных неколлинеарных evaluation landmarks; canonical
   coordinates получать независимо, uncertainty ≤0.00375 по текущей draft policy.
   Calibration/evaluation IDs и точки не смешивать. Оригиналы не редактировать.
5. `register` → локальный `annotate` → review другим человеком → immutable `export` →
   `data rebuild` → manifest/snapshot. Процедура и команды: `dataset_tools/mobile/README.md`.
   Сверить все используемые train/dev/test источники и capture sessions, exact/near hashes.
6. Зафиксировать algorithm commit, измеренный board template, GT revision и policy;
   выполнять `validate_all --final --manifest ... --snapshot ... --board-map ... --output ...`
   в чистом дереве. Runner создаёт 18 отчётов, consolidated.json и status.json. Только
   независимый PASS всех обязательных gates разрешает Stage 4. После tuning по FAIL
   необходима новая заранее отложенная независимая выборка.

## Повторный запуск pilot без изменения Python-кода

```sh
python -m archery_ml.geometry.pilot --input /path/to/originals \
  --output data/reports/my-pilot --selections /path/to/confirmed-selections.json
python -m archery_ml.geometry.pilot --input /path/to/originals \
  --output data/reports/my-pilot --verify-replay
python -m archery_ml.geometry.pilot --input /path/to/originals \
  --output data/reports/my-pilot --serve-only
```

Для измеренного щита добавить `--board-template`; для partial independent GT —
`--independent-gt`. Raw photos, derivatives, JSON и viewer остаются в `data/`.
Локальные результаты этой проверки: `data/reports/stage3-finalization/pilot/`.
Количественная приёмка требует другого, независимого набора, а не повторного pilot.
Сводка без фотографий: `docs/reports/stage-3-finalization-2026-10-26.json`.
CI проверенной реализации: https://github.com/Sasha-v-tanke/ArcheryHelper/actions/runs/40277056918.
