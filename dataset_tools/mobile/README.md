# Stage 3: сбор, независимая разметка и физическая калибровка

Статус: инструменты Python reference. Реальная съёмка, независимая человеческая
разметка и итоговая аттестация ещё не выполнены. Синтетические примеры не закрывают Stage 3.
Запускайте команды из корня репозитория с активным Python environment проекта.

## 1. Подготовить щит

OpenCV 4.12 содержит ArUco; дополнительная dependency не нужна.
В `dataset_tools/boards/` три draft-шаблона: SINGLE, VERTICAL, TRIANGULAR.
Система координат щита: mm, x вправо, y вниз; UP означает отрицательное y.
Канонический центр face `(0,0)`, полный номинальный радиус `1.0`, crop `512×512`.
У демонстрационных шаблонов диаметр face 200 mm, щит 600×800 mm.
Для другого размера нужны новая версия шаблона и фактический замер; не растягивайте печать молча.

```sh
python -m archery_ml.geometry.calibration_tools print-board \
  --template dataset_tools/boards/vertical-v1.json \
  --minimum-scoring-zone 6 --ten-ring-mode RECURVE --output /tmp/vertical.svg
```

Печатайте SVG в масштабе 100% без fit-to-page. Физически проверьте размер, центры,
порядок face, стороны 50 mm marker и их положения. Сохраните измерения, независимый
review и новую revision щита. Шаблоны draft не являются доказательством точности печати.
Не поворачивайте отдельные marker glyph относительно их шаблонных координат углов.
Нужно минимум три из четырёх маркеров, распределённых по щиту. Один может быть перекрыт.
Два маркера, неизвестные ID, зеркальное изображение, несогласованные correspondences
или вырожденные матрицы приводят к отказу калибровки. Пределы калибровки в template —
development proposal; проверить и согласовать их по mobile_dev до final freeze.

Калибратор использует реальные marker corners изображения и шаблон; annotations GT
не читает. Он добавляет только ортогональный поворот осей и сопоставление физических
face IDs. Центр, масштаб, projective distortion, fit_error и confidence детектора
не подменяются геометрией щита. Ошибки детектора остаются видимыми в evaluation.

## 2. Зафиксировать план до съёмки

```sh
python -m archery_ml.data.mobile_workflow plan --output data/mobile-plan.json
```

План читает минимумы policy. По текущим минимумам предложит 15 usable и 5 unusable
на конфигурацию: 270 + 90 = 360 оригиналов. Это рекомендация, не обещание READY/PASS.
Для mobile_dev нужен отдельный набор. Не расходуйте final holdout на отладку.
Заранее зарезервируйте дополнительный неиспользованный holdout для новой аттестации
после возможного FAIL и изменения алгоритма.

Вручную заполните реальные sessions до регистрации фотографий, например:

```json
{"capture_session_id":"session-id","device_model":"actual-camera-model","split":"mobile_dev"}
```

Для holdout указывайте `mobile_real_test`. Сессии не пересекаются; распределение устройств
между development/test фиксируется заранее. Укажите revision плана и после проверки
измените status на `approved-before-capture`. Инструмент сохраняет копию и checksum
плана при первой регистрации; последующее изменение блокирует регистрацию/export.

Минимумы берутся из `stage3_geometry_policy.json`: сейчас 200 оригиналов, 6 сессий,
3 устройства; в каждой из 18 конфигураций 10 оригиналов, 2 сессии, 8 READY,
usable acceptance ≥80%. Пока реальных данных нет: **BLOCKED: EXTERNAL CAPTURE REQUIRED**.

Снимайте фронтально, с умеренной и граничной перспективой; меняйте расстояние,
размер в кадре, освещение, тени, блики и ориентацию телефона. Включайте мишени
со стрелами и без стрел, частичные перекрытия. Среди unusable нужны отсутствие
мишени, clipping, неверное число face, сильная перспектива, blur, пересвет,
маленькая мишень и неразличимые кольца. Сохраняйте все исходные оригиналы.

90 отрицательных снимков дают point estimate false acceptance. Для односторонней
95% верхней границы ниже 1% при нуле false accepts нужно не менее 299 независимых
отрицательных кадров: `1 - 0.05**(1/n) < 0.01`. Не выдавайте point estimate за такое подтверждение.

## 3. Зарегистрировать оригиналы и разметить

```sh
python -m archery_ml.data.mobile_workflow register --root data/mobile-v1 \
  --plan data/mobile-plan.json --session session-id --image /path/to/original.jpg \
  --format TRIPLE --triple-layout VERTICAL --minimum-scoring-zone 6 \
  --ten-ring-mode RECURVE --annotator actual-person-id --gt-revision mobile-gt-v1
```

Оригинал копируется побайтово в `originals/`, ID = исходный SHA256. Повторная регистрация
отклоняется. EXIF orientation по умолчанию должен быть 1. Явный `--normalize-exif`
создаёт отдельный PNG в `derivatives/`; оригинал, его SHA и EXIF сохраняются.
`original_sha256` всегда относится к оригиналу, `image_sha256` — к используемому изображению.
Coordinates GT задаются в пространстве используемого изображения, размеры — после EXIF.
Для не-RGB оригинала требуется явный `--normalize-rgb`. Import проверяет не только
оба checksum, но и совпадение pixels производного изображения с заявленным
EXIF/RGB преобразованием оригинала; подменить derivative новым checksum недостаточно.

Редактируйте `drafts/<sha>.json` обычным JSON-редактором. Нужны suitability,
orientation_reference, независимый метод GT и annotations. Draft не проходит importer
как проверенный GT. Для каждого usable face задайте:

- один физический центр `target_center`, image point `[x/width,y/height]`;
- минимум четыре уникальные неколлинеарные пары image/canonical landmarks;
- `face_index`: SINGLE 0; TRIPLE — физический ID из шаблона щита;
- обоснованную `uncertainty_norm` в единицах полного canonical radius;
- `evaluation_landmark_ids`, отдельно от `calibration_landmark_ids`.

Для ввода image points кликами можно создать автономный HTML-редактор:

```sh
python -m archery_ml.data.mobile_workflow annotate --root data/mobile-v1 \
  --record data/mobile-v1/drafts/IMAGE_SHA.json --output /tmp/annotate.html
```

Откройте HTML локально. Введите физический face ID, независимо измеренные canonical
X/Y и uncertainty, затем кликните соответствующую точку изображения. Для центра
выберите «Физический центр». Скачанный draft JSON заменяет только соответствующий
файл drafts/. Фото не изменяется, сеть не используется, review автоматически не ставится.
Для grouped annotations или исправлений полного payload можно редактировать JSON вручную.

Пример одной точки контракта:

```json
{"kind":"landmark","points":[[0.4,0.3]],"face_index":0,
 "canonical_points":[[-0.3,-0.3]],"uncertainty_norm":0.002}
```

Числа выше — иллюстрация формата. Не переносите их в реальные фотографии.
Canonical GT получают физическим измерением, проверенной геометрией печати или
независимой калибровкой. Запрещены predictions проверяемого детектора и оценка accuracy
только по тем же точкам, по которым калибровались оси. Для unusable точные координаты
не обязательны; задайте независимую suitability и причину в review notes.

## 4. Независимый review и immutable export

Второй человек проверяет фото, физические координаты, face IDs, uncertainty и метод GT.
Расхождения устраняются до review. Команду выполняет проверяющий после проверки:

```sh
python -m archery_ml.data.mobile_workflow preview --root data/mobile-v1 \
  --record data/mobile-v1/drafts/IMAGE_SHA.json --output /tmp/annotations.png
python -m archery_ml.data.mobile_workflow review --root data/mobile-v1 \
  --record data/mobile-v1/drafts/IMAGE_SHA.json --reviewer actual-other-person-id \
  --review-notes 'Physical center, independent landmarks, face IDs and uncertainty checked'
python -m archery_ml.data.mobile_workflow export --root data/mobile-v1
```

Разметчик и reviewer должны различаться. Укажите independent_gt_method:
`physical_measurement`, `surveyed_print_template`, `independent_calibration`;
для отрицательных кадров допустим `independent_suitability_review`.
Новый draft budget uncertainty ≤0.00375 задан policy и требует согласования по dev.
Review сохраняется отдельно, не перезаписывается. Export создаёт geometry.json и
sources.json, отказывает при существующем export, смешанных revisions, leakage или
неверных checksum. Исправления после export требуют новой версии набора/GT.
В `audits/` сохраняется receipt review с SHA reviewed record, reviewer, методом GT
и notes. Изменение reviewed record после проверки обнаруживается при export.

```sh
python -m archery_ml.data import --registry data/mobile-v1/sources.json \
  --source mobile_geometry --workspace data/mobile-work --manifest data/mobile-work/manifest.json
python -m archery_ml.data rebuild --registry data/mobile-v1/sources.json \
  --workspace data/mobile-work
python -m archery_ml.data.mobile_workflow coverage --root data/mobile-v1 \
  --output data/mobile-work/coverage.json
```

Coverage раздельный по split и всем конфигурациям; READY остаётся `null` до evaluation.
Для безопасной проверки формата есть синтетический development example:

```sh
python -m dataset_tools.mobile.make_example --output data/mobile-example
python -m archery_ml.data rebuild --registry data/mobile-example/sources.json \
  --workspace data/mobile-example-work
```

Пример явно помечен synthetic, не содержит человеческого review и не может пройти final gate.

## 5. Development, freeze и независимая аттестация

```sh
python -m archery_ml.geometry.calibration_tools calibrate \
  --template dataset_tools/boards/vertical-v1.json --image /path/to/photo.png \
  --minimum-scoring-zone 6 --output /tmp/calibration.json
python -m archery_ml.geometry --manifest data/mobile-work/manifests/dataset_manifest.json \
  --split mobile_dev --format TRIPLE --triple-layout VERTICAL \
  --minimum-scoring-zone 6 --ten-ring-mode RECURVE --face-diameter-mm 200 \
  --board-template dataset_tools/boards/vertical-v1.json --output /tmp/dev-vertical.json
```

Calibration CLI сохраняет record, overlay физических осей и canonical crops.
Без маркера geometry может остаться READY, orientation — UNANCHORED; final gate,
требующий физических осей, выдаст FAIL. Не исключайте такие кадры из denominator.

На mobile_dev исследуйте ошибки/coverage/uncertainty по устройствам, масштабу,
освещению и перспективе; отдельно SINGLE и обе TRIPLE. Сопоставьте geometry budget
со Stage 4 localization tolerance. Пороги автоматически не ослабляются.
После согласования владельцем: отдельный commit frozen policy, versioned измеренный
board template, immutable GT/manifest/snapshot, algorithm SHA, зависимости, чистое дерево.
Сейчас эти действия **BLOCKED: REAL DEVELOPMENT EVIDENCE AND OWNER APPROVAL REQUIRED**.

Final run: для каждого элемента policy выполните evaluation `--split mobile_real_test`
с той же manifest, соответствующими config и board template, затем:

```sh
python -m archery_ml.geometry --aggregate-report /path/to/single-report.json \
  --aggregate-report /path/to/vertical-report.json \
  --aggregate-report /path/to/other-config-report.json \
  --policy dataset_tools/stage3_geometry_policy.json --output /path/to/final.json
```

Нужны все 18 отчётов; сокращённый пример команды не покрывает матрицу сам по себе.
Для полного прохода матрицы есть runner, который не перезаписывает каталог результатов:

```sh
python -m dataset_tools.mobile.validate_all \
  --manifest data/mobile-work/manifests/dataset_manifest.json \
  --snapshot data/mobile-work/snapshots/dataset_snapshot.json \
  --output data/reports/mobile-dev-run-001
```

По умолчанию он оценивает только mobile_dev и сохраняет Stage 3 status BLOCKED,
даже если отдельные development-метрики хорошие. `--final` разрешает holdout
только с frozen policy, чистым algorithm SHA и snapshot, соответствующим manifest.
Для других измеренных board templates используйте `--board-map`: JSON map
`configuration_key → template path`. Runner сохраняет partial FAIL отчётов;
отдельный report одного config не покрывает всю матрицу. Итог — consolidated.json.
Ошибка конфигурации сохраняется в status.json, а не скрывается.
Общие данные и algorithm SHA должны совпадать. Не используйте test GT для inference
калибратора. Вручную проверьте source/GT overlay, axis overlay и crops для всех layout
и сложных условий: центры, радиусы, face order, отсутствие зеркальности и clipping.
Проверьте отказ downstream normalization для rejected geometry.

При FAIL сохраните report неизменным. Отладка — на development данных. После изменения
алгоритма или порогов нужна новая заранее зарезервированная независимая выборка;
не добивайтесь PASS tuning на том же holdout и не удаляйте неудобные фотографии.
Stage 3 закрывается только по реальному независимому PASS, а не по зелёному CI.
