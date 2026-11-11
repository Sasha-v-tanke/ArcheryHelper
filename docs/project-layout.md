# Project layout

The repository is split by runtime responsibility rather than by implementation language.

```text
app/
dataset_tools/
neural_network/
model_converter/
data/                # local data, ignored
models/              # local model artifacts, ignored
docs/
path_manager.py
requirements.txt
```

## app

Android application and its complete Gradle project.

```text
app/
  app/
  gradle/
  build.gradle.kts
  settings.gradle.kts
  gradlew
```

The application owns Android UI, CameraX, Room persistence, scoring integration, and the on-device model runtime.

## dataset_tools

Tools used to obtain, normalize, annotate, and convert training data.

```text
dataset_tools/
  data_converter/
  prepare_new_dataset/
  roboflow/
  yolo/
```

These are legacy dataset utilities preserved for the baseline. The future unified dataset pipeline can replace them without mixing dataset preparation with model training.

## neural_network

Python model code, training, evaluation, inference helpers, contracts, and tests.

```text
neural_network/
  configs/
  tests/
  model.py
  train.py
  evaluation.py
  predictions.py
  model_metadata.py
```

Manual model visualization lives in `visualize.py`; automated tests live in `tests/`. This avoids the previous `test.py` / `test/` module-name collision.

## model_converter

Conversion/export boundary between the trained Python model and the Android runtime.

Today it contains the legacy PyTorch Lite exporter:

```text
model_converter/
  convert_pth.py
```

When ExecuTorch replaces PyTorch Mobile, the export implementation belongs here. Training logic stays in `neural_network/`, and Android runtime code stays in `app/`.

## Dependency direction

```text
dataset_tools  ──────> data/
                         |
                         v
                  neural_network
                         |
                         v
                  model_converter
                         |
                         v
                       app
```

The directories describe ownership, not a runtime call chain. In particular:

- `app/` does not depend on Python source code;
- `model_converter/` may import `neural_network/`;
- `neural_network/` must not depend on Android code;
- `dataset_tools/` must not depend on Android code;
- generated datasets and model artifacts remain outside source directories.

## Baseline commands

Python unit baseline:

```bash
for test_file in \
  test_contracts.py \
  test_evaluation.py \
  test_predictions.py \
  test_splits.py \
  test_training_config.py
do
  python -m unittest discover -s neural_network/tests -p "$test_file"
done
```

Android unit tests and debug build:

```bash
cd app
./gradlew testDebugUnitTest assembleDebug
```

Android instrumentation command:

```bash
cd app
./gradlew connectedDebugAndroidTest
```
