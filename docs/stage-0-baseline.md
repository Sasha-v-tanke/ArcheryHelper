# Stage 0 baseline

## Goal

Establish a reproducible, reviewable baseline before the architecture migration starts. Stage 0 must not change product behavior, persistence semantics, ML contracts, model runtime, or legacy dataset tooling.

The baseline is the comparison point for later PRs: the current Python unit tests, Android regression tests, Android debug build, and Room schema v1 must stay reproducible.

## Scope

Stage 0 changes only build/test infrastructure and Room schema export.

In scope:

- pin the Python baseline used by CI;
- record the Java/Gradle/Android toolchain already used by the project;
- record the current Python, Android unit, Android instrumentation, and build commands;
- enable Room schema export and commit schema v1;
- run a minimal CI baseline on pushes and pull requests;
- keep the existing Android regression tests as explicit migration guards.

Out of scope:

- changing Room schema version or entities;
- adding Room v2 migrations;
- replacing Distance/Series/Shot domain or persistence models;
- changing scoring behavior;
- changing the PyTorch Mobile runtime or model contract;
- deleting or reorganizing legacy ML/data code.

## Toolchain baseline

- Python: 3.11
- Java: 17
- Gradle wrapper: 8.13
- Android Gradle Plugin: 8.11.1
- Kotlin: 2.1.20
- KSP: 2.1.20-2.0.1
- Room: 2.8.5
- compileSdk / targetSdk: 36
- minSdk: 31

Python package versions remain defined in the root `requirements.txt`. Stage 0 does not change them.

## Baseline commands

Run commands from the repository root unless another directory is shown.

### Python unit tests

The current automated Python unit baseline is:

```bash
python -m unittest \
  src.ai.test.test_contracts \
  src.ai.test.test_evaluation \
  src.ai.test.test_predictions \
  src.ai.test.test_splits \
  src.ai.test.test_training_config
```

`src/ai/test/test_output.py` and `src/ai/test/test_ptl.py` are executable/manual diagnostic scripts and currently contain no `unittest.TestCase` tests, so they are not part of the unit baseline.

### Android unit tests

```bash
cd AndroidApp/ArcheryHelper
./gradlew testDebugUnitTest
```

The following existing tests are migration regressions and must remain enabled until their corresponding components are functionally replaced:

- `ShotEditorTest`
- `ScoringEngineTest`
- `PredictionParserTest`

### Android instrumentation tests

```bash
cd AndroidApp/ArcheryHelper
./gradlew connectedDebugAndroidTest
```

This command requires an attached API 31+ device or emulator. The repository currently has no `app/src/androidTest` test sources, so Stage 0 records the command but does not provision an emulator in the minimal CI job.

### Android debug build

```bash
cd AndroidApp/ArcheryHelper
./gradlew assembleDebug
```

## Room v1 baseline

`ArcheryDatabase` remains at schema version 1. Stage 0 only enables schema export.

The committed Room schema is generated into:

```text
AndroidApp/ArcheryHelper/app/schemas/
  com.direwolf.archeryhelper.data.room.ArcheryDatabase/
    1.json
```

CI regenerates Room schemas during the Android build and fails if `app/schemas` becomes dirty. Any future Room schema change must therefore either be accidental and reverted, or intentional and accompanied by the corresponding migration work.

## Minimal CI contract

Every push and pull request must run:

1. Python unit baseline on Python 3.11.
2. Android `testDebugUnitTest` on Java 17.
3. Android `assembleDebug` on Java 17.
4. Room schema cleanliness check after the Android build.

Instrumentation tests remain a documented local/device gate until a later stage introduces instrumentation coverage and emulator CI.

## Legacy freeze for Stage 0

Do not delete, move, or behaviorally modify the following during Stage 0:

- `src/ai`;
- `src/yolo`;
- `src/roboflow`;
- `DataConverter`;
- current Room entities and DAO contracts;
- `TorchShotDetector`, `PredictionParser`, `model.ptl`, and PyTorch Android dependencies.

Later stages may replace these components only after the parity/migration gates defined in the project plan are satisfied.

## Definition of Done

Stage 0 is complete when:

- Python unit baseline passes in CI;
- Android unit tests pass in CI;
- Android debug build passes in CI;
- Python, Java, Gradle, Kotlin, KSP, Room, and Android SDK baselines are recorded;
- Room schema v1 is committed and CI detects uncommitted schema drift;
- the three named Android regression tests remain present and enabled;
- no legacy ML/runtime/data subsystem has been removed or behaviorally migrated.
