# TEST REPORT

## 1) System Overview

- Project type: PyTorch training pipeline + Streamlit inference app.
- Primary runtime entry points:
  - Training: [train.py](train.py)
  - Inference UI: [app.py](app.py)
- Core model artifact path used by train/inference:
  - Save: [train.py](train.py#L326-L332)
  - Load: [app.py](app.py#L61-L92)
- Model bundle schema used in code:
  - models, class_names, metrics

## 2) Issues Found

### Critical (found during audit/stress)

- Insecure/unsafe artifact load and missing corruption guard in inference load path (pre-fix): [app.py](app.py#L61-L67)
- Unsafe archive extraction behavior in training setup (pre-fix): [train.py](train.py#L25-L31), [train.py](train.py#L47-L51)
- Train/val transform leakage risk from shared dataset split object (pre-fix): [train.py](train.py#L88-L92)

### Major (found during audit/stress)

- No held-out test metrics in final reported model metrics (pre-fix): [train.py](train.py#L313-L320)
- Non-deterministic split behavior (pre-fix): [train.py](train.py#L102)
- Invalid/non-image upload crash path (pre-fix): [app.py](app.py#L121), [app.py](app.py#L128-L131)
- Bundle schema assumptions without validation (pre-fix): [app.py](app.py#L87-L89)
- Unknown model name path without explicit error (pre-fix): [app.py](app.py#L30-L58)

### Minor (found during audit)

- Dead imports and redundant logic:
  - [app.py](app.py) (unused imports removed)
  - [train.py](train.py) (unused imports removed)
- Legacy unused templates present (removed): [templates/index.html](templates/index.html), [templates/result.html](templates/result.html)

## 3) Tests Created

- Test scaffolding and dependency stubs:
  - [tests/conftest.py](tests/conftest.py)
- Unit + ML/inference tests:
  - [tests/test_app_unit_ml.py](tests/test_app_unit_ml.py)
- Unit + integration/training tests:
  - [tests/test_train_unit_integration.py](tests/test_train_unit_integration.py)
- Stress harness:
  - [tests/phase4_stress_runner.py](tests/phase4_stress_runner.py)

Coverage intent implemented:
- Unit tests for model factory, preprocessing, bundle load behavior.
- Integration tests for training orchestration and end-to-end inference path.
- ML checks for prediction validity, output shape/type, probability integrity.
- Edge tests for missing files, corrupted artifacts, invalid/unsupported uploads, wrong schema, null metadata.

## 4) Stress Results

Latest validation loop results (Phase 6):

- Test suite:
  - Command: python -m pytest -q
  - Result: 17 passed, 0 failed
- Stress suite:
  - Command: python tests/phase4_stress_runner.py
  - Result: total 13, passed 13, failed 0

Scenario evidence (latest):
- System stress: large image processing, repeated inference calls, rapid UI interactions, csv/video upload handling -> all ok
- ML stress: batch processing, cpu/gpu switching, missing model file, corrupted model file handling -> all ok
- Data stress: wrong schema handling, null metadata handling, large dataset loader behavior -> all ok

## 5) Fixes Applied

### Inference/runtime hardening

- Added unknown-model guard in model factory: [app.py](app.py#L56)
- Added bundle load try/except + schema validation: [app.py](app.py#L66-L92)
- Added invalid upload handling before inference flow: [app.py](app.py#L124-L131)
- Added EDA class metadata guard: [app.py](app.py#L247-L249)
- Device strategy aligned to cuda-if-available: [app.py](app.py#L25)

### Training/ML pipeline correctness

- Added safe zip extraction guard: [train.py](train.py#L25-L31)
- Added download timeout: [train.py](train.py#L47)
- Added dataset schema validation before ImageFolder use: [train.py](train.py#L72-L77)
- Replaced split logic with deterministic seeded index split and separate datasets/transforms for train/val/test: [train.py](train.py#L88-L132)
- Added held-out test metric evaluation for final reported metrics: [train.py](train.py#L175-L205), [train.py](train.py#L313-L320)
- Added graceful loader preparation failure handling in main: [train.py](train.py#L293-L297)

### Test updates for fixed behavior

- Updated app tests to validated error handling behavior: [tests/test_app_unit_ml.py](tests/test_app_unit_ml.py)
- Updated train tests for train/val/test signature and evaluation flow: [tests/test_train_unit_integration.py](tests/test_train_unit_integration.py)
- Updated stress harness to reflect guarded outcomes and stable runtime profile: [tests/phase4_stress_runner.py](tests/phase4_stress_runner.py)

## 6) Cleanup Done

- Removed unused legacy template files:
  - [templates/index.html](templates/index.html)
  - [templates/result.html](templates/result.html)
- Removed dead imports and redundant code paths in active modules:
  - [app.py](app.py)
  - [train.py](train.py)
- Removed generated cache artifacts during cleanup cycle (recreated transiently during subsequent runs as expected by Python tooling).

## 7) Final Stability

- Regression status: stable
- Current automated validation status:
  - Unit/integration/ML tests: pass
  - Stress scenarios: pass
- Remaining known blocking failures: none in latest validation loop

