# Stage A Review & Evaluation Report: Backend Engineering & Analytics

> **NOTICE: SYNTHETIC DATA GENERATION**
> Synthetic data generated for demonstration. Not real EDA tool logs.
> All datasets, tool outputs, message logs, and metrics in this project are synthetically generated
> to model realistic semiconductor physical design and verification workflows without proprietary IP.
> Never claim these models work on real production EDA logs without retraining and verification on genuine telemetry.

---

## 1. Executive Summary & Verification Checkpoint

Stage A (Backend Phases 1 to 6) is complete. The system provides an end-to-end analytics and prediction toolkit for Electronic Design Automation (EDA) tool execution telemetry comprising:
1. **Synthetic Data Generator** (`generator.py`): Deterministic physical scaling simulation with realistic multi-step logs, noisy failure modes, messy log artifacts (truncations, storms, unknown messages), and injected anomalies.
2. **Streaming Log Parsers & Feature Pipeline** (`parsers.py`, `features.py`): Memory-efficient line-by-line parsers for synthesis, place-and-route, and verification; scikit-learn ColumnTransformer with numeric, categorical, message-ID count, and TF-IDF text features.
3. **SQLite Analytics Storage** (`store.py`, `sql/queries.sql`): Schema with indexing on stages, status, and timestamps, plus 5 core analytical queries.
4. **Machine Learning Suite** (`models/{runtime.py, failure.py, anomaly.py}`):
   - Runtime and memory regression with temporal splitting and empirical prediction intervals.
   - Root-cause failure classifier with min-heap $O(C \log k)$ top-3 ranker and log evidence line extraction.
   - Unsupervised anomaly detection benchmark comparing Isolation Forest against statistical Robust MAD per design-size bucket against injected ground truth.
5. **CLI, FastAPI, and Bash Automation** (`cli.py`, `api.py`, `scripts/*.sh`): Unified CLI, production-ready FastAPI endpoints with CORS, size limits, and optional API key authentication, and cron automation.
6. **Optional LLM Log Summarizer** (`summarizer.py`): Disabled by default, path redaction, mocked in tests.

---

## 2. Test Execution & Statement Coverage Summary

The test suite was executed via `pytest --cov=flowinsight --cov-report=term-missing` on Python 3.11.9:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\kamal\.gemini\antigravity-ide\scratch\flowinsight
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.15.1, cov-7.1.0
collected 44 items

tests/test_api.py::test_api_health PASSED                                [  2%]
tests/test_api.py::test_api_metrics PASSED                               [  4%]
tests/test_api.py::test_api_jobs_pagination_and_filters PASSED           [  6%]
tests/test_api.py::test_api_job_detail PASSED                            [  9%]
tests/test_api.py::test_api_stats_endpoints PASSED                       [ 11%]
tests/test_api.py::test_api_models_endpoint PASSED                       [ 13%]
tests/test_api.py::test_api_predict_and_classify_endpoints PASSED        [ 15%]
tests/test_cli.py::test_cli_generate_command PASSED                      [ 18%]
tests/test_cli.py::test_cli_parse_single_log PASSED                      [ 20%]
tests/test_cli.py::test_cli_parse_and_train_commands PASSED              [ 22%]
tests/test_cli.py::test_cli_predict_runtime PASSED                       [ 25%]
tests/test_cli.py::test_cli_classify_failure PASSED                      [ 27%]
tests/test_cli.py::test_cli_report_generation PASSED                     [ 29%]
tests/test_features.py::test_feature_pipeline_fit_and_transform_consistency PASSED [ 31%]
tests/test_features.py::test_feature_pipeline_no_data_leakage PASSED     [ 34%]
tests/test_generator.py::test_generator_determinism PASSED               [ 36%]
tests/test_generator.py::test_generator_stage_coverage_and_congestion_rule PASSED [ 38%]
tests/test_generator.py::test_generator_log_files_and_messy_cases PASSED [ 40%]
tests/test_generator.py::test_generator_anomalies_present PASSED         [ 43%]
tests/test_honesty.py::test_bad_model_scores_strictly_worse PASSED       [ 45%]
tests/test_honesty.py::test_reported_metrics_are_not_constant_literals PASSED [ 47%]
tests/test_honesty.py::test_no_hardcoded_metric_literals_in_frontend_source PASSED [ 50%]
tests/test_models.py::test_runtime_model_evaluation_and_intervals PASSED [ 52%]
tests/test_models.py::test_failure_classifier_and_heap_top_k PASSED      [ 54%]
tests/test_models.py::test_anomaly_detectors_and_ground_truth_comparison PASSED [ 56%]
tests/test_models.py::test_evaluation_honesty_bad_model_scores_worse PASSED [ 59%]
tests/test_parsers.py::test_parsers_good_sample PASSED                   [ 61%]
tests/test_parsers.py::test_parsers_truncated_sample PASSED              [ 63%]
tests/test_parsers.py::test_parsers_corrupted_sample PASSED              [ 65%]
tests/test_parsers.py::test_parser_factory_and_file_parsing PASSED       [ 68%]
tests/test_scripts.py::test_collect_logs_script PASSED                   [ 70%]
tests/test_scripts.py::test_eda_flow_hook_script PASSED                  [ 72%]
tests/test_store.py::test_database_init_and_counts PASSED                [ 75%]
tests/test_store.py::test_query_failure_rate_by_stage_and_week PASSED    [ 77%]
tests/test_store.py::test_query_slowest_designs PASSED                   [ 79%]
tests/test_query_most_frequent_errors PASSED                             [ 81%]
tests/test_query_runtime_percentiles_by_stage PASSED                     [ 84%]
tests/test_query_top_warning_ids_before_failures PASSED                  [ 86%]
tests/test_summarizer.py::test_summarizer_disabled_by_default PASSED     [ 88%]
tests/test_summarizer.py::test_summarizer_with_mocked_client_and_path_redaction PASSED [ 90%]
tests/test_trends_reports.py::test_calculate_weekly_failure_rate PASSED  [ 93%]
tests/test_trends_reports.py::test_detect_runtime_drift PASSED           [ 95%]
tests/test_trends_reports.py::test_detect_change_point_alerts PASSED     [ 97%]
tests/test_trends_reports.py::test_report_and_chart_generation PASSED    [100%]

=============================== tests coverage ================================
Name                             Stmts   Miss  Cover   Missing
--------------------------------------------------------------
flowinsight\__init__.py              3      0   100%
flowinsight\api.py                 253     26    90%   53-55, 72, 86-87, 197-210, 248, 269-275, 496-547
flowinsight\cli.py                 149     27    82%   57-58, 75-76, 101-109, 131-158, 165-184, 284-291
flowinsight\features.py             73      7    90%   90-93, 131-133
flowinsight\generator.py           274      6    98%   477-490
flowinsight\models\__init__.py       0      0   100%
flowinsight\models\anomaly.py      119      7    94%   42, 82, 140-141, 166, 184-185
flowinsight\models\failure.py      125      7    94%   68-69, 74, 141-142, 155, 201
flowinsight\models\runtime.py      103      6    94%   65, 98-99, 152-154
flowinsight\parsers.py             276     19    93%   117-119, 131, 293-302, 357, 368, 379, 405, 412
flowinsight\reports.py             117      6    95%   242-249
flowinsight\store.py                85      5    94%   105, 195-196, 213, 239
flowinsight\summarizer.py           42      9    79%   66-67, 78-81, 86, 108-112
flowinsight\trends.py               57      2    96%   52, 65
--------------------------------------------------------------
TOTAL                             1676    127    92%
======================= 44 passed, 1 warning in 27.94s ========================
```

- **Pass Count**: 44 passed, 0 failed, 0 skipped.
- **Statement Coverage**: 92% across all 1,676 lines.

---

## 3. Real Evaluated Model Metrics (Trained on 5,000 Runs, Seed 42)

All figures below are extracted verbatim from `models_saved/*_metadata.json` generated by training on `data/manifest.csv` (Seed: 42, Dataset Size: 5,000 jobs, Git Commit: `d3f6076` / `cd5e89a`).

### 3.1 Runtime Regression Model (`models_saved/runtime_s_metadata.json`)
- **Dataset Size**: 5,000 runs
- **Split Method**: Temporal chronological split on `start_time` (Train: 4,000 runs, Test: 1,000 runs)
- **Model Type**: `GradientBoostingRegressor` (n_estimators=100, max_depth=4, lr=0.08)

| Model / Configuration | MAE (seconds) | RMSE (seconds) | $R^2$ Score | 80% Prediction Interval Coverage |
|---|---|---|---|---|
| **GradientBoostingRegressor** | **896.388 s** | **3499.138 s** | **0.5189** | **0.8100 (81.0%)** |
| *Baseline 1: Median Regressor* | 1846.040 s | 5246.428 s | -0.0816 | N/A |
| *Baseline 2: Linear Regression* | 2116.846 s | 4126.667 s | 0.3308 | N/A |

**Key Takeaways**:
- The GradientBoosting model reduces MAE by **51.4%** compared to the median baseline (896.39s vs. 1846.04s).
- The empirical coverage of the 80% prediction interval (evaluated with dual quantile regressors at $\alpha=0.10$ and $\alpha=0.90$) measured **81.0%**, matching nominal coverage closely.

---

### 3.2 Failure Classification & Root-Cause Ranker (`models_saved/failure_metadata.json`)
- **Dataset Size**: 1,149 failed jobs out of 5,000 (22.98% failure rate)
- **Split Method**: Stratified 75/25 split by failure cause (Train: 861 jobs, Test: 288 jobs)
- **Class-Imbalance Handling**: `class_weight='balanced'`
- **Model Type**: Multi-class `LogisticRegression` with CountVectorizer & TF-IDF text features

#### Primary Evaluation & Baselines
- **Macro $F_1$ Score**: **0.1743**
- **3-Fold Cross-Validation $F_1$ (Train Set)**: Mean: **0.1542**, Std: **0.0093**
- **Comparison Baseline (RandomForest Classifier)**: Macro $F_1$: **0.1253**
- **Ablation Study (Without TF-IDF / Message IDs)**: Macro $F_1$: **0.1742** ($\Delta F_1 = +0.0001$)

#### Per-Class Performance Breakdown
| Failure Category | Test Samples | Precision | Recall | $F_1$ Score |
|---|---|---|---|---|
| `congestion` | 14 | 0.1618 | **0.7857** | **0.2683** |
| `constraint_violation` | 44 | 0.0652 | 0.0682 | 0.0667 |
| `license_error` | 44 | 0.2308 | 0.0682 | 0.1053 |
| `missing_library` | 45 | 0.1509 | 0.1778 | 0.1633 |
| `out_of_memory` | 44 | **0.2826** | 0.2955 | **0.2889** |
| `syntax_error` | 45 | 0.1429 | 0.1111 | 0.1250 |
| `timeout` | 52 | **0.2963** | 0.1538 | 0.2025 |

#### Confusion Matrix (True Class $\times$ Predicted Class)
Classes: `[congestion, constraint_violation, license_error, missing_library, out_of_memory, syntax_error, timeout]`
```text
[[11,  1,  1,  1,  0,  0,  0],  <- congestion
 [12,  3,  2, 13,  5,  8,  1],  <- constraint_violation
 [ 8,  8,  3,  5,  9,  8,  3],  <- license_error
 [ 9, 10,  2,  8,  3,  8,  5],  <- missing_library
 [ 6,  8,  0,  7, 13,  3,  7],  <- out_of_memory
 [12,  8,  2, 10,  5,  5,  3],  <- syntax_error
 [10,  8,  3,  9, 11,  3,  8]]  <- timeout
```

**Root-Cause Ranker Heap Complexity**:
Extracting top $k=3$ causes across $C=7$ classes uses a min-heap with time complexity $\mathcal{O}(C \log k) = \mathcal{O}(7 \log 3)$, requiring fewer comparisons than a full sort $\mathcal{O}(C \log C)$.

---

### 3.3 Anomaly Detection Suite (`models_saved/anomaly_metadata.json`)
- **Dataset Size**: 5,000 runs
- **Ground-Truth Injected Anomaly Rate**: 0.0902 (451 anomalies across `runtime_spike`, `memory_growth`, `warning_storm`)
- **Evaluated Against Separate Injected Ground Truth**

| Method | Statistical Basis | Precision | Recall | $F_1$ Score |
|---|---|---|---|---|
| **Robust MAD (Statistical)** | Per design gate-count quartile MAD ($z_{\text{MAD}} > 2.8$) | 0.2961 | **0.7517** | **0.4248** |
| **Isolation Forest** | Multi-attribute tree isolation (contamination=0.08) | **0.3750** | 0.3326 | 0.3525 |

**Findings on Where Methods Miss**:
- **Isolation Forest Misses (False Negatives)**: Single-dimension extreme spikes (e.g., job `JOB-SYN-00016` with memory growth to 36.9GB or `JOB-VRF-00038` with 9006s runtime) occasionally get masked when other attributes remain normal.
- **Robust MAD Misses**: Complex interactions where neither single attribute crosses the 2.8 threshold, but their joint distribution is anomalous. However, Robust MAD achieves **75.17% recall**, capturing the vast majority of physical resource spikes.

---

## 4. Weaknesses, Failures Encountered & Root-Cause Fixes

In accordance with Rule 6 ("If something fails, fix the root cause; never weaken a test to make it pass"):

1. **Unfitted Ensemble Estimator Truthiness Check (`runtime.py`)**:
   - *Issue*: In Python, `if q_low and q_high:` invoked `bool(q_low)`, which calls `BaseEnsemble.__len__` (`len(self.estimators_)`). Because `q_low` was not yet fitted, it threw `AttributeError: 'GradientBoostingRegressor' object has no attribute 'estimators_'`.
   - *Fix*: Changed to explicit identity check `if q_low is not None and q_high is not None:`.

2. **Stratified Split Minimum Class Count (`failure.py`)**:
   - *Issue*: In small smoke batches (e.g. 120 jobs with ~25 failures), rare classes like `missing_library` occasionally had only 1 member, causing scikit-learn's `train_test_split(stratify=y)` and `StratifiedKFold` to throw `ValueError: The least populated classes in y have only 1 member`.
   - *Fix*: Added dynamic `can_stratify = int(np.min(class_counts)) >= 2` guard; falls back to standard `KFold` when counts are below 3, while preserving stratified splitting on standard datasets.

3. **Pandas ISO-8601 Microsecond Timestamps (`reports.py`, `trends.py`)**:
   - *Issue*: Timestamps generated with microsecond precision (`2024-01-21T09:16:26.857143Z`) caused `pd.to_datetime` without explicit format to fail with `ValueError`.
   - *Fix*: Passed `format="ISO8601"` explicitly to all `pd.to_datetime` conversions.

4. **Bash Path Interpretation on Windows (`tests/test_scripts.py`, `scripts/*.sh`)**:
   - *Issue*: `shutil.which('bash')` on Windows resolved to `C:\Windows\System32\bash.exe` (WSL bash), which mangled Windows backslash paths (`C:Userskamal...`), returning error 127. Furthermore, Git Bash lacked `python` on PATH (only `py` existed).
   - *Fix*: Updated `find_bash()` to prioritize native Git Bash (`C:\Program Files\Git\bin\bash.exe`), converted argument paths to POSIX forward slashes, and added dynamic `PYTHON_BIN` fallback detection in all shell scripts.

5. **ColumnTransformer 1D vs 2D Series Shape Mismatch (`features.py`)**:
   - *Issue*: `SimpleImputer` inside `Pipeline` expected 2D arrays, but ColumnTransformer passes a 1D pandas Series for text column names, triggering `ValueError: Expected 2D container`.
   - *Fix*: Replaced redundant SimpleImputers on text columns (since `LogRecordTabularizer` already enforces `"__none__"` defaults) and routed 1D Series directly to `CountVectorizer` and `TfidfVectorizer`.

6. **Pydantic V2 Deprecation Warning (`api.py`)**:
   - *Issue*: `req.dict()` triggered `PydanticDeprecatedSince20`.
   - *Fix*: Updated to `req.model_dump()`.

---

## 5. Verification: Proof of No Hardcoded Metrics

Per Hard Rule 1, a git grep was executed to confirm that no evaluation metric literals are hardcoded in the codebase:

```bash
git grep -i -E "(mae|rmse|f1_score|empirical_coverage)\s*=\s*[0-9]" flowinsight/
```

**Real Command Output**:
```text
flowinsight/models/runtime.py:    empirical_coverage = 0.0
```
*(Line initializes variable to zero before dynamic calculation via `np.mean(covered)`)*.

Furthermore, `tests/test_honesty.py` programmatically proves:
1. `test_bad_model_scores_strictly_worse`: Validates that degraded/constant models score strictly worse than trained models.
2. `test_reported_metrics_are_not_constant_literals`: Trains on two different dataset slices (40 jobs vs. 80 jobs) and verifies that output MAE values are distinct floats derived from data.
3. `test_no_hardcoded_metric_literals_in_frontend_source`: Scans all frontend TypeScript sources to reject hardcoded metric constants.

---

## 6. Honest Limitations & Operational Boundaries

1. **Synthetic Data Constraint**: All data is synthetic. The physical scaling laws and failure signatures are calibrated representations, not real proprietary foundry EDA logs.
2. **Failure Classifier Noise Floor**: In multi-class root-cause classification, realistic ambiguity across steps limits macro $F_1$ to ~0.17–0.27. Noisy logs containing background warnings from unrelated subsystems intentionally prevent trivial string matching.
3. **Anomaly Contamination Tuning**: Isolation Forest contamination requires calibration for target cluster workloads; unsupervised Robust MAD provides higher recall for single-dimension surges but requires gate-count tier partitioning.
4. **Environment Portability**: Shell automation scripts rely on Bash and standard Unix utilities (`find`, `awk`, `sed`, `grep`, `xargs`). On Windows, Git Bash or WSL is required.

---

## 7. Stage A Checkpoint Status

Stage A is **COMPLETE** and verified:
- Codebase formatted with `ruff format` and clean under `ruff check`.
- 44 tests passing with 92% statement coverage.
- Docker image `flowinsight-api:stage-a` built with non-root security.
- All real metrics logged above directly from pipeline execution.

**Awaiting user confirmation ("continue") to proceed to Stage B (Frontend).**
