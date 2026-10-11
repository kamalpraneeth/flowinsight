# FlowInsight: EDA Tool Log Analytics & Prediction Toolkit

> **Problem Statement**
> Chip design teams run thousands of EDA tool jobs (synthesis, place-and-route, verification). Engineers lose time reading long logs to find why a job failed, how long the next job will take, or whether a run looks abnormal. FlowInsight parses tool logs, predicts runtime, classifies failures with ranked likely root causes, detects anomalies, and generates trend reports to help design-ops teams unblock tapeouts faster.

---

## Overview

FlowInsight is an end-to-end analytics and predictive machine learning toolkit purpose-built for semiconductor EDA (Electronic Design Automation) execution logs across Synthesis, Place-and-Route (P&R), and Verification.

### Core Capabilities
- **Log Streaming & Parser Engine**: Streaming parser supporting modern bracketed, hyphenated, and legacy log syntax without loading entire multi-gigabyte logs into memory.
- **Runtime & Peak Memory Regressor**: Gradient-boosted regressor predicting execution duration and peak resident memory with calibrated 80% prediction intervals.
- **Root-Cause Failure Classifier & Priority Ranker**: Multi-class logistic regression and random forest classifying failure mechanisms with $O(C \log k)$ min-heap top-3 ranking and cited log evidence lines.
- **Multi-Method Anomaly Detection**: Dual-technique detection combining unsupervised Isolation Forest with non-parametric Robust Median Absolute Deviation (MAD) for runtime spikes, memory growth, and warning storms.
- **Design Trend Analysis & Reporting**: Automated cross-run regression tracking, timing degradation alerts, and Markdown/HTML report synthesis.
- **REST API & CLI**: Complete FastAPI service and CLI tool for pipeline integration.

---

## Limitations of the Synthetic Scores

The machine learning models developed and evaluated in FlowInsight achieve high performance metrics (e.g., failure classifier macro F1 > 0.90, Top-3 root cause ranking accuracy > 96%). However, the following critical limitations and context must be acknowledged:

1. **Synthetic Signature Reliance**:
   The high classification and ranking scores reflect signature message IDs (`TIME-501`, `SYS-004`, `MEM-902`, `RTE-702`, `CONG-801`, `LIC-101`, `LIB-204`, etc.) and structured log syntax defined by the synthetic log generator. In synthetic environments, tools reliably log identifiable error codes or precursor warnings prior to process termination.

2. **Unvalidated on Real Production EDA Logs**:
   These models have **not been tested on real EDA tool logs** from production semiconductor design environments. Real industrial EDA logs (from tools such as Synopsys Design Compiler/ICC2, Cadence Innovus/Genus, Siemens Questa/Calibre) exhibit substantially messier behavior, including proprietary undocumented return codes, intermingled multi-threaded stdout/stderr output, silent crashes with SIGSEGV/SIGBUS devoid of tool summaries, vendor-encrypted IP block messages, and complex cascading failures.

3. **Origin of the Initial Low Baseline Score**:
   The initial near-chance failure classification score (macro F1 ~0.17) observed during early development was caused by a **pipeline bug**: the training dataset manifest CSV initially lacked the parsed text columns (`error_text` and `message_ids_str`), leaving the NLP vectorizers with empty/dummy inputs while tabular hardware features alone lacked root-cause discriminative power. Once this bug was resolved and log text columns were properly populated, feature vectorization functioned as intended.

4. **Documented Design Choice for Evidence Availability**:
   The synthetic generator's evidence-availability mix—specifically the inclusion of step-level precursor warnings (70% rate), controlled truncation (15% rate), and uninformative ambiguous terminations (5% rate)—was established as an explicit, documented design choice. This benchmark configuration ensures that while models learn realistic correlation between warning signatures and fatal terminations, performance on real-world logs will depend heavily on the logging verbosity and error-handling fidelity of actual EDA tools.

---

## Model Benchmark & Evaluation Summary

Evaluated on 5,000 synthetic runs across 10 semiconductor design templates (85k to 2.35M gates):

| Model / Pipeline | Metric | Score | Baseline / Reference |
|---|---|---|---|
| **Failure Classifier** (Full Pipeline) | Macro F1 | **0.9031** | Dummy Most Frequent: 0.0434, Stratified: 0.0790 |
| **Failure Classifier** (Cross-Validation) | 3-Fold Stratified CV | **0.8931 ± 0.0066** | Low variance across validation folds |
| **Root-Cause Top-3 Ranker** | Top-3 Accuracy | **96.77%** | 270 / 279 true failure causes in Top 3 |
| **Text Feature Ablation Lift** | $\Delta F_1$ (with vs without text) | **+0.7560** | Text-ablated model F1: 0.1471 |
| **Runtime Regressor** | $R^2$ / RMSE | **0.972** / 42.1s | Dummy Mean Baseline $R^2$: 0.00 |
| **Anomaly Detection** (MAD + iForest) | Precision / Recall | **0.912** / **0.884** | Evaluated on injected anomaly ground truth |

---

## Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/eda-tools/flowinsight.git
cd flowinsight

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 1. Generate Synthetic EDA Benchmark Logs
```bash
flowinsight generate --jobs 5000 --seed 42 --output-dir data
```

### 2. Train Models
```bash
flowinsight train --manifest data/manifest.csv --seed 42 --output-dir models_saved
```

### 3. Parse and Classify a Failing Job Log
```bash
flowinsight classify --log data/logs/JOB-SYN-00001.log --model-dir models_saved
```

### 4. Run Test Suite
```bash
pytest
```
