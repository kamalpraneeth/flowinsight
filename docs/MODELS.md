# Machine Learning Models Specification and Methodologies

> **NOTICE: SYNTHETIC DATA GENERATION**
> Synthetic data generated for demonstration. Not real EDA tool logs.
> All models described herein are fitted and evaluated strictly on synthetic semiconductor tool telemetry.
> Never claim these models work on genuine EDA logs without domain-specific re-benchmarking.

---

## 1. Runtime and Memory Regression Modeling

### Problem Formulation
Predict total wallclock duration (`runtime_s`) and peak RAM consumption (`peak_memory_mb`) from design characteristics (gate count, routing layers, clock period) and tool run configurations (thread count, effort level, optimization target).

### Train / Test Split Strategy
- **Temporal Splitting**: Rather than random shuffling, jobs are sorted chronologically by `start_time`. The earliest 80% forms the training set and the latest 20% forms the evaluation test set.
- **Rationale**: Eliminates forward-looking data leakage, reflecting real EDA production environments where future chip projects run on historical logs.

### Model Architecture and Baselines
1. **Baselines**:
   - `DummyMedianRegressor`: Predicts the training set median.
   - `LinearRegression`: Ordinary least squares on standard-scaled numeric features.
2. **Main Model**:
   - `GradientBoostingRegressor` (100 trees, learning rate 0.08, max depth 4).
   - Alternatively: `RandomForestRegressor` (80 estimators, max depth 8).

### Prediction Intervals and Empirical Coverage
- In physical chip design, engineers need conservative planning margins rather than single point estimates.
- FlowInsight trains paired quantile regressors ($\alpha = 0.10$ for the lower bound and $\alpha = 0.90$ for the upper bound).
- **Nominal Coverage**: $80\%$.
- **Empirical Coverage**: Measured dynamically on the held-out test split:
  $$\text{Coverage} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}\left(y_i^{\text{lower}} \le y_i \le y_i^{\text{upper}}\right)$$

---

## 2. Failure Classification and Root-Cause Ranking

### Problem Formulation
Multi-class classification over 7 root-cause categories:
`out_of_memory`, `timeout`, `missing_library`, `constraint_violation`, `license_error`, `syntax_error`, `congestion`.

### Class Imbalance and Validation
- Class-balanced weighting (`class_weight='balanced'`) inversely scales loss by class frequency:
  $$w_c = \frac{N}{K \cdot N_c}$$
- Stratified 3-fold cross validation on training data.

### Root-Cause Ranker Algorithm & Complexity
When an EDA tool aborts, the system outputs the top $k=3$ most probable causes along with the specific lines in the raw log that substantiate each cause.

```python
# Complexity: O(C * log(k))
heap = []
for cause, prob in zip(classes, probabilities):
    if len(heap) < k:
        heapq.heappush(heap, (prob, cause))
    elif prob > heap[0][0]:
        heapq.heapreplace(heap, (prob, cause))
```
- For $C$ total classes ($C=7$) and $k=3$, maintaining a min-heap operates in $\mathcal{O}(C \log k)$ time, asymptotically superior to full sorting $\mathcal{O}(C \log C)$.

### Feature Ablation Experiment
To demonstrate that textual log error lines convey discriminative signal beyond tabular gate counts:
- **Baseline Feature Set**: Numeric attributes only (gate count, threads, step wallclock).
- **Full Feature Set**: Numeric + Message-ID Bag of Words + TF-IDF n-grams of error lines.
- **Ablation Result**: Evaluated and recorded dynamically in metadata JSON.

---

## 3. Anomaly Detection: Isolation Forest vs. Robust MAD

### Problem Formulation
Unsupervised identification of pathological runs (unexpected runtime lockups, excessive memory leaks, DRC warning floods) without relying on ground-truth training labels.

### Methods Compared
1. **Isolation Forest**:
   - Tree-based multi-attribute anomaly isolation using random axis partitions.
   - Contamination parameter calibrated to ~8%.
2. **Statistical Robust MAD (Per Design-Size Bucket)**:
   - Partitions designs into gate count quartiles (tiers 1-4).
   - In each bucket, computes Median Absolute Deviation (MAD):
     $$\text{MAD} = \text{median}\left(|x_i - \text{median}(x)|\right) \times 1.4826$$
   - Calculates robust z-scores:
     $$z_{\text{MAD}} = \frac{|x_i - \text{median}(x)|}{\text{MAD}}$$
   - Flags anomaly if $z_{\text{MAD}} > 2.8$ across runtime, memory, or warning counts.

### Ground-Truth Evaluation
Both unsupervised detectors are evaluated on synthetic injected ground truth (`is_anomaly` column) reporting:
- Precision, Recall, F1 score
- False Positive (FP) and False Negative (FN) sample logs
