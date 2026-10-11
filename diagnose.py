"""Diagnostic script for Failure Classifier investigation (Steps 1 to 5).
DOES NOT MODIFY ANY SOURCE CODE.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from flowinsight.features import build_feature_pipeline
from flowinsight.generator import SyntheticDataGenerator
from flowinsight.parsers import parse_log_file


def run_diagnostics():
    manifest_path = "data/manifest.csv"
    if not os.path.exists(manifest_path):
        print("Manifest not found! Generating small 200 batch...")
        gen = SyntheticDataGenerator(seed=42)
        gen.generate_dataset(num_jobs=200, output_dir="data")

    df = pd.read_csv(manifest_path)
    failed_df = df[df["status"] == "failed"].copy()
    y = failed_df["failure_cause"].to_numpy()
    classes = sorted(np.unique(y))

    print("=" * 80)
    print("STEP 1: DUMMY CLASSIFIER BASELINES")
    print("=" * 80)
    train_df, test_df, y_train, y_test = train_test_split(
        failed_df, y, test_size=0.25, random_state=42, stratify=y
    )

    # Dummy: most_frequent
    dummy_mf = DummyClassifier(strategy="most_frequent")
    dummy_mf.fit(train_df, y_train)
    pred_mf = dummy_mf.predict(test_df)
    f1_mf = f1_score(y_test, pred_mf, average="macro", zero_division=0)
    print(f"DummyClassifier (strategy='most_frequent') Macro F1: {f1_mf:.4f}")

    # Dummy: stratified
    dummy_strat = DummyClassifier(strategy="stratified", random_state=42)
    dummy_strat.fit(train_df, y_train)
    pred_strat = dummy_strat.predict(test_df)
    f1_strat = f1_score(y_test, pred_strat, average="macro", zero_division=0)
    print(f"DummyClassifier (strategy='stratified')    Macro F1: {f1_strat:.4f}")

    # Dummy: uniform
    dummy_uni = DummyClassifier(strategy="uniform", random_state=42)
    dummy_uni.fit(train_df, y_train)
    pred_uni = dummy_uni.predict(test_df)
    f1_uni = f1_score(y_test, pred_uni, average="macro", zero_division=0)
    print(f"DummyClassifier (strategy='uniform')       Macro F1: {f1_uni:.4f}")
    print(f"Expected random chance across 7 classes: 1/7 = {1.0/7.0:.4f}")

    print("\n" + "=" * 80)
    print("STEP 2: LABEL ALIGNMENT CHECK")
    print("=" * 80)
    # Check 5 failed jobs
    sample_failed = failed_df.head(5)
    for idx, (_, row) in enumerate(sample_failed.iterrows(), 1):
        job_id = row["job_id"]
        true_cause = row["failure_cause"]
        log_path = os.path.join("data", "logs", f"{job_id}.log")
        print(f"\n--- Job {idx}: {job_id} ---")
        print(f"Manifest Label: status={row['status']}, failure_cause={true_cause}, is_truncated={row['is_truncated']}")
        if os.path.exists(log_path):
            parsed = parse_log_file(log_path)
            print(f"Parsed Record:  status={parsed.status}, failure_cause={parsed.failure_cause}")
            print(f"Parsed Error Text: {parsed.raw_error_text[:120]}...")
            print(f"Parsed Message IDs: {parsed.raw_message_ids[:8]}")
            # Check for alignment
            label_match = (parsed.failure_cause == true_cause) if parsed.failure_cause else "Truncated/No summary"
            print(f"Log Summary Cause vs Manifest: {label_match}")
        else:
            print(f"Log file not found at: {log_path}")

    print("\n" + "=" * 80)
    print("STEP 3: EVIDENCE CHECK ACROSS ALL 7 CAUSES")
    print("=" * 80)
    evidence_keywords = {
        "out_of_memory": ["MEM-902", "oom-killer", "cgroup", "SYS-MEM-99", "memory allocation failure"],
        "timeout": ["TIME-501", "execution limit", "watchdog", "SYS-004", "timeout"],
        "missing_library": ["LIB-204", "LIB-901", "DB-301", "Cannot resolve target technology cell"],
        "constraint_violation": ["TIM-302", "DRC-401", "STA-500", "slack violation"],
        "license_error": ["LIC-101", "LIC-108", "LIC-999", "FLEXlm error", "Cannot checkout feature"],
        "syntax_error": ["HDL-105", "PARSE-010", "Syntax error"],
        "congestion": ["RTE-702", "CONG-801", "PNR-880", "congestion overflow", "Gcell"],
    }

    evidence_counts = {c: 0 for c in classes}
    total_cause_counts = {c: 0 for c in classes}
    no_evidence_count = 0
    truncated_no_evidence = 0
    samples_per_cause = {c: [] for c in classes}

    for _, row in failed_df.iterrows():
        job_id = row["job_id"]
        cause = row["failure_cause"]
        total_cause_counts[cause] += 1
        log_path = os.path.join("data", "logs", f"{job_id}.log")

        has_evidence = False
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            kws = evidence_keywords.get(cause, [])
            matched_lines = []
            for line in content.splitlines():
                if any(kw.lower() in line.lower() for kw in kws):
                    matched_lines.append(line.strip())

            if matched_lines:
                has_evidence = True
                evidence_counts[cause] += 1
                if len(samples_per_cause[cause]) < 2:
                    samples_per_cause[cause].append(matched_lines[0])

        if not has_evidence:
            no_evidence_count += 1
            if row.get("is_truncated", 0) == 1:
                truncated_no_evidence += 1

    print(f"Total failed jobs analyzed: {len(failed_df)}")
    for c in classes:
        tot = total_cause_counts[c]
        ev = evidence_counts[c]
        pct = (ev / tot * 100.0) if tot > 0 else 0.0
        print(f"Cause '{c}': {ev}/{tot} have evidence ({pct:.1f}%)")
        for s in samples_per_cause[c]:
            print(f"   Example: {s[:100]}")

    pct_no_ev = (no_evidence_count / len(failed_df) * 100.0) if len(failed_df) else 0.0
    print(f"\nFailed logs with NO cause-specific evidence: {no_evidence_count}/{len(failed_df)} ({pct_no_ev:.1f}%)")
    print(f"Of those with no evidence, truncated logs: {truncated_no_evidence}/{no_evidence_count}")

    print("\n" + "=" * 80)
    print("STEP 4: FEATURE CHECK (MANIFEST DF vs PARSED LOGS)")
    print("=" * 80)
    print(f"Columns present in failed_df (manifest): {list(failed_df.columns)}")
    print(f"Is 'error_text' in manifest columns? {'error_text' in failed_df.columns}")
    print(f"Is 'message_ids_str' in manifest columns? {'message_ids_str' in failed_df.columns}")

    # Build feature pipeline on raw manifest df as currently done in train_failure_classifier
    pipe_manifest = build_feature_pipeline(include_text=True, include_msg_ids=True)
    X_manifest = pipe_manifest.fit_transform(failed_df)
    print(f"Shape of feature matrix using manifest DF directly: {X_manifest.shape}")

    # Inspect TF-IDF step
    tfidf_step = pipe_manifest.named_steps["preprocessor"].named_transformers_["text_tfidf"]
    print(f"TF-IDF vocabulary on manifest DF: {list(tfidf_step.vocabulary_.keys())}")

    # Train a baseline classifier on manifest DF to inspect weights
    from sklearn.linear_model import LogisticRegression
    clf_manifest = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    clf_manifest.fit(X_manifest, y)
    feature_names_manifest = pipe_manifest.named_steps["preprocessor"].get_feature_names_out()
    print("\n15 Highest-weight features per class on current MANIFEST DF (No real text):")
    for c_idx, c_name in enumerate(clf_manifest.classes_):
        top15_idx = np.argsort(clf_manifest.coef_[c_idx])[::-1][:15]
        top15_feats = [f"{feature_names_manifest[i]} ({clf_manifest.coef_[c_idx][i]:.2f})" for i in top15_idx]
        print(f"  Class '{c_name}':")
        print(f"    {', '.join(top15_feats)}")

    # Now inspect if we populate error_text from parsed logs
    print("\nSimulating feature extraction WITH parsed log error text:")
    enriched_rows = []
    for _, row in failed_df.iterrows():
        job_id = row["job_id"]
        log_path = os.path.join("data", "logs", f"{job_id}.log")
        rec = parse_log_file(log_path)
        row_dict = dict(row)
        row_dict["error_text"] = rec.raw_error_text
        row_dict["message_ids_str"] = " ".join(rec.raw_message_ids)
        enriched_rows.append(row_dict)
    enriched_df = pd.DataFrame(enriched_rows)

    pipe_enriched = build_feature_pipeline(include_text=True, include_msg_ids=True)
    X_enriched = pipe_enriched.fit_transform(enriched_df)
    print(f"Shape of feature matrix with parsed log error_text: {X_enriched.shape}")
    tfidf_enriched = pipe_enriched.named_steps["preprocessor"].named_transformers_["text_tfidf"]
    print(f"TF-IDF vocabulary size with parsed logs: {len(tfidf_enriched.vocabulary_)}")
    print(f"Sample 15 vocabulary tokens from parsed logs: {list(tfidf_enriched.vocabulary_.keys())[:15]}")

    # Check non-zero TF-IDF features per row
    tfidf_matrix = tfidf_enriched.transform(enriched_df["error_text"])
    nonzeros_per_row = np.diff(tfidf_matrix.indptr)
    print(f"Average non-zero TF-IDF features per row: {np.mean(nonzeros_per_row):.1f} (min: {np.min(nonzeros_per_row)}, max: {np.max(nonzeros_per_row)})")

    # Fit LogisticRegression on enriched data and print top 15 features per class
    clf_enriched = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    clf_enriched.fit(X_enriched, y)
    feature_names_enriched = pipe_enriched.named_steps["preprocessor"].get_feature_names_out()
    print("\n15 Highest-weight features per class WITH PARSED TEXT:")
    for c_idx, c_name in enumerate(clf_enriched.classes_):
        top15_idx = np.argsort(clf_enriched.coef_[c_idx])[::-1][:15]
        top15_feats = [f"{feature_names_enriched[i].replace('text_tfidf__', '').replace('msg_bow__', '')} ({clf_enriched.coef_[c_idx][i]:.2f})" for i in top15_idx]
        print(f"  Class '{c_name}':")
        print(f"    {', '.join(top15_feats)}")

    # Step 5: Easy-mode test
    print("\n" + "=" * 80)
    print("STEP 5: EASY-MODE TEST (NOISE=OFF, TRUNCATION=OFF, AMBIGUITY=OFF)")
    print("=" * 80)
    import tempfile
    easy_dir = tempfile.mkdtemp()
    
    # Subclass or instantiate generator with truncation and noise forced OFF
    class EasyGenerator(SyntheticDataGenerator):
        def _generate_single_job(self, index, start_time):
            rec = super()._generate_single_job(index, start_time)
            # Easy mode: truncation OFF
            rec.is_truncated = 0
            return rec

    easy_gen = EasyGenerator(seed=42)
    easy_jobs = easy_gen.generate_dataset(num_jobs=600, output_dir=easy_dir, write_logs=True)
    easy_manifest = pd.read_csv(os.path.join(easy_dir, "manifest.csv"))
    easy_failed = easy_manifest[easy_manifest["status"] == "failed"].copy()

    easy_enriched = []
    for _, row in easy_failed.iterrows():
        log_path = os.path.join(easy_dir, "logs", f"{row['job_id']}.log")
        rec = parse_log_file(log_path)
        row_dict = dict(row)
        row_dict["error_text"] = rec.raw_error_text
        row_dict["message_ids_str"] = " ".join(rec.raw_message_ids)
        easy_enriched.append(row_dict)
    easy_df = pd.DataFrame(easy_enriched)

    y_easy = easy_df["failure_cause"].to_numpy()
    tr_df, ts_df, y_tr, y_ts = train_test_split(easy_df, y_easy, test_size=0.25, random_state=42, stratify=y_easy)

    # Train model on enriched features with text
    easy_pipe = build_feature_pipeline(include_text=True, include_msg_ids=True)
    X_tr = easy_pipe.fit_transform(tr_df)
    X_ts = easy_pipe.transform(ts_df)

    easy_clf = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    easy_clf.fit(X_tr, y_tr)
    y_pred_easy = easy_clf.predict(X_ts)
    f1_easy = f1_score(y_ts, y_pred_easy, average="macro")
    acc_easy = np.mean(y_ts == y_pred_easy)
    print(f"Easy-Mode Logistic Regression Accuracy with Parsed Text: {acc_easy * 100:.2f}%")
    print(f"Easy-Mode Logistic Regression Macro F1 with Parsed Text:   {f1_easy:.4f}")
    print("\nEasy-Mode Classification Report with Parsed Text:")
    print(classification_report(y_ts, y_pred_easy, zero_division=0))

    # Ablation on easy mode without text
    easy_pipe_notext = build_feature_pipeline(include_text=False, include_msg_ids=False)
    X_tr_no = easy_pipe_notext.fit_transform(tr_df)
    X_ts_no = easy_pipe_notext.transform(ts_df)
    clf_no = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    clf_no.fit(X_tr_no, y_tr)
    pred_no = clf_no.predict(X_ts_no)
    f1_no = f1_score(y_ts, pred_no, average="macro")
    print(f"Easy-Mode Logistic Regression Macro F1 WITHOUT Text:       {f1_no:.4f}")
    print(f"Easy-Mode Delta F1 from Text Features:                     {f1_easy - f1_no:+.4f}")


if __name__ == "__main__":
    run_diagnostics()

