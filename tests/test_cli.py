"""Unit tests for Phase 5 CLI subcommands.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

import os
import shutil
import tempfile

import pytest

from flowinsight.cli import main
from flowinsight.generator import SyntheticDataGenerator


@pytest.fixture(scope="module")
def cli_env():
    tmp_dir = tempfile.mkdtemp()
    gen = SyntheticDataGenerator(seed=42)
    gen.generate_dataset(num_jobs=35, output_dir=tmp_dir, write_logs=True)
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_cli_generate_command():
    tmp = tempfile.mkdtemp()
    try:
        ret = main(["generate", "--jobs", "15", "--seed", "99", "--output-dir", tmp])
        assert ret == 0
        assert os.path.isfile(os.path.join(tmp, "manifest.csv"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cli_parse_single_log(cli_env):
    log_file = os.path.join(cli_env, "logs", os.listdir(os.path.join(cli_env, "logs"))[0])
    ret = main(["parse", "--log", log_file])
    assert ret == 0


def test_cli_parse_and_train_commands(cli_env):
    db_file = os.path.join(cli_env, "test.db")
    manifest = os.path.join(cli_env, "manifest.csv")
    models_dir = os.path.join(cli_env, "models_saved")

    # Parse into DB
    ret_parse = main(["parse", "--manifest", manifest, "--db", db_file])
    assert ret_parse == 0
    assert os.path.isfile(db_file)

    # Train models
    ret_train = main(["train", "--manifest", manifest, "--output-dir", models_dir])
    assert ret_train == 0
    assert os.path.isfile(os.path.join(models_dir, "runtime_s_model.joblib"))
    assert os.path.isfile(os.path.join(models_dir, "failure_model.joblib"))
    assert os.path.isfile(os.path.join(models_dir, "anomaly_model.joblib"))


def test_cli_predict_runtime(cli_env):
    models_dir = os.path.join(cli_env, "models_saved")
    if not os.path.exists(models_dir):
        manifest = os.path.join(cli_env, "manifest.csv")
        main(["train", "--manifest", manifest, "--output-dir", models_dir])

    ret = main(
        [
            "predict-runtime",
            "--size",
            "300000",
            "--stage",
            "place_and_route",
            "--threads",
            "8",
            "--model-dir",
            models_dir,
        ]
    )
    assert ret == 0


def test_cli_classify_failure(cli_env):
    models_dir = os.path.join(cli_env, "models_saved")
    # Find a failed log
    log_files = [f for f in os.listdir(os.path.join(cli_env, "logs")) if f.endswith(".log")]
    assert len(log_files) > 0
    sample_log = os.path.join(cli_env, "logs", log_files[0])

    ret = main(["classify-failure", "--log", sample_log, "--model-dir", models_dir])
    assert ret == 0


def test_cli_report_generation(cli_env):
    manifest = os.path.join(cli_env, "manifest.csv")
    rep_out = os.path.join(cli_env, "out_report.md")
    ret = main(["report", "--manifest", manifest, "--output", rep_out])
    assert ret == 0
    assert os.path.isfile(rep_out)
