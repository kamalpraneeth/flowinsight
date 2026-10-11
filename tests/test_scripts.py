"""Subprocess tests for Bash automation scripts.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

import os
import shutil
import subprocess
import sys
import tempfile

import pytest


def find_bash():
    """Locate bash executable on Linux or Git Bash on Windows."""
    if os.name == "nt":
        git_bash = "C:\\Program Files\\Git\\bin\\bash.exe"
        if os.path.exists(git_bash):
            return git_bash
    bash_path = shutil.which("bash")
    if bash_path and not bash_path.lower().endswith("system32\\bash.exe"):
        return bash_path
    return None


@pytest.fixture
def bash_exe():
    exe = find_bash()
    if not exe:
        pytest.skip("Compatible Bash executable not found in environment.")
    return exe


def test_collect_logs_script(bash_exe):
    tmp_src = tempfile.mkdtemp()
    tmp_dest = tempfile.mkdtemp()
    try:
        # Create dummy logs
        log1 = os.path.join(tmp_src, "run_01.log")
        with open(log1, "w", encoding="utf-8") as f:
            f.write(
                "# EDA Tool Suite - Subsystem: SYNTHESIS\n[INFO] Run completed.\nEXECUTION SUMMARY\n"
            )

        log2 = os.path.join(tmp_src, "stub.txt")
        with open(log2, "w", encoding="utf-8") as f:
            f.write("small")

        script_path = os.path.abspath("scripts/collect_logs.sh").replace("\\", "/")
        src_posix = tmp_src.replace("\\", "/")
        dest_posix = tmp_dest.replace("\\", "/")
        cmd = [bash_exe, script_path, src_posix, dest_posix, "10"]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        assert proc.returncode == 0
        assert os.path.isfile(os.path.join(tmp_dest, "run_01.log"))
        assert not os.path.isfile(os.path.join(tmp_dest, "stub.txt"))
    finally:
        shutil.rmtree(tmp_src, ignore_errors=True)
        shutil.rmtree(tmp_dest, ignore_errors=True)


def test_eda_flow_hook_script(bash_exe):
    tmp_dir = tempfile.mkdtemp()
    try:
        log_file = os.path.join(tmp_dir, "failed_run.log")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(
                "# EDA Tool Suite - Subsystem: SYNTHESIS\n"
                "ERROR: [MEM-902] Fatal memory allocation failure\n"
                "Execution Status : FAILED\n"
            )

        script_path = os.path.abspath("scripts/eda_flow_hook_example.sh").replace("\\", "/")
        log_posix = log_file.replace("\\", "/")
        cmd = [bash_exe, script_path, log_posix, "riscv_core", "synthesis"]
        env = {**os.environ, "PYTHON": sys.executable.replace("\\", "/")}
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

        assert proc.returncode == 0
        assert "EXAMPLE HOOK" in proc.stdout
        assert (
            "Notice: Illustrative demonstration only" in proc.stdout
            or "NOTICE: Illustrative" in proc.stdout
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
