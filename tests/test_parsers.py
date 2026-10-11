"""Unit tests for Phase 2 Streaming Log Parsers.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

import tempfile

import pytest

from flowinsight.parsers import (
    PlaceAndRouteLogParser,
    SynthesisLogParser,
    VerificationLogParser,
    get_parser,
    parse_log_file,
)

SAMPLE_SYNTH_LOG = """################################################################################
# EDA Tool Suite - Subsystem: SYNTHESIS
# Release Version: v2024.06
# Design: riscv_core (Gate count: 85000)
# Invocation: run_synthesis -design riscv_core -threads 8
# Timestamp: 2024-02-01T10:00:00Z
# Host: compute-node-101.eda.cluster
# NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
################################################################################

>> [STEP 1/5] STARTING: INITIALIZATION
   [INFO-012] Entering module initialization with target timing
   [WARN] [TIM-104] Non-critical timing margin degradation net_12
   [STEP 1 TIMING] CPU: 12.0s | Wall: 8.0s | Mem: 450.0MB
>> [STEP 1/5] FINISHED: INITIALIZATION

>> [STEP 2/5] STARTING: MAPPING
   [INFO-012] Entering module mapping with target timing
   [STEP 2 TIMING] CPU: 45.0s | Wall: 32.0s | Mem: 720.0MB
>> [STEP 2/5] FINISHED: MAPPING

================================================================================
                               EXECUTION SUMMARY
================================================================================
Design Name             : riscv_core
Gate Count              : 85000
Stage                   : synthesis
Execution Status        : COMPLETED
Total Wallclock Time    : 32.00 s
Total CPU Time          : 45.00 s
Peak Memory Allocated   : 720.0 MB
Total Warnings Logged   : 1
Total Errors Logged     : 0
Exit Code               : 0
================================================================================
End of Log File.
"""

SAMPLE_TRUNCATED_LOG = """################################################################################
# EDA Tool Suite - Subsystem: PLACE_AND_ROUTE
# Release Version: v23.10-s095_1
# Design: gpu_shader_slice (Gate count: 1450000)
# Invocation: run_place_and_route -design gpu_shader_slice -threads 8
# Timestamp: 2024-02-02T11:00:00Z
################################################################################

>> [STEP 1/6] STARTING: FLOORPLAN
   [STEP 1 TIMING] CPU: 15.0s | Wall: 12.0s | Mem: 1200.0MB
>> [STEP 1/6] FINISHED: FLOORPLAN

>> [STEP 2/6] STARTING: PLACEMENT
   ERROR: [MEM-902] Fatal memory allocation failure: requested chunk exceeded host limits.
   FATAL: Kernel cgroup memory controller invoked oom-killer (peak resident memory: 4800.0MB).
   [STEP 2 TIMING] CPU: 80.0s | Wall: 65.0s | Mem: 4800.0MB
... [LOG STREAM TRUNCATED BY HOST AGENT / SUDDEN PROCESS TERMINATION] ...
"""

SAMPLE_CORRUPTED_LOG = """Random garbage line 1: &&^%$#@!
Null byte simulation or non-utf8 corrupted line
[ERROR:CORRUPT-999] Unexpected internal core dump at unknown offset
**WARNING: (TIM-104) Minor slew transition
Execution Status : FAILED
Total Wallclock Time : not-a-number s
Total Errors Logged : 3
"""


def test_parsers_good_sample():
    parser = SynthesisLogParser()
    record = parser.parse_stream(SAMPLE_SYNTH_LOG.splitlines())

    assert record.stage == "synthesis"
    assert record.design_name == "riscv_core"
    assert record.design_size == 85000
    assert record.tool_version == "v2024.06"
    assert record.threads == 8
    assert record.status == "completed"
    assert record.runtime_s == 32.0
    assert record.peak_memory_mb == 720.0
    assert len(record.steps) == 2
    assert record.steps[0].step_name == "INITIALIZATION"
    assert record.steps[1].step_name == "MAPPING"
    assert len(record.messages) >= 1
    msg_ids = [m.message_id for m in record.messages]
    assert "TIM-104" in msg_ids
    assert any(m.severity == "WARN" for m in record.messages)
    assert len(record.parse_warnings) == 0


def test_parsers_truncated_sample():
    parser = PlaceAndRouteLogParser()
    record = parser.parse_stream(SAMPLE_TRUNCATED_LOG.splitlines())

    assert record.stage == "place_and_route"
    assert record.design_name == "gpu_shader_slice"
    # Even without summary, fallback should capture from steps
    assert record.runtime_s == 65.0
    assert record.peak_memory_mb == 4800.0
    assert record.status == "failed"
    assert any("truncated" in w.lower() for w in record.parse_warnings)
    assert any("MEM-902" in m.message_id for m in record.messages)
    assert "MEM-902" in record.raw_error_text


def test_parsers_corrupted_sample():
    parser = VerificationLogParser()
    # Parsing corrupted lines must never raise exception
    record = parser.parse_stream(SAMPLE_CORRUPTED_LOG.splitlines())

    assert record.status == "failed"
    assert len(record.messages) >= 1
    assert any("truncated" in w.lower() for w in record.parse_warnings)


def test_parser_factory_and_file_parsing():
    p1 = get_parser("synthesis")
    assert isinstance(p1, SynthesisLogParser)
    p2 = get_parser("place_and_route")
    assert isinstance(p2, PlaceAndRouteLogParser)
    p3 = get_parser("verification")
    assert isinstance(p3, VerificationLogParser)

    with pytest.raises(ValueError):
        get_parser("unknown_stage_name")

    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(SAMPLE_SYNTH_LOG)
        f_name = f.name

    parsed = parse_log_file(f_name)
    assert parsed.stage == "synthesis"
    assert parsed.design_name == "riscv_core"
