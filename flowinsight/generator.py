"""Synthetic EDA tool log generator for FlowInsight.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from flowinsight.parsers import parse_log_file


@dataclass
class JobRecord:
    job_id: str
    stage: str
    design_name: str
    design_size: int
    tool_version: str
    config_json: str
    start_time: str
    runtime_s: float
    peak_memory_mb: float
    warnings_count: int
    errors_count: int
    status: str
    failure_cause: str
    is_anomaly: int
    anomaly_type: str
    is_truncated: int
    data_source: str = "synthetic"
    error_text: str = ""
    message_ids_str: str = ""


# Design pool
DESIGNS = [
    ("riscv_core", 85_000),
    ("pcie_controller", 175_000),
    ("dram_ctrl", 240_000),
    ("dsp_mac_array", 360_000),
    ("noc_router", 510_000),
    ("crypto_aes_sha", 640_000),
    ("ethernet_mac_100g", 890_000),
    ("gpu_shader_slice", 1_450_000),
    ("neural_engine_pe", 1_920_000),
    ("usb4_phy_wrapper", 2_350_000),
]

TOOL_VERSIONS = {
    "synthesis": ["v2023.03-SP1", "v2023.12-SP2", "v2024.06"],
    "place_and_route": ["v22.11-s082_1", "v23.10-s095_1", "v24.04-patch3"],
    "verification": ["v2023.09-patch1", "v2024.03-patch2", "v2024.08-rc1"],
}

FAILURE_CAUSES = [
    "out_of_memory",
    "timeout",
    "missing_library",
    "constraint_violation",
    "license_error",
    "syntax_error",
    "congestion",  # P&R only
]

EFFORT_LEVELS = ["low", "medium", "high", "extreme"]
TARGETS = ["area", "timing", "power"]
EFFORT_MULTIPLIERS = {"low": 0.75, "medium": 1.0, "high": 1.35, "extreme": 1.85}
TARGET_MULTIPLIERS = {"area": 0.95, "power": 1.05, "timing": 1.20}


class SyntheticDataGenerator:
    """Generates synthetic EDA tool execution runs, text logs, and manifests."""

    def __init__(
        self,
        seed: int = 42,
        enable_noise: bool = True,
        enable_truncation: bool = True,
        enable_ambiguity: bool = True,
        truncation_rate: float = 0.15,
        ambiguity_rate: float = 0.05,
    ) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.enable_noise = enable_noise
        self.enable_truncation = enable_truncation
        self.enable_ambiguity = enable_ambiguity
        self.truncation_rate = truncation_rate
        self.ambiguity_rate = ambiguity_rate

    def generate_dataset(
        self,
        num_jobs: int = 5000,
        output_dir: str = "data",
        write_logs: bool = True,
    ) -> List[JobRecord]:
        """Generate a complete dataset of jobs and write log files and manifest CSV."""
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        manifest_path = os.path.join(output_dir, "manifest.csv")
        anomalies_path = os.path.join(output_dir, "anomalies.csv")

        jobs: List[JobRecord] = []
        base_time = datetime(2024, 1, 15, 8, 0, 0)

        for i in range(num_jobs):
            # Time progression across 180 days
            job_time = base_time + timedelta(
                days=(i / max(1, num_jobs)) * 180,
                minutes=self.rng.randint(0, 1439),
                seconds=self.rng.randint(0, 59),
            )
            job = self._generate_single_job(i, job_time)

            if write_logs:
                log_content = self._generate_log_content(job)
                log_path = os.path.join(output_dir, "logs", f"{job.job_id}.log")
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(log_content)
                parsed = parse_log_file(log_path)
                job.error_text = parsed.raw_error_text
                job.message_ids_str = " ".join(parsed.raw_message_ids)

            jobs.append(job)

        # Write manifest CSV
        fieldnames = list(asdict(jobs[0]).keys())
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for job in jobs:
                writer.writerow(asdict(job))

        # Write separate anomalies CSV with ground truth
        anomaly_jobs = [job for job in jobs if job.is_anomaly == 1]
        with open(anomalies_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for job in anomaly_jobs:
                writer.writerow(asdict(job))

        return jobs

    def _generate_single_job(self, index: int, start_time: datetime) -> JobRecord:
        stages = ["synthesis", "place_and_route", "verification"]
        stage = self.rng.choice(stages)
        stage_prefix = {"synthesis": "SYN", "place_and_route": "PNR", "verification": "VRF"}[stage]
        job_id = f"JOB-{stage_prefix}-{index + 1:05d}"

        design_template, base_size = self.rng.choice(DESIGNS)
        design_size = int(base_size * self.rng.uniform(0.85, 1.15))

        tool_version = self.rng.choice(TOOL_VERSIONS[stage])
        effort = self.rng.choice(EFFORT_LEVELS)
        target = self.rng.choice(TARGETS)
        threads = self.rng.choice([2, 4, 8, 16])
        clock_period_ns = round(self.rng.uniform(0.8, 3.5), 2)
        layers = self.rng.choice([4, 6, 8, 10]) if stage == "place_and_route" else 0

        config: Dict[str, Any] = {
            "effort": effort,
            "target": target,
            "threads": threads,
            "clock_period_ns": clock_period_ns,
        }
        if stage == "place_and_route":
            config["max_routing_layers"] = layers

        # Failure assignment (~22% failure rate overall)
        is_failed = self.rng.random() < 0.22
        failure_cause = "none"
        if is_failed:
            if stage == "place_and_route":
                candidate_causes = FAILURE_CAUSES
            else:
                candidate_causes = [c for c in FAILURE_CAUSES if c != "congestion"]
            failure_cause = self.rng.choice(candidate_causes)

        # Base physical scaling formula
        norm_size = design_size / 100_000.0
        effort_mult = EFFORT_MULTIPLIERS[effort]
        target_mult = TARGET_MULTIPLIERS[target]

        if stage == "synthesis":
            base_runtime = 120.0 * (norm_size**1.15) * effort_mult * target_mult * (threads**-0.55)
            base_memory = 800.0 + 420.0 * (norm_size**1.05) * (1.0 + 0.03 * threads)
        elif stage == "place_and_route":
            layer_mult = 1.0 + 0.05 * (layers - 4)
            base_runtime = 280.0 * (norm_size**1.35) * effort_mult * layer_mult * (threads**-0.45)
            base_memory = 950.0 + 720.0 * (norm_size**1.12) * (1.0 + 0.04 * threads)
        else:  # verification
            base_runtime = 90.0 * (norm_size**1.05) * effort_mult * (threads**-0.35)
            base_memory = 600.0 + 310.0 * (norm_size**0.98) * (1.0 + 0.02 * threads)

        # Stochastic noise
        if self.enable_noise:
            noise_factor = math.exp(self.rng.gauss(0.0, 0.12))
            mem_noise = self.rng.gauss(0.0, 65.0)
            warnings_count = max(0, int(self.rng.gauss(14.0 + norm_size * 2.5, 6.0)))
        else:
            noise_factor = 1.0
            mem_noise = 0.0
            warnings_count = max(0, int(14.0 + norm_size * 2.5))

        runtime_s = max(15.0, round(base_runtime * noise_factor, 2))
        peak_memory_mb = max(250.0, round(base_memory + mem_noise, 1))
        errors_count = 0 if not is_failed else self.rng.randint(1, 5)

        # Adjustments by failure cause
        if failure_cause == "out_of_memory":
            peak_memory_mb = round(peak_memory_mb * self.rng.uniform(1.8, 2.5), 1)
        elif failure_cause == "timeout":
            runtime_s = round(runtime_s * self.rng.uniform(2.5, 3.5), 2)
        elif failure_cause == "syntax_error":
            runtime_s = max(8.0, round(runtime_s * 0.12, 2))
            peak_memory_mb = max(300.0, round(peak_memory_mb * 0.25, 1))

        # Anomaly injection
        is_anomaly = 0
        anomaly_type = "none"
        anomaly_roll = self.rng.random()
        if anomaly_roll < 0.03:
            is_anomaly = 1
            anomaly_type = "runtime_spike"
            runtime_s = round(runtime_s * self.rng.uniform(3.5, 5.8), 2)
        elif anomaly_roll < 0.06:
            is_anomaly = 1
            anomaly_type = "memory_growth"
            peak_memory_mb = round(peak_memory_mb * self.rng.uniform(2.5, 4.2), 1)
        elif anomaly_roll < 0.09:
            is_anomaly = 1
            anomaly_type = "warning_storm"
            warnings_count += self.rng.randint(220, 550)

        is_truncated = (
            1
            if (self.enable_truncation and is_failed and self.rng.random() < self.truncation_rate)
            else 0
        )
        status = "failed" if is_failed else "completed"

        return JobRecord(
            job_id=job_id,
            stage=stage,
            design_name=design_template,
            design_size=design_size,
            tool_version=tool_version,
            config_json=json.dumps(config, sort_keys=True),
            start_time=start_time.isoformat() + "Z",
            runtime_s=runtime_s,
            peak_memory_mb=peak_memory_mb,
            warnings_count=warnings_count,
            errors_count=errors_count,
            status=status,
            failure_cause=failure_cause,
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            is_truncated=is_truncated,
            data_source="synthetic",
        )

    def _generate_log_content(self, job: JobRecord) -> str:
        """Construct realistic multi-step log text with realistic headers and messages."""
        lines: List[str] = []
        config = json.loads(job.config_json)
        th = config.get("threads", 4)
        target = config.get("target", "timing")

        # Header block
        lines.append("#" * 80)
        lines.append(f"# EDA Tool Suite - Subsystem: {job.stage.upper()}")
        lines.append(f"# Release Version: {job.tool_version}")
        lines.append(f"# Design: {job.design_name} (Gate count: {job.design_size})")
        lines.append(f"# Invocation: run_{job.stage} -design {job.design_name} -threads {th}")
        lines.append(f"# Timestamp: {job.start_time}")
        lines.append(f"# Host: compute-node-{self.rng.randint(101, 299)}.eda.cluster")
        lines.append(
            "# NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs."
        )
        lines.append("#" * 80)
        lines.append("")

        if job.stage == "synthesis":
            steps = ["INITIALIZATION", "READ_RTL", "ELABORATION", "MAPPING", "TIMING_REPORT"]
        elif job.stage == "place_and_route":
            steps = ["INITIALIZATION", "FLOORPLAN", "POWER_GRID", "PLACEMENT", "CTS", "ROUTING"]
        else:
            steps = ["ENV_SETUP", "COMPILE_TB", "ELABORATE", "SIMULATION_INIT", "EXECUTE_VECTORS"]

        total_steps = len(steps)
        has_precursor = False
        is_ambiguous = False
        if job.status == "failed":
            if self.enable_ambiguity and (self.rng.random() < self.ambiguity_rate):
                is_ambiguous = True
            elif job.is_truncated == 1:
                # 60% of truncated failed jobs retain precursor evidence; 40% are clean cut evidence-free
                has_precursor = self.rng.random() < 0.60
            else:
                has_precursor = True

        if job.status == "failed":
            if job.failure_cause in ("syntax_error", "missing_library", "license_error"):
                cutoff_step = self.rng.randint(1, min(3, total_steps))
            elif job.failure_cause == "out_of_memory":
                cutoff_step = self.rng.randint(2, total_steps)
            else:
                cutoff_step = self.rng.randint(2, total_steps)
            active_steps = steps[:cutoff_step]
        else:
            active_steps = steps

        elapsed_cum = 0.0
        mem_cum = 300.0
        step_delta_t = job.runtime_s / max(1, len(active_steps))
        step_delta_m = (job.peak_memory_mb - 300.0) / max(1, len(active_steps))

        is_modern = "v2024" in job.tool_version or "v24" in job.tool_version
        is_legacy = "v22" in job.tool_version or "SP1" in job.tool_version

        for idx, step_name in enumerate(active_steps, 1):
            elapsed_cum += step_delta_t * self.rng.uniform(0.85, 1.15)
            mem_cum = min(
                job.peak_memory_mb,
                mem_cum + max(0.0, step_delta_m * self.rng.uniform(0.7, 1.3)),
            )

            lines.append(f">> [STEP {idx}/{total_steps}] STARTING: {step_name}")
            lines.append(f"   [INFO-012] Entering module {step_name.lower()} with target {target}")

            if self.rng.random() < 0.35:
                if is_modern:
                    t_str = f"{job.start_time[:10]}T12:{idx:02d}:00"
                    lines.append(f"   [{t_str}] [INFO] [CORE-100] Parallel workers active: {th}")
                elif is_legacy:
                    lines.append(
                        f"   **INFO: (TOOL-SYS-04) Thread affinity mapped across {th} cores."
                    )
                else:
                    lines.append("   [SYS-001] Dispatched worker thread pool on local node")

            # Unknown message types occasionally
            if self.rng.random() < 0.15:
                lines.append(
                    "   Tcl Script Note: User constraint override detected in config hook."
                )
            if self.rng.random() < 0.10:
                lines.append("   **DIAG-NOTICE** Internal cache utilization index: 94.2%")

            # Benign warnings
            step_warnings = job.warnings_count // len(active_steps)
            for _ in range(min(5, step_warnings)):
                w_id = self.rng.choice(["TIM-104", "DRC-201", "LIB-102", "PIN-040", "ENV-011"])
                r_num = self.rng.randint(1, 64)
                if is_modern:
                    lines.append(
                        f"   [WARN] [{w_id}] Non-critical timing margin degradation net_{r_num}"
                    )
                elif is_legacy:
                    lines.append(
                        f"   **WARNING: ({w_id}) Minor slew transition slack below margin U{r_num}"
                    )
                else:
                    lines.append(
                        f"   [WARNING:{w_id}] Fanout exceeds threshold for buffer inst_buf_{r_num}"
                    )

            # Warning storm anomaly handling
            if job.anomaly_type == "warning_storm" and idx == len(active_steps):
                lines.append("   " + "-" * 60)
                lines.append("   >>> REPEATED WARNING CASCADE COMMENCING <<<")
                for storm_idx in range(min(120, job.warnings_count)):
                    lines.append(
                        f"   [WARNING:DRC-STORM-99] Geometric grid spacing violation at ({storm_idx * 10}, {storm_idx * 15})"
                    )
                lines.append("   >>> REPEATED WARNING CASCADE SUPPRESSED <<<")

            # Precursor warning injection on the final active step for failing jobs
            if job.status == "failed" and idx == len(active_steps):
                if is_ambiguous:
                    lines.append(
                        "   [WARN] [GEN-002] Diagnostic execution anomaly detected in worker thread."
                    )
                elif has_precursor:
                    lines.extend(self._get_precursor_warning_lines(job.failure_cause))

            lines.append(
                f"   [STEP {idx} TIMING] CPU: {elapsed_cum * 1.8:.1f}s | Wall: {elapsed_cum:.1f}s | Mem: {mem_cum:.1f}MB"
            )
            lines.append(f">> [STEP {idx}/{total_steps}] FINISHED: {step_name}")
            lines.append("")

        active_steps_end_idx = len(lines)

        # Failure injection
        if job.status == "failed":
            lines.append("=" * 80)
            lines.append("FATAL TERMINATION ENCOUNTERED")
            lines.append("=" * 80)
            if is_ambiguous:
                lines.append("ERROR: [GEN-001] Unspecified fatal tool error during execution.")
                lines.append(
                    "FATAL: [GEN-002] Solver process terminated abnormally with generic error."
                )
            else:
                cause_lines = self._get_failure_log_lines(
                    job.failure_cause, job.design_name, job.peak_memory_mb, config
                )
                lines.extend(cause_lines)
            lines.append("")

        # Truncation check
        if job.is_truncated == 1:
            if has_precursor:
                cut_idx = active_steps_end_idx
            else:
                cut_idx = max(8, int(active_steps_end_idx * self.rng.uniform(0.45, 0.70)))
            truncated_lines = lines[:cut_idx]
            truncated_lines.append(
                "... [LOG STREAM TRUNCATED BY HOST AGENT / SUDDEN PROCESS TERMINATION] ..."
            )
            return "\n".join(truncated_lines)

        # Execution summary
        lines.append("=" * 80)
        lines.append("                               EXECUTION SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Design Name             : {job.design_name}")
        lines.append(f"Gate Count              : {job.design_size}")
        lines.append(f"Stage                   : {job.stage}")
        lines.append(
            f"Execution Status        : {'COMPLETED' if job.status == 'completed' else 'FAILED'}"
        )
        if job.status == "failed":
            lines.append(f"Primary Failure Code    : {job.failure_cause.upper()}")
        lines.append(f"Total Wallclock Time    : {job.runtime_s:.2f} s")
        lines.append(f"Total CPU Time          : {job.runtime_s * th * 0.72:.2f} s")
        lines.append(f"Peak Memory Allocated   : {job.peak_memory_mb:.1f} MB")
        lines.append(f"Total Warnings Logged   : {job.warnings_count}")
        lines.append(f"Total Errors Logged     : {job.errors_count}")
        lines.append(f"Exit Code               : {0 if job.status == 'completed' else 1}")
        lines.append("=" * 80)
        lines.append("End of Log File.")

        return "\n".join(lines)

    def _get_failure_log_lines(
        self,
        cause: str,
        design: str,
        peak_mem: float,
        config: Dict[str, Any],
    ) -> List[str]:
        evidence: List[str] = []

        if cause == "out_of_memory":
            evidence.append(
                "ERROR: [MEM-902] Fatal memory allocation failure: requested chunk exceeded host limits."
            )
            evidence.append(
                f"FATAL: Kernel cgroup memory controller invoked oom-killer (peak resident memory: {peak_mem:.1f}MB)."
            )
            evidence.append(
                "ERROR: [SYS-MEM-99] Unable to allocate 4194304 bytes during graph partition."
            )
            if self.rng.random() < 0.4:
                evidence.append(
                    "WARNING: [LIB-102] Library search path cache freed in emergency reclaim."
                )

        elif cause == "timeout":
            evidence.append(
                "ERROR: [TIME-501] CPU execution limit exceeded. Maximum allowable wallclock window expired."
            )
            evidence.append(
                "FATAL: [SYS-004] Tool elapsed wallclock time exceeded maximum configured execution limit."
            )
            evidence.append(
                "Process killed by watchdog timer after non-converging optimization iterations."
            )

        elif cause == "missing_library":
            cell = self.rng.choice(["INV_X4_LVT", "DFF_X2_RVT", "NAND2_X8_HVT", "SRAM_2P_512x64"])
            evidence.append(
                f"ERROR: [LIB-204] Cannot resolve target technology cell '{cell}' in loaded technology library."
            )
            evidence.append(
                f"FATAL: [LIB-901] Technology file or LEF definition missing for design entity '{design}'."
            )
            evidence.append(
                "ERROR: [DB-301] Unresolved reference cell pin count mismatch in netlist binding."
            )

        elif cause == "constraint_violation":
            slack = round(self.rng.uniform(-2.5, -0.8), 2)
            cp = config.get("clock_period_ns", 1.0)
            evidence.append(
                f"ERROR: [TIM-302] Setup slack violation exceeds threshold: {slack}ns on reg_pipe_3/D."
            )
            evidence.append(
                "ERROR: [DRC-401] Max transition violation: 0.65ns > 0.40ns on clock network root."
            )
            evidence.append(
                f"FATAL: [STA-500] Unrecoverable timing slack failure; cannot close timing within {cp}ns."
            )

        elif cause == "license_error":
            feature = self.rng.choice(
                ["Design-Compiler-Ultra", "Innovus-Hierarchical-P&R", "VCS-Multicore-Sim"]
            )
            evidence.append(
                f"ERROR: [LIC-101] FLEXlm error: Cannot checkout feature '{feature}' - No such feature (-5,343)."
            )
            evidence.append(
                "ERROR: [LIC-108] License server unreachable at 27000@lic-server-eda01."
            )
            evidence.append(
                "FATAL: [LIC-999] License checkout timed out after 3 retry attempts; aborting run."
            )

        elif cause == "syntax_error":
            line_no = self.rng.randint(32, 280)
            evidence.append(
                f"ERROR: [HDL-105] HDL parsing grammar error in file 'rtl/{design}.v' line {line_no}: unexpected token ';'."
            )
            evidence.append(
                "FATAL: [PARSE-010] Elaboration stopped due to unrecoverable HDL parse grammar failure."
            )

        elif cause == "congestion":
            overflow = round(self.rng.uniform(4.5, 9.2), 1)
            evidence.append(
                f"ERROR: [RTE-702] Global routing failed: Gcell channel routing density overflow exceeds {overflow}% in M3/M4."
            )
            evidence.append(
                "ERROR: [CONG-801] Spatial routing density hotspot exceeds routability threshold: tile (120, 84)."
            )
            evidence.append(
                "FATAL: [PNR-880] Routing completion abort: maximum DRC iterations exhausted with shorts."
            )

        else:
            evidence.append("ERROR: [GEN-001] Unspecified fatal tool error during execution.")

        return evidence

    def _get_precursor_warning_lines(self, cause: str) -> List[str]:
        """Precursor diagnostic warnings emitted in execution steps before failure."""
        if cause == "out_of_memory":
            return [
                "   [WARN] [MEM-101] Host memory utilization critical: node resident set size climbing past 85% limit."
            ]
        elif cause == "timeout":
            return [
                "   [WARN] [TIME-102] Execution step convergence rate sub-nominal: wallclock pacing watchdog alert."
            ]
        elif cause == "missing_library":
            return [
                "   [WARN] [LIB-102] Target technology cell resolution warning in macro binding pass."
            ]
        elif cause == "constraint_violation":
            return [
                "   [WARN] [TIM-104] Critical path setup timing margin degradation observed."
            ]
        elif cause == "license_error":
            return [
                "   [WARN] [LIC-105] Tool license heartbeat verification delayed: scheduling retry attempt."
            ]
        elif cause == "syntax_error":
            return [
                "   [WARN] [HDL-102] Parse grammar irregularity encountered in module elaboration."
            ]
        elif cause == "congestion":
            return [
                "   [WARN] [CONG-102] Routing cell density warning: global routing hotspot tile."
            ]
        return ["   [WARN] [GEN-002] Generic system diagnostic warning in thread worker."]


def generate_synthetic_data(
    num_jobs: int = 5000,
    seed: int = 42,
    output_dir: str = "data",
    enable_noise: bool = True,
    enable_truncation: bool = True,
    enable_ambiguity: bool = True,
    truncation_rate: float = 0.15,
    ambiguity_rate: float = 0.05,
) -> List[JobRecord]:
    """Helper entry point for CLI and tests."""
    generator = SyntheticDataGenerator(
        seed=seed,
        enable_noise=enable_noise,
        enable_truncation=enable_truncation,
        enable_ambiguity=enable_ambiguity,
        truncation_rate=truncation_rate,
        ambiguity_rate=ambiguity_rate,
    )
    return generator.generate_dataset(num_jobs=num_jobs, output_dir=output_dir, write_logs=True)
