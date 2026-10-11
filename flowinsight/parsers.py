"""Streaming log parsers for EDA tool executions.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class LogMessage:
    line_number: int
    severity: str
    message_id: str
    text: str


@dataclass
class StepTiming:
    step_index: int
    step_name: str
    cpu_s: float
    wall_s: float
    mem_mb: float


@dataclass
class ParsedLogRecord:
    job_id: str
    stage: str
    design_name: Optional[str] = None
    design_size: Optional[int] = None
    tool_version: Optional[str] = None
    threads: Optional[int] = None
    status: str = "unknown"
    runtime_s: float = 0.0
    peak_memory_mb: float = 0.0
    warnings_count: int = 0
    errors_count: int = 0
    exit_code: int = 0
    failure_cause: Optional[str] = None
    steps: List[StepTiming] = field(default_factory=list)
    messages: List[LogMessage] = field(default_factory=list)
    parse_warnings: List[str] = field(default_factory=list)
    raw_error_text: str = ""
    raw_message_ids: List[str] = field(default_factory=list)
    data_source: str = "synthetic"


class LogParser(ABC):
    """Abstract base class for streaming line-by-line EDA log parsers."""

    def __init__(self, stage: str) -> None:
        self.stage = stage

        # Regex patterns compiled for performance
        self.re_header_stage = re.compile(
            r"#\s*EDA Tool Suite - Subsystem:\s*([A-Za-z0-9_\-]+)", re.IGNORECASE
        )
        self.re_header_ver = re.compile(
            r"#\s*Release Version:\s*([A-Za-z0-9_\.\-]+)", re.IGNORECASE
        )
        self.re_header_design = re.compile(
            r"#\s*Design:\s*([A-Za-z0-9_]+)\s*\(Gate count:\s*(\d+)\)", re.IGNORECASE
        )
        self.re_header_invoc = re.compile(r"#\s*Invocation:.*-threads\s*(\d+)", re.IGNORECASE)

        self.re_step_start = re.compile(r">> \[STEP (\d+)/(\d+)\] STARTING:\s*([A-Za-z0-9_]+)")
        self.re_step_timing = re.compile(
            r"\[STEP (\d+) TIMING\]\s*CPU:\s*([\d\.]+)s\s*\|\s*Wall:\s*([\d\.]+)s\s*\|\s*Mem:\s*([\d\.]+)MB"
        )

        # Message formats
        # Modern timestamped: [2024-04-10T12:01:00] [WARN] [TIM-104] text
        self.re_msg_modern = re.compile(
            r"\[[^\]]*\]\s*\[(WARN|WARNING|ERROR|FATAL|INFO)\]\s*\[([A-Za-z0-9_\-]+)\]\s*(.*)"
        )
        # Bracket pair without timestamp: [WARN] [TIM-104] text
        self.re_msg_bracket_pair = re.compile(
            r"\[(WARN|WARNING|ERROR|FATAL|INFO)\]\s*\[([A-Za-z0-9_\-]+)\]\s*(.*)"
        )
        # Hyphenated code: [INFO-012] or [WARN-TIM-104] text
        self.re_msg_hyphen = re.compile(
            r"\[(INFO|WARN|WARNING|ERROR|FATAL)-([A-Za-z0-9_\-]+)\]\s*(.*)"
        )
        # Standard: [WARNING:TIM-104] or ERROR: [MEM-902] text
        self.re_msg_bracket = re.compile(
            r"(?:\[(WARN|WARNING|ERROR|FATAL|INFO):([A-Za-z0-9_\-]+)\]|(ERROR|FATAL|WARNING|WARN|INFO):\s*\[([A-Za-z0-9_\-]+)\])\s*(.*)"
        )
        # Legacy: **WARNING: (TIM-104) text
        self.re_msg_legacy = re.compile(
            r"\*\*(ERROR|FATAL|WARNING|WARN|INFO):\s*\(([A-Za-z0-9_\-]+)\)\s*(.*)"
        )
        # Fallback error line without bracket ID
        self.re_msg_fallback = re.compile(r"(ERROR|FATAL|WARNING|WARN):\s*(.*)", re.IGNORECASE)

        # Execution Summary lines
        self.re_sum_status = re.compile(r"Execution Status\s*:\s*([A-Za-z]+)", re.IGNORECASE)
        self.re_sum_cause = re.compile(
            r"Primary Failure Code\s*:\s*([A-Za-z0-9_\-]+)", re.IGNORECASE
        )
        self.re_sum_wall = re.compile(r"Total Wallclock Time\s*:\s*([\d\.]+)\s*s", re.IGNORECASE)
        self.re_sum_mem = re.compile(r"Peak Memory Allocated\s*:\s*([\d\.]+)\s*MB", re.IGNORECASE)
        self.re_sum_warn = re.compile(r"Total Warnings Logged\s*:\s*(\d+)", re.IGNORECASE)
        self.re_sum_err = re.compile(r"Total Errors Logged\s*:\s*(\d+)", re.IGNORECASE)
        self.re_sum_exit = re.compile(r"Exit Code\s*:\s*(\d+)", re.IGNORECASE)

    def parse_file(self, filepath: str) -> ParsedLogRecord:
        """Stream a file line-by-line without loading entire file into memory."""
        job_id = os.path.splitext(os.path.basename(filepath))[0]
        if not os.path.exists(filepath):
            record = ParsedLogRecord(job_id=job_id, stage=self.stage)
            record.parse_warnings.append(f"File not found: {filepath}")
            return record

        def line_generator() -> Iterable[str]:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    yield line

        return self.parse_stream(line_generator(), default_job_id=job_id)

    @abstractmethod
    def validate_stage_specifics(self, record: ParsedLogRecord) -> None:
        """Validate stage-specific requirements or syntax."""
        pass

    def parse_stream(
        self, stream: Iterable[str], default_job_id: str = "JOB-UNKNOWN"
    ) -> ParsedLogRecord:
        """Parse log lines in streaming fashion."""
        record = ParsedLogRecord(job_id=default_job_id, stage=self.stage)
        has_summary_header = False
        saw_eof_marker = False
        current_step_name = "INIT"
        step_index_map: Dict[int, str] = {}
        error_snippets: List[str] = []

        line_num = 0
        for raw_line in stream:
            line_num += 1
            line = raw_line.strip()
            if not line:
                continue

            # Check header items
            if line.startswith("#"):
                m_stg = self.re_header_stage.search(line)
                if m_stg:
                    header_stage = m_stg.group(1).lower()
                    if header_stage in ("synth", "synthesis"):
                        record.stage = "synthesis"
                    elif header_stage in ("pnr", "place_and_route"):
                        record.stage = "place_and_route"
                    elif header_stage in ("vrf", "verification"):
                        record.stage = "verification"

                m_ver = self.re_header_ver.search(line)
                if m_ver:
                    record.tool_version = m_ver.group(1)

                m_des = self.re_header_design.search(line)
                if m_des:
                    record.design_name = m_des.group(1)
                    record.design_size = int(m_des.group(2))

                m_inv = self.re_header_invoc.search(line)
                if m_inv:
                    record.threads = int(m_inv.group(1))
                continue

            # Step boundary
            m_step = self.re_step_start.search(line)
            if m_step:
                s_idx = int(m_step.group(1))
                current_step_name = m_step.group(3)
                step_index_map[s_idx] = current_step_name
                continue

            # Step timing
            m_time = self.re_step_timing.search(line)
            if m_time:
                s_idx = int(m_time.group(1))
                cpu_s = float(m_time.group(2))
                wall_s = float(m_time.group(3))
                mem_mb = float(m_time.group(4))
                s_name = step_index_map.get(s_idx, f"STEP_{s_idx}")
                record.steps.append(
                    StepTiming(
                        step_index=s_idx,
                        step_name=s_name,
                        cpu_s=cpu_s,
                        wall_s=wall_s,
                        mem_mb=mem_mb,
                    )
                )
                continue

            # Summary markers
            if "EXECUTION SUMMARY" in line:
                has_summary_header = True
                continue

            if "End of Log File." in line:
                saw_eof_marker = True
                continue

            if has_summary_header:
                m_st = self.re_sum_status.search(line)
                if m_st:
                    record.status = m_st.group(1).lower()
                m_cau = self.re_sum_cause.search(line)
                if m_cau:
                    record.failure_cause = m_cau.group(1).lower()
                m_wal = self.re_sum_wall.search(line)
                if m_wal:
                    record.runtime_s = float(m_wal.group(1))
                m_me = self.re_sum_mem.search(line)
                if m_me:
                    record.peak_memory_mb = float(m_me.group(1))
                m_wa = self.re_sum_warn.search(line)
                if m_wa:
                    record.warnings_count = int(m_wa.group(1))
                m_er = self.re_sum_err.search(line)
                if m_er:
                    record.errors_count = int(m_er.group(1))
                m_ex = self.re_sum_exit.search(line)
                if m_ex:
                    record.exit_code = int(m_ex.group(1))
                continue

            # Messages parsing (errors, warnings, fatal, info)
            # Try Modern
            m_mod = self.re_msg_modern.search(line)
            if m_mod:
                sev = "WARN" if "WARN" in m_mod.group(1).upper() else m_mod.group(1).upper()
                msg_id = m_mod.group(2)
                txt = m_mod.group(3)
                self._record_message(record, line_num, sev, msg_id, txt, error_snippets)
                continue

            # Try Bracket Pair: [WARN] [TIM-104]
            m_bp = self.re_msg_bracket_pair.search(line)
            if m_bp:
                sev = "WARN" if "WARN" in m_bp.group(1).upper() else m_bp.group(1).upper()
                msg_id = m_bp.group(2)
                txt = m_bp.group(3)
                self._record_message(record, line_num, sev, msg_id, txt, error_snippets)
                continue

            # Try Hyphen: [INFO-012]
            m_hy = self.re_msg_hyphen.search(line)
            if m_hy:
                sev = "WARN" if "WARN" in m_hy.group(1).upper() else m_hy.group(1).upper()
                msg_id = f"{m_hy.group(1)}-{m_hy.group(2)}"
                txt = m_hy.group(3)
                self._record_message(record, line_num, sev, msg_id, txt, error_snippets)
                continue

            # Try Bracket
            m_brk = self.re_msg_bracket.search(line)
            if m_brk:
                raw_sev = m_brk.group(1) or m_brk.group(3)
                sev = "WARN" if "WARN" in raw_sev.upper() else raw_sev.upper()
                msg_id = m_brk.group(2) or m_brk.group(4)
                txt = m_brk.group(5)
                self._record_message(record, line_num, sev, msg_id, txt, error_snippets)
                continue

            # Try Legacy
            m_leg = self.re_msg_legacy.search(line)
            if m_leg:
                sev = "WARN" if "WARN" in m_leg.group(1).upper() else m_leg.group(1).upper()
                msg_id = m_leg.group(2)
                txt = m_leg.group(3)
                self._record_message(record, line_num, sev, msg_id, txt, error_snippets)
                continue

            # Fallback error line without structured message ID
            m_fb = self.re_msg_fallback.search(line)
            if m_fb:
                sev = "WARN" if "WARN" in m_fb.group(1).upper() else m_fb.group(1).upper()
                txt = m_fb.group(2)
                # Synthetic unique ID based on keywords
                kw = "GEN"
                if "mem" in txt.lower() or "cgroup" in txt.lower():
                    kw = "MEM-RAW"
                elif "timeout" in txt.lower() or "watchdog" in txt.lower():
                    kw = "TIME-RAW"
                elif "syntax" in txt.lower() or "token" in txt.lower():
                    kw = "HDL-RAW"
                elif "licen" in txt.lower():
                    kw = "LIC-RAW"
                elif "congest" in txt.lower():
                    kw = "CONG-RAW"
                elif "slack" in txt.lower() or "transition" in txt.lower():
                    kw = "TIM-RAW"
                self._record_message(record, line_num, sev, kw, txt, error_snippets)
                continue

        # Post-parse validation and fallback recovery for truncated / corrupted logs
        if not has_summary_header or not saw_eof_marker:
            record.parse_warnings.append("Log appears truncated or execution summary is missing.")
            # Recover runtime and memory from recorded steps if missing
            if record.steps:
                last_step = record.steps[-1]
                if record.runtime_s == 0.0:
                    record.runtime_s = round(last_step.wall_s, 2)
                if record.peak_memory_mb == 0.0:
                    record.peak_memory_mb = max(s.mem_mb for s in record.steps)
            if record.status == "unknown":
                # If error messages were present, mark failed, else incomplete
                record.status = "failed" if record.errors_count > 0 else "unknown"

        record.raw_error_text = " ".join(error_snippets)
        self.validate_stage_specifics(record)
        return record

    def _record_message(
        self,
        record: ParsedLogRecord,
        line_num: int,
        severity: str,
        message_id: str,
        text: str,
        error_snippets: List[str],
    ) -> None:
        msg = LogMessage(
            line_number=line_num,
            severity=severity,
            message_id=message_id,
            text=text,
        )
        record.messages.append(msg)
        record.raw_message_ids.append(message_id)

        if severity in ("ERROR", "FATAL"):
            record.errors_count += 1
            error_snippets.append(f"{message_id} {text}")
        elif severity in ("WARN", "WARNING"):
            record.warnings_count += 1
            if len(error_snippets) < 15:
                error_snippets.append(f"{message_id} {text}")


class SynthesisLogParser(LogParser):
    def __init__(self) -> None:
        super().__init__(stage="synthesis")

    def validate_stage_specifics(self, record: ParsedLogRecord) -> None:
        if record.stage != "synthesis":
            record.parse_warnings.append(
                f"Expected synthesis stage but header indicated {record.stage}"
            )


class PlaceAndRouteLogParser(LogParser):
    def __init__(self) -> None:
        super().__init__(stage="place_and_route")

    def validate_stage_specifics(self, record: ParsedLogRecord) -> None:
        if record.stage != "place_and_route":
            record.parse_warnings.append(
                f"Expected place_and_route stage but header indicated {record.stage}"
            )


class VerificationLogParser(LogParser):
    def __init__(self) -> None:
        super().__init__(stage="verification")

    def validate_stage_specifics(self, record: ParsedLogRecord) -> None:
        if record.stage != "verification":
            record.parse_warnings.append(
                f"Expected verification stage but header indicated {record.stage}"
            )


def get_parser(stage: str) -> LogParser:
    """Factory returning the parser corresponding to the specified stage."""
    normalized = stage.lower().replace("-", "_").replace(" ", "_")
    if normalized in ("synthesis", "synth"):
        return SynthesisLogParser()
    elif normalized in ("place_and_route", "pnr"):
        return PlaceAndRouteLogParser()
    elif normalized in ("verification", "verif", "sim"):
        return VerificationLogParser()
    raise ValueError(f"Unknown stage: {stage}")


def parse_log_file(filepath: str, stage_hint: Optional[str] = None) -> ParsedLogRecord:
    """Detect stage from file header or hint and parse streaming."""
    # Sniff first 15 lines for subsystem if not provided
    stage = stage_hint
    if not stage and os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                m = re.search(r"Subsystem:\s*([A-Za-z0-9_\-]+)", line, re.IGNORECASE)
                if m:
                    stage = m.group(1).lower()
                    break

    if not stage:
        stage = "synthesis"  # Default fallback
    parser = get_parser(stage)
    return parser.parse_file(filepath)
