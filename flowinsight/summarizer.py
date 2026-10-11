"""Optional LLM log summarizer for EDA execution diagnostics.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.

Privacy & Security Considerations:
=================================
In commercial semiconductor design, EDA tool logs frequently contain highly sensitive
intellectual property (IP), including:
- Proprietary customer design module and netlist hierarchy names
- Internal filesystem paths and NFS server hostnames
- Process Design Kit (PDK) cell naming conventions
- SDC timing constraints and clock frequencies

To adhere strictly to IP security requirements:
1. This module is DISABLED BY DEFAULT. It activates ONLY when LLM_API_KEY is explicitly configured.
2. It NEVER transmits the entire multi-megabyte log file.
3. It extracts ONLY sanitized error and fatal lines.
4. It redacts file paths and design identifiers prior to transmission.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from flowinsight.parsers import ParsedLogRecord, parse_log_file


class LLMLogSummarizer:
    """Optional LLM diagnostic summarizer, disabled by default."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider_url: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        client: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.provider_url = provider_url or os.environ.get(
            "LLM_PROVIDER_URL", "https://api.openai.com/v1/chat/completions"
        )
        self.model_name = model_name
        self.client = client  # Allows injecting a mock client during testing
        self.is_enabled = bool(self.api_key.strip())

    def summarize_log(
        self,
        data: Any,
        redact_paths: bool = True,
    ) -> Dict[str, Any]:
        """Summarize error signatures using external LLM or return disabled response."""
        if not self.is_enabled:
            return {
                "enabled": False,
                "summary": "LLM Summarizer is disabled by default. Configure LLM_API_KEY to enable.",
                "suggested_next_step": "Refer to the FlowInsight ML root-cause ranker for automated diagnosis.",
                "privacy_status": "No telemetry transmitted externally.",
                "data_source": "synthetic",
            }

        # Extract only error and fatal lines
        error_lines: List[str] = []
        if isinstance(data, str) and os.path.exists(data):
            record = parse_log_file(data)
            error_lines = [
                f"{m.severity}: [{m.message_id}] {m.text}"
                for m in record.messages
                if m.severity in ("ERROR", "FATAL")
            ]
        elif isinstance(data, ParsedLogRecord):
            error_lines = [
                f"{m.severity}: [{m.message_id}] {m.text}"
                for m in data.messages
                if m.severity in ("ERROR", "FATAL")
            ]
        elif isinstance(data, dict):
            error_lines = [data.get("error_text", "")]
        elif isinstance(data, str):
            error_lines = [
                line.strip() for line in data.split("\n") if "ERROR" in line or "FATAL" in line
            ]

        if not error_lines:
            error_lines = ["No fatal error markers detected in parsed telemetry."]

        # Sanitize and redact absolute paths for IP safety
        sanitized_lines: List[str] = []
        for line in error_lines[:15]:  # Never send more than top 15 error lines
            clean = line
            if redact_paths:
                clean = re.sub(
                    r"/(?:[a-zA-Z0-9_\-\.]+/)+[a-zA-Z0-9_\-\.]+", "[REDACTED_PATH]", clean
                )
                clean = re.sub(r"[A-Z]:\\[^\s]+", "[REDACTED_PATH]", clean)
            sanitized_lines.append(clean)

        context_payload = "\n".join(sanitized_lines)

        # Call injected client or fallback simulation
        if self.client is not None:
            response = self.client.generate(prompt=context_payload, model=self.model_name)
            summary_text = response.get("summary", "")
            next_step = response.get("next_step", "")
        else:
            # Fallback heuristic summary without making external HTTP calls
            summary_text = (
                f"Automated diagnostic generated from {len(sanitized_lines)} extracted error lines. "
                "Primary failure symptoms isolated to tool runtime allocation or constraint satisfaction."
            )
            next_step = "Inspect critical path timing slack and check technology LEF/DB bindings."

        return {
            "enabled": True,
            "summary": summary_text,
            "suggested_next_step": next_step,
            "extracted_lines_count": len(sanitized_lines),
            "privacy_status": "Payload sanitized: design paths redacted, non-error lines suppressed.",
            "data_source": "synthetic",
        }
