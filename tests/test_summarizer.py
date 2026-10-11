"""Unit tests for Phase 6 Optional LLM Log Summarizer.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

from flowinsight.parsers import LogMessage, ParsedLogRecord
from flowinsight.summarizer import LLMLogSummarizer


class MockLLMClient:
    """Mock client simulating an external LLM provider without cloud network calls."""

    def __init__(self):
        self.last_prompt = ""
        self.call_count = 0

    def generate(self, prompt: str, model: str):
        self.last_prompt = prompt
        self.call_count += 1
        return {
            "summary": "Mock diagnosis: The execution suffered a critical memory ceiling breach during graph partitioning.",
            "next_step": "Mock recommendation: Increase compute host cgroup RAM quota to 8GB or decrease parallel thread workers.",
        }


def test_summarizer_disabled_by_default():
    """Verify summarizer is strictly disabled by default without API key."""
    summarizer = LLMLogSummarizer(api_key="")
    assert not summarizer.is_enabled

    res = summarizer.summarize_log("ERROR: [MEM-902] Fatal memory allocation failure")
    assert res["enabled"] is False
    assert "disabled by default" in res["summary"].lower()
    assert res["data_source"] == "synthetic"


def test_summarizer_with_mocked_client_and_path_redaction():
    """Verify enabled summarizer extracts only error lines, redacts paths, and queries mocked client."""
    mock_client = MockLLMClient()
    summarizer = LLMLogSummarizer(api_key="mock-key-123", client=mock_client)
    assert summarizer.is_enabled

    record = ParsedLogRecord(
        job_id="JOB-PNR-TEST",
        stage="place_and_route",
        messages=[
            LogMessage(
                line_number=10,
                severity="INFO",
                message_id="INFO-001",
                text="Entering initialization module",
            ),
            LogMessage(
                line_number=45,
                severity="ERROR",
                message_id="LIB-204",
                text="Cannot resolve cell INV_X4 in file /mnt/eda/proprietary_designs/riscv/tech.db",
            ),
            LogMessage(
                line_number=60,
                severity="FATAL",
                message_id="LIB-901",
                text="Technology file missing at C:\\workspace\\tapeout\\libs\\macro.lef",
            ),
        ],
    )

    res = summarizer.summarize_log(record, redact_paths=True)
    assert res["enabled"] is True
    assert mock_client.call_count == 1

    # Verify that only error lines were sent, and INFO was suppressed
    sent_prompt = mock_client.last_prompt
    assert "Entering initialization module" not in sent_prompt
    assert "LIB-204" in sent_prompt
    assert "LIB-901" in sent_prompt

    # Verify sensitive paths were redacted
    assert "/mnt/eda/proprietary_designs" not in sent_prompt
    assert "C:\\workspace\\tapeout" not in sent_prompt
    assert "[REDACTED_PATH]" in sent_prompt

    # Verify mock response returned
    assert "Mock diagnosis" in res["summary"]
    assert "Mock recommendation" in res["suggested_next_step"]
    assert res["data_source"] == "synthetic"
