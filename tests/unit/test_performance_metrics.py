# -*- coding: utf-8 -*-

"""Unit tests for content-free upstream performance metrics."""

from unittest.mock import patch

import httpx

from kiro.performance_metrics import (
    UpstreamPerformanceMetrics,
    attach_upstream_metrics,
    count_cache_points,
    create_upstream_metrics,
    extract_model_id,
    get_upstream_metrics,
)


def _payload(model: str = "claude-sonnet-5") -> dict:
    """Build a minimal Kiro payload containing a model ID and secret content."""
    return {
        "conversationState": {
            "currentMessage": {
                "userInputMessage": {
                    "modelId": model,
                    "content": "TOP_SECRET_PROMPT_TEXT",
                }
            }
        }
    }


class TestPerformanceMetricCreation:
    """Tests for generation-request metric creation and safe model extraction."""

    def test_extract_model_id_handles_valid_and_malformed_payloads(self):
        """
        What it does: Extracts only the outbound model identifier.
        Purpose: Keep instrumentation robust without reading request content.
        """
        assert extract_model_id(_payload()) == "claude-sonnet-5"
        assert extract_model_id(None) == "unknown"
        assert extract_model_id({}) == "unknown"
        assert extract_model_id({"conversationState": []}) == "unknown"

    def test_count_cache_points_handles_nested_payloads(self):
        """
        What it does: Counts cache markers across system, message, and tool paths.
        Purpose: Prove cache intent reached Kiro without logging prompt content.
        """
        payload = {
            "history": [{"userInputMessage": {"cachePoint": {"type": "default"}}}],
            "tools": [
                {"name": "tool"},
                {"cachePoint": {"type": "default"}},
            ],
            "current": {"cachePoint": {"type": "default"}},
        }

        assert count_cache_points(payload) == 3
        assert count_cache_points(None) == 0

    def test_create_metrics_only_for_generation_endpoint(self):
        """
        What it does: Instruments model generation but skips control-plane calls.
        Purpose: Avoid noisy model-list and authentication timing logs.
        """
        metrics = create_upstream_metrics(
            method="post",
            url="https://runtime.example/generateAssistantResponse",
            payload=_payload(),
            stream=True,
            attempt=1,
            max_attempts=3,
            payload_bytes=321,
        )
        skipped = create_upstream_metrics(
            method="GET",
            url="https://management.example/List-Available-Models",
            payload=None,
            stream=False,
            attempt=1,
            max_attempts=3,
            payload_bytes=0,
        )

        assert metrics is not None
        assert metrics.model == "claude-sonnet-5"
        assert metrics.payload_bytes == 321
        assert metrics.cache_points == 0
        assert skipped is None


class TestPerformanceMetricLogging:
    """Tests for timing, usage, idempotency, and content-free log output."""

    def test_logs_headers_first_byte_completion_and_cache_usage_without_content(self):
        """
        What it does: Logs every requested metric using deterministic timings.
        Purpose: Verify observability while preventing prompt leakage.
        """
        metrics = UpstreamPerformanceMetrics(
            method="POST",
            endpoint="/generateAssistantResponse",
            model="claude-sonnet-5",
            stream=True,
            attempt=1,
            max_attempts=3,
            payload_bytes=456,
            cache_points=2,
            trace_id="trace123",
            started_at=10.0,
        )

        with (
            patch("kiro.performance_metrics.time.perf_counter", side_effect=[10.2, 10.7, 11.5]),
            patch("kiro.performance_metrics.logger.info") as mock_log,
        ):
            metrics.record_headers(200)
            metrics.record_first_byte()
            metrics.record_usage(
                {
                    "cacheReadInputTokens": 120,
                    "cacheWriteInputTokens": 20,
                    "cacheCreationInputTokens": 30,
                }
            )
            metrics.complete("success")

        messages = "\n".join(call.args[0] for call in mock_log.call_args_list)
        assert "phase=headers" in messages
        assert "headers_ms=200.0" in messages
        assert "phase=first_byte" in messages
        assert "first_byte_ms=700.0" in messages
        assert "phase=complete" in messages
        assert "stream_ms=1300.0" in messages
        assert "total_ms=1500.0" in messages
        assert "payload_bytes=456" in messages
        assert "cache_points=2" in messages
        assert "cache_read_input_tokens=120" in messages
        assert "cache_write_input_tokens=20" in messages
        assert "cache_creation_input_tokens=30" in messages
        assert "TOP_SECRET_PROMPT_TEXT" not in messages

    def test_usage_accepts_snake_case_and_ignores_invalid_values(self):
        """
        What it does: Handles both upstream naming styles and malformed counters.
        Purpose: Prevent misleading cache metrics from invalid usage events.
        """
        metrics = UpstreamPerformanceMetrics(
            method="POST",
            endpoint="/generateAssistantResponse",
            model="claude-sonnet-5",
            stream=True,
            attempt=1,
            max_attempts=1,
            payload_bytes=1,
        )
        metrics.record_usage(
            {
                "cache_read_input_tokens": 12.8,
                "cache_write_input_tokens": True,
                "cache_creation_input_tokens": "34",
            }
        )

        assert metrics.cache_read_input_tokens == 12
        assert metrics.cache_write_input_tokens is None
        assert metrics.cache_creation_input_tokens is None

    def test_first_byte_and_completion_logs_are_idempotent(self):
        """
        What it does: Ignores duplicate parser callbacks.
        Purpose: Emit one stable first-byte and completion record per attempt.
        """
        metrics = UpstreamPerformanceMetrics(
            method="POST",
            endpoint="/generateAssistantResponse",
            model="claude-sonnet-5",
            stream=True,
            attempt=1,
            max_attempts=1,
            payload_bytes=1,
        )
        with patch("kiro.performance_metrics.logger.info") as mock_log:
            metrics.record_first_byte()
            metrics.record_first_byte()
            metrics.complete("success")
            metrics.complete("stream_error", "RuntimeError")

        phases = [call.args[0].split()[1] for call in mock_log.call_args_list]
        assert phases.count("phase=first_byte") == 1
        assert phases.count("phase=complete") == 1


class TestPerformanceMetricResponseAttachment:
    """Tests for passing metrics from httpx request code to the stream parser."""

    def test_attach_and_get_metrics_round_trip(self):
        """
        What it does: Stores metrics in the response extension dictionary.
        Purpose: Avoid global mutable state across concurrent requests.
        """
        response = httpx.Response(200)
        metrics = UpstreamPerformanceMetrics(
            method="POST",
            endpoint="/generateAssistantResponse",
            model="claude-sonnet-5",
            stream=True,
            attempt=1,
            max_attempts=1,
            payload_bytes=1,
        )

        attach_upstream_metrics(response, metrics)

        assert get_upstream_metrics(response) is metrics
        assert get_upstream_metrics(httpx.Response(200)) is None
