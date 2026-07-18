# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Content-free upstream performance metrics for Kiro runtime requests."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

import httpx
from loguru import logger


PERFORMANCE_EXTENSION_KEY = "kiro_gateway.performance_metrics"
UNKNOWN_VALUE = "unknown"


def _duration_ms(started_at: Optional[float], finished_at: Optional[float]) -> str:
    """
    Format a monotonic-clock duration in milliseconds.

    Args:
        started_at: Start timestamp from ``time.perf_counter``.
        finished_at: End timestamp from ``time.perf_counter``.

    Returns:
        Milliseconds with one decimal place, or ``unknown`` when unavailable.
    """
    if started_at is None or finished_at is None:
        return UNKNOWN_VALUE
    return f"{max(0.0, (finished_at - started_at) * 1000):.1f}"


def _optional_integer(value: Any) -> Optional[int]:
    """
    Convert numeric usage values to integers without accepting booleans.

    Args:
        value: Upstream usage value.

    Returns:
        Integer value when numeric, otherwise None.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def extract_model_id(payload: Optional[Dict[str, Any]]) -> str:
    """
    Extract the outbound model identifier without logging request content.

    Args:
        payload: Kiro request payload.

    Returns:
        Model identifier, or ``unknown`` when the payload shape is incomplete.
    """
    if not isinstance(payload, dict):
        return UNKNOWN_VALUE

    conversation_state = payload.get("conversationState")
    if not isinstance(conversation_state, dict):
        return UNKNOWN_VALUE
    current_message = conversation_state.get("currentMessage")
    if not isinstance(current_message, dict):
        return UNKNOWN_VALUE
    user_input = current_message.get("userInputMessage")
    if not isinstance(user_input, dict):
        return UNKNOWN_VALUE
    model_id = user_input.get("modelId")
    return model_id if isinstance(model_id, str) and model_id else UNKNOWN_VALUE


def count_cache_points(value: Any) -> int:
    """
    Count Kiro ``cachePoint`` keys without retaining their surrounding content.

    Args:
        value: JSON-like Kiro payload fragment.

    Returns:
        Number of cachePoint keys in the payload.
    """
    if isinstance(value, dict):
        return (1 if "cachePoint" in value else 0) + sum(
            count_cache_points(child) for child in value.values()
        )
    if isinstance(value, list):
        return sum(count_cache_points(child) for child in value)
    return 0


@dataclass
class UpstreamPerformanceMetrics:
    """Track one Kiro upstream attempt without retaining request or response text."""

    method: str
    endpoint: str
    model: str
    stream: bool
    attempt: int
    max_attempts: int
    payload_bytes: int
    cache_points: int = 0
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.perf_counter)
    headers_at: Optional[float] = None
    first_byte_at: Optional[float] = None
    completed_at: Optional[float] = None
    status_code: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    cache_write_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    _completion_logged: bool = False

    def record_headers(self, status_code: int) -> None:
        """
        Record and log the upstream response-header latency.

        Args:
            status_code: Upstream HTTP status code.
        """
        if self.headers_at is not None:
            return
        self.headers_at = time.perf_counter()
        self.status_code = status_code
        logger.info(
            "[UpstreamPerformance] phase=headers "
            f"trace_id={self.trace_id} method={self.method} endpoint={self.endpoint} "
            f"model={self.model} stream={self.stream} attempt={self.attempt}/{self.max_attempts} "
            f"status={status_code} payload_bytes={self.payload_bytes} "
            f"cache_points={self.cache_points} "
            f"headers_ms={_duration_ms(self.started_at, self.headers_at)}"
        )

    def record_first_byte(self) -> None:
        """Record and log latency until the first upstream response-body byte."""
        if self.first_byte_at is not None:
            return
        self.first_byte_at = time.perf_counter()
        logger.info(
            "[UpstreamPerformance] phase=first_byte "
            f"trace_id={self.trace_id} model={self.model} stream={self.stream} "
            f"attempt={self.attempt}/{self.max_attempts} "
            f"first_byte_ms={_duration_ms(self.started_at, self.first_byte_at)}"
        )

    def record_usage(self, usage: Optional[Dict[str, Any]]) -> None:
        """
        Record upstream cache-token counters when a usage event provides them.

        Args:
            usage: Kiro usage event data.
        """
        if not isinstance(usage, dict):
            return

        mappings = (
            (("cacheReadInputTokens", "cache_read_input_tokens"), "cache_read_input_tokens"),
            (("cacheWriteInputTokens", "cache_write_input_tokens"), "cache_write_input_tokens"),
            (
                ("cacheCreationInputTokens", "cache_creation_input_tokens"),
                "cache_creation_input_tokens",
            ),
        )
        for source_keys, attribute_name in mappings:
            for source_key in source_keys:
                parsed_value = _optional_integer(usage.get(source_key))
                if parsed_value is not None:
                    setattr(self, attribute_name, parsed_value)
                    break

    def complete(self, outcome: str, error_type: Optional[str] = None) -> None:
        """
        Log the final duration and cache counters exactly once.

        Args:
            outcome: Stable result label such as ``success`` or ``first_byte_timeout``.
            error_type: Optional exception class name. Exception messages are never logged here.
        """
        if self._completion_logged:
            return
        self._completion_logged = True
        self.completed_at = time.perf_counter()
        error_value = error_type or "none"
        cache_read = (
            str(self.cache_read_input_tokens)
            if self.cache_read_input_tokens is not None
            else UNKNOWN_VALUE
        )
        cache_write = (
            str(self.cache_write_input_tokens)
            if self.cache_write_input_tokens is not None
            else UNKNOWN_VALUE
        )
        cache_creation = (
            str(self.cache_creation_input_tokens)
            if self.cache_creation_input_tokens is not None
            else UNKNOWN_VALUE
        )
        logger.info(
            "[UpstreamPerformance] phase=complete "
            f"trace_id={self.trace_id} model={self.model} stream={self.stream} "
            f"attempt={self.attempt}/{self.max_attempts} status={self.status_code or UNKNOWN_VALUE} "
            f"outcome={outcome} error_type={error_value} "
            f"headers_ms={_duration_ms(self.started_at, self.headers_at)} "
            f"first_byte_ms={_duration_ms(self.started_at, self.first_byte_at)} "
            f"stream_ms={_duration_ms(self.headers_at, self.completed_at)} "
            f"total_ms={_duration_ms(self.started_at, self.completed_at)} "
            f"payload_bytes={self.payload_bytes} "
            f"cache_points={self.cache_points} "
            f"cache_read_input_tokens={cache_read} "
            f"cache_write_input_tokens={cache_write} "
            f"cache_creation_input_tokens={cache_creation}"
        )


def create_upstream_metrics(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]],
    stream: bool,
    attempt: int,
    max_attempts: int,
    payload_bytes: int,
) -> Optional[UpstreamPerformanceMetrics]:
    """
    Create metrics only for model-generation requests.

    Args:
        method: HTTP method.
        url: Upstream URL.
        payload: Kiro request payload.
        stream: Whether httpx is reading the response as a stream.
        attempt: One-based retry attempt.
        max_attempts: Maximum retry attempts.
        payload_bytes: Exact serialized request-body byte count.

    Returns:
        Metrics object for GenerateAssistantResponse, otherwise None.
    """
    endpoint = urlsplit(url).path or "/"
    if not endpoint.lower().endswith("/generateassistantresponse"):
        return None
    return UpstreamPerformanceMetrics(
        method=method.upper(),
        endpoint=endpoint,
        model=extract_model_id(payload),
        stream=stream,
        attempt=attempt,
        max_attempts=max_attempts,
        payload_bytes=payload_bytes,
        cache_points=count_cache_points(payload),
    )


def attach_upstream_metrics(
    response: httpx.Response,
    metrics: Optional[UpstreamPerformanceMetrics],
) -> None:
    """
    Attach metrics to an httpx response for the stream parser.

    Args:
        response: Upstream HTTP response.
        metrics: Metrics object, or None for uninstrumented endpoints.
    """
    if metrics is None:
        return
    extensions = getattr(response, "extensions", None)
    if not isinstance(extensions, dict):
        extensions = {}
        response.extensions = extensions
    extensions[PERFORMANCE_EXTENSION_KEY] = metrics


def get_upstream_metrics(response: httpx.Response) -> Optional[UpstreamPerformanceMetrics]:
    """
    Read attached metrics from an httpx response.

    Args:
        response: Upstream HTTP response.

    Returns:
        Attached metrics object, otherwise None.
    """
    extensions = getattr(response, "extensions", None)
    if not isinstance(extensions, dict):
        return None
    metrics = extensions.get(PERFORMANCE_EXTENSION_KEY)
    return metrics if isinstance(metrics, UpstreamPerformanceMetrics) else None
