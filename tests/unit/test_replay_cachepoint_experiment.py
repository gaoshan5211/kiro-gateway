# -*- coding: utf-8 -*-

"""
Unit tests for the standalone cachePoint replay experiment script.

The replay script must keep network execution opt-in, so these tests cover only
offline payload mutation and stream metric parsing.
"""

import copy
import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "replay_cachepoint_experiment.py"
)


def load_replay_module():
    """
    Load the standalone replay script as a module for unit testing.

    Returns:
        Imported replay script module.
    """
    spec = importlib.util.spec_from_file_location(
        "replay_cachepoint_experiment",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestCachePointPayloadMutation:
    """Tests for transforming historical debug payloads with Kiro cachePoint markers."""

    def test_adds_cache_points_for_system_current_message_and_tools(self):
        """
        What it does: Adds cachePoint markers where source cache_control exists.
        Purpose: Verify the script tests the suspected Kiro-native cachePoint path.
        """
        print("Setup: Anthropic request with system, current message, and tool cache_control...")
        replay = load_replay_module()
        client_request = {
            "system": [
                {"type": "text", "text": "system A"},
                {
                    "type": "text",
                    "text": "system B",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "messages": [
                {"role": "user", "content": "first user"},
                {"role": "assistant", "content": "first answer"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "current user",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "name": "write_file",
                    "description": "Write a file",
                    "input_schema": {"type": "object"},
                },
            ],
        }
        kiro_payload = {
            "conversationState": {
                "conversationId": "conversation-1",
                "history": [
                    {"userInputMessage": {"content": "system A\nsystem B\n\nfirst user"}},
                    {"assistantResponseMessage": {"content": "first answer"}},
                ],
                "currentMessage": {
                    "userInputMessage": {
                        "content": "current user",
                        "modelId": "claude-sonnet-5",
                        "origin": "AI_EDITOR",
                        "userInputMessageContext": {
                            "tools": [
                                {
                                    "toolSpecification": {
                                        "name": "read_file",
                                        "description": "Read a file",
                                        "inputSchema": {"json": {"type": "object"}},
                                    }
                                },
                                {
                                    "toolSpecification": {
                                        "name": "write_file",
                                        "description": "Write a file",
                                        "inputSchema": {"json": {"type": "object"}},
                                    }
                                },
                            ]
                        },
                    }
                },
            },
            "profileArn": "arn:aws:codewhisperer:us-east-1:123:profile/test",
        }
        original_payload = copy.deepcopy(kiro_payload)

        print("Action: Applying cachePoint markers...")
        mutated_payload, report = replay.apply_cachepoint_markers(
            client_request,
            kiro_payload,
        )

        print("Checking: Original payload is not mutated...")
        assert kiro_payload == original_payload

        history_user = mutated_payload["conversationState"]["history"][0][
            "userInputMessage"
        ]
        current_user = mutated_payload["conversationState"]["currentMessage"][
            "userInputMessage"
        ]
        tools = current_user["userInputMessageContext"]["tools"]

        print("Checking: system cache_control maps to first Kiro user message...")
        assert history_user["cachePoint"] == {"type": "default"}

        print("Checking: current message cache_control maps to current userInputMessage...")
        assert current_user["cachePoint"] == {"type": "default"}

        print("Checking: tool cache_control inserts a Kiro tool cachePoint marker...")
        assert tools[0]["toolSpecification"]["name"] == "read_file"
        assert tools[1] == {"cachePoint": {"type": "default"}}
        assert tools[2]["toolSpecification"]["name"] == "write_file"

        print("Checking: report captures marker paths...")
        assert report["cache_controls_seen"] == 3
        assert report["markers_added"] == 3
        assert "conversationState.history.0.userInputMessage.cachePoint" in report["paths"]
        assert (
            "conversationState.currentMessage.userInputMessage.cachePoint"
            in report["paths"]
        )
        assert (
            "conversationState.currentMessage.userInputMessage.userInputMessageContext.tools.1.cachePoint"
            in report["paths"]
        )

    def test_ignores_unsupported_cache_control_types(self):
        """
        What it does: Ignores cache_control types other than ephemeral.
        Purpose: Avoid sending guessed Kiro cachePoint markers for unknown semantics.
        """
        print("Setup: Unsupported cache_control type...")
        replay = load_replay_module()
        client_request = {
            "system": [
                {
                    "type": "text",
                    "text": "system",
                    "cache_control": {"type": "persistent"},
                }
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }
        kiro_payload = {
            "conversationState": {
                "currentMessage": {
                    "userInputMessage": {
                        "content": "system\n\nhello",
                        "modelId": "claude-sonnet-5",
                    }
                }
            }
        }

        print("Action: Applying cachePoint markers...")
        mutated_payload, report = replay.apply_cachepoint_markers(
            client_request,
            kiro_payload,
        )

        print("Checking: No cachePoint marker was added...")
        current_user = mutated_payload["conversationState"]["currentMessage"][
            "userInputMessage"
        ]
        assert "cachePoint" not in current_user
        assert report["cache_controls_seen"] == 1
        assert report["markers_added"] == 0
        assert report["unsupported_cache_controls"] == [
            "system.0.cache_control.type=persistent"
        ]


class TestResponseMetricExtraction:
    """Tests for parsing raw Kiro stream bytes into experiment metrics."""

    def test_extracts_credits_context_usage_and_cache_token_fields(self):
        """
        What it does: Extracts credit and cache-token metrics from raw stream bytes.
        Purpose: Let replay results distinguish real upstream cache from local usage display.
        """
        print("Setup: Raw stream bytes with Kiro event prefixes and token usage metadata...")
        replay = load_replay_module()
        raw_stream = (
            b'\x00contextUsageEvent\x00event{"contextUsagePercentage":12.5}'
            b'\x00meteringEvent\x00event{"unit":"credit","unitPlural":"credits","usage":0.123}'
            b'\x00metadataEvent\x00event{"metadata":{"tokenUsage":{"uncachedInputTokens":100,'
            b'"cacheReadInputTokens":80,"cacheWriteInputTokens":20,"outputTokens":5}}}'
        )

        print("Action: Extracting stream metrics...")
        metrics = replay.extract_stream_metrics(raw_stream)

        print("Checking: Credits and context usage were extracted...")
        assert metrics["credits"] == [0.123]
        assert metrics["latest_credit"] == 0.123
        assert metrics["context_usage_percentages"] == [12.5]
        assert metrics["latest_context_usage_percentage"] == 12.5

        print("Checking: Cache token fields were extracted without inventing values...")
        assert metrics["token_usage_events"] == [
            {
                "uncachedInputTokens": 100,
                "cacheReadInputTokens": 80,
                "cacheWriteInputTokens": 20,
                "outputTokens": 5,
            }
        ]
        assert metrics["latest_token_usage"]["cacheReadInputTokens"] == 80
