#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Replay historical Kiro debug payloads with experimental cachePoint markers.

This script is intentionally standalone and opt-in for network execution. By
default it performs a dry run: it reads debug_logs/requests, generates baseline
and cachePoint payloads, and writes them under debug_logs/cachepoint_replay.

Use --execute to actually send the generated payloads to Kiro.
"""

import argparse
import asyncio
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import uuid

import httpx
from fastapi import HTTPException


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


KIRO_CACHE_POINT: Dict[str, str] = {"type": "default"}
TOOL_CACHE_POINT: Dict[str, Dict[str, str]] = {"cachePoint": KIRO_CACHE_POINT}
TOKEN_USAGE_KEYS = {
    "uncachedInputTokens",
    "cacheReadInputTokens",
    "cacheWriteInputTokens",
    "cacheCreationInputTokens",
    "outputTokens",
    "inputTokenCount",
    "outputTokenCount",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
    "cache_creation_input_tokens",
    "uncached_input_tokens",
    "output_tokens",
}


def load_json_file(path: Path) -> Dict[str, Any]:
    """
    Load a JSON object from disk.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If the file does not contain a JSON object.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def write_json_file(path: Path, data: Any) -> None:
    """
    Write JSON data to disk with stable formatting.

    Args:
        path: Output path.
        data: JSON-serializable value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def format_json_path(path: Sequence[Any]) -> str:
    """
    Format a path inside a JSON document.

    Args:
        path: Sequence of path components.

    Returns:
        Dot-separated path string.
    """
    return ".".join(str(part) for part in path)


def iter_cache_control_entries(
    value: Any,
    path: Optional[List[Any]] = None,
) -> Iterable[Tuple[List[Any], Any]]:
    """
    Yield every cache_control entry in a JSON-like value.

    Args:
        value: JSON-like value to inspect.
        path: Current traversal path.

    Yields:
        Tuples of (path, cache_control value).
    """
    current_path = path or []
    if isinstance(value, dict):
        if "cache_control" in value:
            yield current_path + ["cache_control"], value["cache_control"]
        for key, child in value.items():
            if key == "cache_control":
                continue
            yield from iter_cache_control_entries(child, current_path + [key])
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_cache_control_entries(child, current_path + [index])


def is_supported_cache_control(cache_control: Any) -> bool:
    """
    Check whether an Anthropic cache_control value maps to Kiro cachePoint.

    Args:
        cache_control: cache_control value from the client request.

    Returns:
        True when the value is the known ephemeral cache_control type.
    """
    return (
        isinstance(cache_control, dict)
        and cache_control.get("type") == "ephemeral"
    )


def describe_unsupported_cache_control(path: Sequence[Any], cache_control: Any) -> str:
    """
    Describe an unsupported cache_control value for diagnostics.

    Args:
        path: Path to the cache_control field.
        cache_control: Unsupported value.

    Returns:
        Human-readable diagnostic string.
    """
    path_text = format_json_path(path)
    if isinstance(cache_control, dict) and "type" in cache_control:
        return f"{path_text}.type={cache_control.get('type')}"
    return f"{path_text}=unsupported"


def collect_supported_cache_control_paths(value: Any, root_path: List[Any]) -> List[str]:
    """
    Collect supported cache_control paths under a value.

    Args:
        value: JSON-like value to inspect.
        root_path: Root path for diagnostics.

    Returns:
        List of supported cache_control path strings.
    """
    return [
        format_json_path(path)
        for path, cache_control in iter_cache_control_entries(value, root_path)
        if is_supported_cache_control(cache_control)
    ]


def has_supported_cache_control(value: Any, root_path: List[Any]) -> bool:
    """
    Check if a value contains a supported cache_control entry.

    Args:
        value: JSON-like value to inspect.
        root_path: Root path for diagnostics.

    Returns:
        True when at least one supported cache_control exists.
    """
    return bool(collect_supported_cache_control_paths(value, root_path))


def build_mutation_report(client_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the initial mutation report from the original client request.

    Args:
        client_request: Original client request_body.json content.

    Returns:
        Mutable report dictionary.
    """
    entries = list(iter_cache_control_entries(client_request, []))
    unsupported = [
        describe_unsupported_cache_control(path, cache_control)
        for path, cache_control in entries
        if not is_supported_cache_control(cache_control)
    ]
    return {
        "cache_controls_seen": len(entries),
        "supported_cache_controls": [
            format_json_path(path)
            for path, cache_control in entries
            if is_supported_cache_control(cache_control)
        ],
        "unsupported_cache_controls": unsupported,
        "markers_added": 0,
        "paths": [],
    }


def add_message_cache_point(
    message: Dict[str, Any],
    report: Dict[str, Any],
    path: str,
) -> None:
    """
    Add a cachePoint property to a Kiro message object.

    Args:
        message: Kiro userInputMessage or assistantResponseMessage.
        report: Mutation report to update.
        path: JSON path to the cachePoint location.
    """
    if "cachePoint" in message:
        return
    message["cachePoint"] = copy.deepcopy(KIRO_CACHE_POINT)
    report["markers_added"] += 1
    report["paths"].append(path)


def get_conversation_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the Kiro conversationState object.

    Args:
        payload: Kiro payload.

    Returns:
        conversationState dictionary, or an empty dictionary if absent.
    """
    state = payload.get("conversationState")
    return state if isinstance(state, dict) else {}


def get_current_user_input(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Return currentMessage.userInputMessage from a Kiro payload.

    Args:
        payload: Kiro payload.

    Returns:
        Current userInputMessage dictionary, if present.
    """
    state = get_conversation_state(payload)
    current = state.get("currentMessage")
    if not isinstance(current, dict):
        return None
    user_input = current.get("userInputMessage")
    return user_input if isinstance(user_input, dict) else None


def get_history(payload: Dict[str, Any]) -> List[Any]:
    """
    Return the Kiro history list from a payload.

    Args:
        payload: Kiro payload.

    Returns:
        History list, or an empty list if absent.
    """
    history = get_conversation_state(payload).get("history")
    return history if isinstance(history, list) else []


def get_first_user_message_target(
    payload: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Find the first Kiro userInputMessage that carries system text.

    Args:
        payload: Kiro payload.

    Returns:
        Tuple of (message dictionary, cachePoint JSON path).
    """
    for index, entry in enumerate(get_history(payload)):
        if not isinstance(entry, dict):
            continue
        user_input = entry.get("userInputMessage")
        if isinstance(user_input, dict):
            return (
                user_input,
                f"conversationState.history.{index}.userInputMessage.cachePoint",
            )

    current_user = get_current_user_input(payload)
    if current_user is not None:
        return (
            current_user,
            "conversationState.currentMessage.userInputMessage.cachePoint",
        )

    return None, None


def get_history_message_target(
    payload: Dict[str, Any],
    conversation_index: int,
    role: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Map a client conversation message index to a Kiro history target.

    Args:
        payload: Kiro payload.
        conversation_index: Index among non-system client messages.
        role: Client message role.

    Returns:
        Tuple of (message dictionary, cachePoint JSON path).
    """
    history = get_history(payload)
    if conversation_index >= len(history):
        return None, None

    entry = history[conversation_index]
    if not isinstance(entry, dict):
        return None, None

    if role == "assistant":
        assistant_message = entry.get("assistantResponseMessage")
        if isinstance(assistant_message, dict):
            return (
                assistant_message,
                (
                    f"conversationState.history.{conversation_index}"
                    ".assistantResponseMessage.cachePoint"
                ),
            )
    else:
        user_message = entry.get("userInputMessage")
        if isinstance(user_message, dict):
            return (
                user_message,
                (
                    f"conversationState.history.{conversation_index}"
                    ".userInputMessage.cachePoint"
                ),
            )

    return None, None


def apply_system_cache_points(
    client_request: Dict[str, Any],
    payload: Dict[str, Any],
    report: Dict[str, Any],
) -> None:
    """
    Add a cachePoint for top-level and inline system cache controls.

    Args:
        client_request: Original client request.
        payload: Mutable Kiro payload copy.
        report: Mutation report to update.
    """
    needs_system_marker = has_supported_cache_control(
        client_request.get("system"),
        ["system"],
    )

    for index, message in enumerate(client_request.get("messages", [])):
        if not isinstance(message, dict) or message.get("role") != "system":
            continue
        if has_supported_cache_control(message, ["messages", index]):
            needs_system_marker = True
            break

    if not needs_system_marker:
        return

    target, path = get_first_user_message_target(payload)
    if target is not None and path is not None:
        add_message_cache_point(target, report, path)


def apply_message_cache_points(
    client_request: Dict[str, Any],
    payload: Dict[str, Any],
    report: Dict[str, Any],
) -> None:
    """
    Add cachePoint markers for non-system client messages.

    Args:
        client_request: Original client request.
        payload: Mutable Kiro payload copy.
        report: Mutation report to update.
    """
    conversation_messages: List[Tuple[int, Dict[str, Any]]] = []
    for original_index, message in enumerate(client_request.get("messages", [])):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "system":
            continue
        conversation_messages.append((original_index, message))

    last_conversation_index = len(conversation_messages) - 1
    for conversation_index, (original_index, message) in enumerate(conversation_messages):
        if not has_supported_cache_control(message, ["messages", original_index]):
            continue

        role = str(message.get("role", "user"))
        if conversation_index == last_conversation_index:
            target = get_current_user_input(payload)
            path = "conversationState.currentMessage.userInputMessage.cachePoint"
        else:
            target, path = get_history_message_target(payload, conversation_index, role)

        if target is not None and path is not None:
            add_message_cache_point(target, report, path)


def is_tool_cache_point_marker(value: Any) -> bool:
    """
    Check if a Kiro tools entry is already a cachePoint marker.

    Args:
        value: Kiro tools list entry.

    Returns:
        True when the entry is a cachePoint marker.
    """
    return isinstance(value, dict) and isinstance(value.get("cachePoint"), dict)


def get_kiro_tool_name(value: Any) -> Optional[str]:
    """
    Extract the toolSpecification name from a Kiro tools entry.

    Args:
        value: Kiro tools list entry.

    Returns:
        Tool name, if present.
    """
    if not isinstance(value, dict):
        return None
    spec = value.get("toolSpecification")
    if not isinstance(spec, dict):
        return None
    name = spec.get("name")
    return name if isinstance(name, str) else None


def apply_tool_cache_points(
    client_request: Dict[str, Any],
    payload: Dict[str, Any],
    report: Dict[str, Any],
) -> None:
    """
    Add Kiro tool cachePoint markers for cached Anthropic tools.

    Args:
        client_request: Original client request.
        payload: Mutable Kiro payload copy.
        report: Mutation report to update.
    """
    cached_tool_names = set()
    for index, tool in enumerate(client_request.get("tools", []) or []):
        if not isinstance(tool, dict):
            continue
        if has_supported_cache_control(tool, ["tools", index]):
            name = tool.get("name")
            if isinstance(name, str) and name:
                cached_tool_names.add(name)

    if not cached_tool_names:
        return

    current_user = get_current_user_input(payload)
    if current_user is None:
        return

    context = current_user.get("userInputMessageContext")
    if not isinstance(context, dict):
        return

    tools = context.get("tools")
    if not isinstance(tools, list):
        return

    new_tools: List[Any] = []
    for index, item in enumerate(tools):
        new_tools.append(item)
        tool_name = get_kiro_tool_name(item)
        if tool_name not in cached_tool_names:
            continue

        next_item = tools[index + 1] if index + 1 < len(tools) else None
        if is_tool_cache_point_marker(next_item):
            continue

        marker_index = len(new_tools)
        new_tools.append(copy.deepcopy(TOOL_CACHE_POINT))
        report["markers_added"] += 1
        report["paths"].append(
            (
                "conversationState.currentMessage.userInputMessage."
                f"userInputMessageContext.tools.{marker_index}.cachePoint"
            )
        )

    context["tools"] = new_tools


def apply_cachepoint_markers(
    client_request: Dict[str, Any],
    kiro_payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build an experimental Kiro payload with cachePoint markers.

    Args:
        client_request: Original debug request_body.json.
        kiro_payload: Existing debug kiro_request_body.json.

    Returns:
        Tuple of (mutated payload copy, mutation report).
    """
    payload = copy.deepcopy(kiro_payload)
    report = build_mutation_report(client_request)

    apply_system_cache_points(client_request, payload, report)
    apply_message_cache_points(client_request, payload, report)
    apply_tool_cache_points(client_request, payload, report)

    return payload, report


def find_matching_brace(text: str, start_pos: int) -> int:
    """
    Find the matching closing brace for a JSON object candidate.

    Args:
        text: Text to scan.
        start_pos: Position of the opening brace.

    Returns:
        Closing brace index, or -1 if no complete object exists.
    """
    if start_pos >= len(text) or text[start_pos] != "{":
        return -1

    depth = 0
    in_string = False
    escape_next = False

    for index in range(start_pos, len(text)):
        char = text[index]
        if escape_next:
            escape_next = False
            continue

        if char == "\\" and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index

    return -1


def iter_json_objects_from_bytes(raw: bytes) -> Iterable[Dict[str, Any]]:
    """
    Extract JSON objects embedded in raw Kiro stream bytes.

    Args:
        raw: Raw response stream bytes.

    Yields:
        Parsed JSON objects.
    """
    text = raw.decode("utf-8", errors="ignore")
    cursor = 0
    while cursor < len(text):
        start = text.find("{", cursor)
        if start == -1:
            break

        end = find_matching_brace(text, start)
        if end == -1:
            cursor = start + 1
            continue

        candidate = text[start:end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            cursor = start + 1
            continue

        if isinstance(parsed, dict):
            yield parsed
        cursor = end + 1


def collect_metric_fields(value: Any, metrics: Dict[str, Any]) -> None:
    """
    Collect credit, context, and token-usage fields recursively.

    Args:
        value: JSON-like value from a stream event.
        metrics: Metrics dictionary to update.
    """
    if isinstance(value, dict):
        usage_value = value.get("usage")
        if isinstance(usage_value, (int, float)):
            unit = value.get("unit")
            unit_plural = value.get("unitPlural")
            if unit == "credit" or unit_plural == "credits" or len(value) == 1:
                metrics["credits"].append(float(usage_value))

        context_usage = value.get("contextUsagePercentage")
        if isinstance(context_usage, (int, float)):
            metrics["context_usage_percentages"].append(float(context_usage))

        token_usage = {
            key: int(field_value)
            for key, field_value in value.items()
            if key in TOKEN_USAGE_KEYS and isinstance(field_value, (int, float))
        }
        if token_usage:
            metrics["token_usage_events"].append(token_usage)

        for child in value.values():
            collect_metric_fields(child, metrics)
    elif isinstance(value, list):
        for child in value:
            collect_metric_fields(child, metrics)


def extract_stream_metrics(raw: bytes) -> Dict[str, Any]:
    """
    Extract replay metrics from raw Kiro stream bytes.

    Args:
        raw: Raw response stream bytes.

    Returns:
        Dictionary with credits, context usage, and token usage data.
    """
    metrics: Dict[str, Any] = {
        "credits": [],
        "latest_credit": None,
        "context_usage_percentages": [],
        "latest_context_usage_percentage": None,
        "token_usage_events": [],
        "latest_token_usage": None,
    }

    for event in iter_json_objects_from_bytes(raw):
        collect_metric_fields(event, metrics)

    if metrics["credits"]:
        metrics["latest_credit"] = metrics["credits"][-1]
    if metrics["context_usage_percentages"]:
        metrics["latest_context_usage_percentage"] = metrics[
            "context_usage_percentages"
        ][-1]
    if metrics["token_usage_events"]:
        metrics["latest_token_usage"] = metrics["token_usage_events"][-1]

    return metrics


def count_cache_points(value: Any) -> int:
    """
    Count Kiro cachePoint markers in a JSON-like value.

    Args:
        value: JSON-like value.

    Returns:
        Number of cachePoint keys.
    """
    if isinstance(value, dict):
        return (1 if "cachePoint" in value else 0) + sum(
            count_cache_points(child) for child in value.values()
        )
    if isinstance(value, list):
        return sum(count_cache_points(child) for child in value)
    return 0


def load_original_stream_metrics(request_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load metrics from a historical debug response stream if present.

    Args:
        request_dir: Per-request debug log directory.

    Returns:
        Extracted stream metrics, or None when no raw response exists.
    """
    raw_stream_path = request_dir / "response_stream_raw.txt"
    if not raw_stream_path.exists():
        return None
    return extract_stream_metrics(raw_stream_path.read_bytes())


def resolve_request_dirs(
    request_dirs: Optional[List[str]],
    debug_dir: Path,
    latest: int,
    include_without_cache_control: bool,
) -> List[Path]:
    """
    Resolve explicit or latest debug request directories.

    Args:
        request_dirs: Explicit request directory paths.
        debug_dir: Debug root directory.
        latest: Number of latest candidates to select.
        include_without_cache_control: Include requests without cache_control.

    Returns:
        List of request directories.
    """
    if request_dirs:
        return [Path(path).expanduser().resolve() for path in request_dirs]

    requests_root = debug_dir / "requests"
    if not requests_root.exists():
        return []

    selected: List[Path] = []
    for directory in sorted(
        [path for path in requests_root.iterdir() if path.is_dir()],
        reverse=True,
    ):
        request_path = directory / "request_body.json"
        kiro_path = directory / "kiro_request_body.json"
        if not request_path.exists() or not kiro_path.exists():
            continue

        if not include_without_cache_control:
            try:
                request_data = load_json_file(request_path)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if not list(iter_cache_control_entries(request_data, [])):
                continue

        selected.append(directory)
        if len(selected) >= latest:
            break

    return selected


def set_conversation_id(payload: Dict[str, Any], conversation_id: str) -> Dict[str, Any]:
    """
    Return a payload copy with a replaced conversation ID.

    Args:
        payload: Kiro payload.
        conversation_id: New conversation ID.

    Returns:
        Payload copy with updated conversationState.conversationId.
    """
    updated = copy.deepcopy(payload)
    state = updated.setdefault("conversationState", {})
    if isinstance(state, dict):
        state["conversationId"] = conversation_id
    return updated


def configure_proxy_environment() -> None:
    """
    Apply the same proxy environment behavior as the main gateway.
    """
    from kiro.config import VPN_PROXY_URL

    if not VPN_PROXY_URL:
        return

    proxy_url = VPN_PROXY_URL if "://" in VPN_PROXY_URL else f"http://{VPN_PROXY_URL}"
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["ALL_PROXY"] = proxy_url

    no_proxy_hosts = os.environ.get("NO_PROXY", "")
    local_hosts = "127.0.0.1,localhost"
    os.environ["NO_PROXY"] = (
        f"{no_proxy_hosts},{local_hosts}" if no_proxy_hosts else local_hosts
    )


async def resolve_auth_manager() -> Tuple[Any, str]:
    """
    Resolve the auth manager used for live replay execution.

    Returns:
        Tuple of (KiroAuthManager, account identifier).

    Raises:
        RuntimeError: If no usable account can be initialized.
    """
    from kiro.account_manager import AccountManager
    from kiro.auth import KiroAuthManager
    from kiro.config import (
        ACCOUNTS_CONFIG_FILE,
        ACCOUNTS_STATE_FILE,
        KIRO_CLI_DB_FILE,
        KIRO_CREDS_FILE,
        PROFILE_ARN,
        REFRESH_TOKEN,
        REGION,
    )

    credentials_path = Path(ACCOUNTS_CONFIG_FILE).expanduser()
    if credentials_path.exists():
        manager = AccountManager(
            credentials_file=ACCOUNTS_CONFIG_FILE,
            state_file=ACCOUNTS_STATE_FILE,
        )
        await manager.load_credentials()
        await manager.load_state()

        for account_id in list(manager._accounts.keys()):
            if await manager._initialize_account(account_id):
                account = manager._accounts[account_id]
                if account.auth_manager is not None:
                    return account.auth_manager, account_id

    auth_manager = KiroAuthManager(
        refresh_token=REFRESH_TOKEN or None,
        profile_arn=PROFILE_ARN or None,
        region=REGION,
        creds_file=KIRO_CREDS_FILE or None,
        sqlite_db=KIRO_CLI_DB_FILE or None,
    )
    await auth_manager.get_access_token()
    return auth_manager, "legacy-env"


async def execute_payload(
    auth_manager: Any,
    payload: Dict[str, Any],
    timeout_seconds: float,
) -> Tuple[int, bytes]:
    """
    Send one Kiro payload and collect the raw response stream.

    Args:
        auth_manager: Initialized KiroAuthManager.
        payload: Kiro request payload.
        timeout_seconds: Maximum wall-clock seconds for the request.

    Returns:
        Tuple of (HTTP status code, raw response bytes).
    """
    from kiro.http_client import KiroHttpClient

    async def _send_and_collect() -> Tuple[int, bytes]:
        url = f"{auth_manager.api_host}/generateAssistantResponse"
        http_client = KiroHttpClient(auth_manager, shared_client=None)
        response = None
        raw = bytearray()
        try:
            response = await http_client.request_with_retry(
                "POST",
                url,
                payload,
                stream=True,
            )
            status_code = response.status_code
            if status_code == 200:
                async for chunk in response.aiter_bytes():
                    raw.extend(chunk)
            else:
                raw.extend(await response.aread())
            return status_code, bytes(raw)
        finally:
            if response is not None:
                await response.aclose()
            await http_client.close()

    return await asyncio.wait_for(_send_and_collect(), timeout=timeout_seconds)


def build_variant_payloads(
    client_request: Dict[str, Any],
    kiro_payload: Dict[str, Any],
    new_conversation_id: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Build baseline and cachePoint payloads for one source request.

    Args:
        client_request: Original client request.
        kiro_payload: Original Kiro payload.
        new_conversation_id: Whether to replace conversation IDs.

    Returns:
        Tuple of (baseline payload, cachePoint payload, mutation report).
    """
    baseline_payload = copy.deepcopy(kiro_payload)
    cachepoint_payload, report = apply_cachepoint_markers(
        client_request,
        kiro_payload,
    )

    if new_conversation_id:
        baseline_payload = set_conversation_id(baseline_payload, str(uuid.uuid4()))
        cachepoint_payload = set_conversation_id(cachepoint_payload, str(uuid.uuid4()))

    return baseline_payload, cachepoint_payload, report


def create_variant_plan(
    baseline_payload: Dict[str, Any],
    cachepoint_payload: Dict[str, Any],
    baseline_runs: int,
    cachepoint_runs: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Create the ordered A/B replay plan for a request.

    Args:
        baseline_payload: Original Kiro payload.
        cachepoint_payload: Mutated Kiro payload.
        baseline_runs: Number of baseline runs.
        cachepoint_runs: Number of cachePoint runs.

    Returns:
        List of (variant name, payload) tuples.
    """
    variants: List[Tuple[str, Dict[str, Any]]] = []
    for run_index in range(1, baseline_runs + 1):
        variants.append((f"baseline-{run_index}", baseline_payload))
    for run_index in range(1, cachepoint_runs + 1):
        variants.append((f"cachepoint-{run_index}", cachepoint_payload))
    return variants


async def run_replay(args: argparse.Namespace) -> int:
    """
    Run the replay experiment.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    debug_dir = Path(args.debug_dir).expanduser()
    output_root = Path(args.output_dir).expanduser()
    request_dirs = resolve_request_dirs(
        args.request_dir,
        debug_dir,
        args.latest,
        args.include_without_cache_control,
    )

    if not request_dirs:
        print("No matching debug request directories found.")
        return 1

    run_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    auth_manager = None
    account_id = None
    if args.execute:
        configure_proxy_environment()
        auth_manager, account_id = await resolve_auth_manager()
        print(f"Live replay account: {account_id}")
    else:
        print("Dry run only. Add --execute to send requests to Kiro.")

    aggregate: Dict[str, Any] = {
        "execute": args.execute,
        "run_dir": str(run_dir),
        "requests": [],
    }

    for request_dir in request_dirs:
        client_request = load_json_file(request_dir / "request_body.json")
        kiro_payload = load_json_file(request_dir / "kiro_request_body.json")
        baseline_payload, cachepoint_payload, mutation_report = build_variant_payloads(
            client_request,
            kiro_payload,
            args.new_conversation_id,
        )
        variants = create_variant_plan(
            baseline_payload,
            cachepoint_payload,
            args.baseline_runs,
            args.cachepoint_runs,
        )

        source_summary: Dict[str, Any] = {
            "source_dir": str(request_dir),
            "cache_controls_seen": mutation_report["cache_controls_seen"],
            "original_metrics": load_original_stream_metrics(request_dir),
            "original_cache_points": count_cache_points(kiro_payload),
            "cachepoint_payload_cache_points": count_cache_points(cachepoint_payload),
            "mutation_report": mutation_report,
            "variants": [],
        }
        aggregate["requests"].append(source_summary)

        source_output_dir = run_dir / request_dir.name
        write_json_file(source_output_dir / "mutation_report.json", mutation_report)

        print(
            f"{request_dir.name}: cache_control={mutation_report['cache_controls_seen']}, "
            f"added_cachePoint={mutation_report['markers_added']}, "
            f"original_credit={source_summary['original_metrics'].get('latest_credit') if source_summary['original_metrics'] else None}"
        )

        for variant_name, payload in variants:
            variant_dir = source_output_dir / variant_name
            write_json_file(variant_dir / "request_payload.json", payload)

            variant_summary: Dict[str, Any] = {
                "variant": variant_name,
                "payload_path": str(variant_dir / "request_payload.json"),
                "cache_points": count_cache_points(payload),
                "executed": False,
                "http_status": None,
                "metrics": None,
                "error": None,
            }

            if args.execute and auth_manager is not None:
                try:
                    status_code, raw = await execute_payload(
                        auth_manager,
                        payload,
                        args.timeout,
                    )
                    (variant_dir / "response_stream_raw.txt").write_bytes(raw)
                    metrics = extract_stream_metrics(raw)
                    variant_summary.update(
                        {
                            "executed": True,
                            "http_status": status_code,
                            "metrics": metrics,
                            "raw_response_path": str(
                                variant_dir / "response_stream_raw.txt"
                            ),
                        }
                    )
                    print(
                        f"  {variant_name}: status={status_code}, "
                        f"credit={metrics['latest_credit']}, "
                        f"context={metrics['latest_context_usage_percentage']}, "
                        f"token_usage={metrics['latest_token_usage']}"
                    )
                    if args.sleep_between_runs > 0:
                        await asyncio.sleep(args.sleep_between_runs)
                except (
                    asyncio.TimeoutError,
                    HTTPException,
                    RuntimeError,
                    OSError,
                    httpx.HTTPError,
                ) as exc:
                    variant_summary["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"  {variant_name}: ERROR {variant_summary['error']}")

            write_json_file(variant_dir / "summary.json", variant_summary)
            source_summary["variants"].append(variant_summary)

    write_json_file(run_dir / "summary.json", aggregate)
    print(f"Replay artifacts written to: {run_dir}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the replay CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Replay Kiro debug payloads with optional cachePoint markers. "
            "Defaults to dry-run and never sends network requests unless --execute is set."
        )
    )
    parser.add_argument(
        "--request-dir",
        action="append",
        help="Specific debug request directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--debug-dir",
        default="debug_logs",
        help="Debug log root directory. Default: debug_logs",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=3,
        help="Number of latest request directories to use when --request-dir is omitted.",
    )
    parser.add_argument(
        "--include-without-cache-control",
        action="store_true",
        help="Include requests that do not contain client cache_control fields.",
    )
    parser.add_argument(
        "--output-dir",
        default="debug_logs/cachepoint_replay",
        help="Directory for replay artifacts. Default: debug_logs/cachepoint_replay",
    )
    parser.add_argument(
        "--baseline-runs",
        type=int,
        default=1,
        help="Number of baseline no-cachePoint runs per request. Default: 1",
    )
    parser.add_argument(
        "--cachepoint-runs",
        type=int,
        default=2,
        help="Number of cachePoint runs per request. Default: 2",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send replay requests to Kiro. This consumes credits.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Maximum seconds for each live replay request. Default: 600",
    )
    parser.add_argument(
        "--sleep-between-runs",
        type=float,
        default=1.0,
        help="Seconds to wait between live replay runs. Default: 1",
    )
    parser.add_argument(
        "--new-conversation-id",
        action="store_true",
        help="Replace conversation IDs in generated payloads.",
    )
    return parser


def main() -> int:
    """
    CLI entry point.

    Returns:
        Process exit code.
    """
    parser = build_arg_parser()
    args = parser.parse_args()
    return asyncio.run(run_replay(args))


if __name__ == "__main__":
    raise SystemExit(main())
