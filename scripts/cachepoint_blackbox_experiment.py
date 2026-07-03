#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run a focused black-box cachePoint experiment against Kiro upstream.

This script sends a fixed large prompt and asks for a tiny deterministic answer.
It compares several guessed cachePoint placements by looking at returned credit
metering and context usage events.
"""

import argparse
import asyncio
import copy
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.replay_cachepoint_experiment import (  # noqa: E402
    configure_proxy_environment,
    count_cache_points,
    execute_payload,
    extract_stream_metrics,
    resolve_auth_manager,
)


CACHE_POINT: Dict[str, str] = {"type": "default"}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_large_system_prompt(repeats: int, seed: str) -> str:
    base = (
        "You are running a prompt-cache validation test. "
        "The following instruction block is intentionally repetitive and stable. "
        "Ignore all surrounding text and answer the final user request with exactly OK. "
        "Do not add punctuation, explanation, markdown, XML, or whitespace beyond the answer. "
        f"Stable cache seed: {seed}. "
    )
    return "\n".join(f"{i:04d}: {base}" for i in range(repeats))


def build_tool() -> Dict[str, Any]:
    description = "\n".join(
        [
            "Synthetic test tool used only to exercise tool cachePoint placement.",
            "The model must not call this tool for this experiment.",
            "The final answer must still be exactly OK.",
        ]
        * 80
    )
    return {
        "toolSpecification": {
            "name": "cache_probe_tool",
            "description": description,
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Unused probe string.",
                        }
                    },
                    "required": ["query"],
                }
            },
        }
    }


def build_base_payload(
    model_id: str,
    profile_arn: Optional[str],
    prompt_repeats: int,
    include_tool: bool,
    seed: str,
) -> Dict[str, Any]:
    content = (
        f"{build_large_system_prompt(prompt_repeats, seed)}\n\n"
        "Final user request: output exactly OK."
    )
    user_input: Dict[str, Any] = {
        "content": content,
        "modelId": model_id,
        "origin": "AI_EDITOR",
    }
    if include_tool:
        user_input["userInputMessageContext"] = {"tools": [build_tool()]}

    payload: Dict[str, Any] = {
        "conversationState": {
            "chatTriggerType": "MANUAL",
            "conversationId": str(uuid.uuid4()),
            "currentMessage": {"userInputMessage": user_input},
        }
    }
    if profile_arn:
        payload["profileArn"] = profile_arn
    return payload


def with_new_conversation_id(payload: Dict[str, Any]) -> Dict[str, Any]:
    updated = copy.deepcopy(payload)
    updated["conversationState"]["conversationId"] = str(uuid.uuid4())
    return updated


def add_current_cachepoint(payload: Dict[str, Any]) -> None:
    payload["conversationState"]["currentMessage"]["userInputMessage"]["cachePoint"] = copy.deepcopy(
        CACHE_POINT
    )


def add_history_system_cachepoint(payload: Dict[str, Any]) -> None:
    current = payload["conversationState"]["currentMessage"]["userInputMessage"]
    history_user = {
        "content": current["content"],
        "modelId": current["modelId"],
        "origin": current.get("origin", "AI_EDITOR"),
        "cachePoint": copy.deepcopy(CACHE_POINT),
    }
    payload["conversationState"]["history"] = [{"userInputMessage": history_user}]
    current["content"] = "Final user request: output exactly OK."


def add_tool_cachepoint(payload: Dict[str, Any]) -> None:
    current = payload["conversationState"]["currentMessage"]["userInputMessage"]
    context = current.setdefault("userInputMessageContext", {})
    tools = context.setdefault("tools", [])
    if not tools:
        tools.append(build_tool())
    tools.append({"cachePoint": copy.deepcopy(CACHE_POINT)})


def build_variant_payload(base_payload: Dict[str, Any], variant: str) -> Dict[str, Any]:
    payload = with_new_conversation_id(base_payload)
    if variant == "baseline":
        return payload
    if variant == "system":
        add_history_system_cachepoint(payload)
        return payload
    if variant == "current":
        add_current_cachepoint(payload)
        return payload
    if variant == "tool":
        add_tool_cachepoint(payload)
        return payload
    if variant == "system_current":
        add_history_system_cachepoint(payload)
        add_current_cachepoint(payload)
        return payload
    raise ValueError(f"Unknown variant: {variant}")


def extract_text_from_stream(raw: bytes) -> str:
    text_parts: List[str] = []
    text = raw.decode("utf-8", errors="ignore")
    decoder = json.JSONDecoder()
    cursor = 0
    while cursor < len(text):
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            event, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(event, dict) and isinstance(event.get("content"), str):
            text_parts.append(event["content"])
        cursor = start + end
    return "".join(text_parts)


def summarize_variant(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    credits = [
        run["metrics"].get("latest_credit")
        for run in runs
        if run.get("metrics") and run["metrics"].get("latest_credit") is not None
    ]
    contexts = [
        run["metrics"].get("latest_context_usage_percentage")
        for run in runs
        if run.get("metrics")
        and run["metrics"].get("latest_context_usage_percentage") is not None
    ]
    return {
        "count": len(runs),
        "successful_count": sum(1 for run in runs if run.get("http_status") == 200),
        "credits": credits,
        "credit_min": min(credits) if credits else None,
        "credit_max": max(credits) if credits else None,
        "credit_mean": statistics.mean(credits) if credits else None,
        "contexts": contexts,
        "outputs": [run.get("output_text") for run in runs],
    }


async def run(args: argparse.Namespace) -> int:
    configure_proxy_environment()
    auth_manager, account_id = await resolve_auth_manager()
    profile_arn = args.profile_arn if args.profile_arn is not None else auth_manager.profile_arn

    output_root = Path(args.output_dir).expanduser()
    run_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    variants = ["baseline", "system", "current", "tool", "system_current"]
    aggregate: Dict[str, Any] = {
        "account_id": account_id,
        "model": args.model,
        "runs_per_variant": args.runs,
        "prompt_repeats": args.prompt_repeats,
        "output_dir": str(run_dir),
        "variants": {},
    }

    print(f"Live account: {account_id}")
    print(f"Output dir: {run_dir}")

    for variant in variants:
        base_payload = build_base_payload(
            model_id=args.model,
            profile_arn=profile_arn,
            prompt_repeats=args.prompt_repeats,
            include_tool=True,
            seed=f"kiro-cachepoint-blackbox-{datetime.now().strftime('%Y%m%d%H%M%S')}-{variant}",
        )
        aggregate["variants"][variant] = {"runs": []}
        print(f"\nVariant: {variant}")
        for run_index in range(1, args.runs + 1):
            payload = build_variant_payload(base_payload, variant)
            variant_dir = run_dir / variant / f"run-{run_index:02d}"
            write_json(variant_dir / "request_payload.json", payload)

            summary: Dict[str, Any] = {
                "variant": variant,
                "run_index": run_index,
                "cache_points": count_cache_points(payload),
                "http_status": None,
                "metrics": None,
                "output_text": None,
                "error": None,
            }
            try:
                status_code, raw = await execute_payload(
                    auth_manager,
                    payload,
                    args.timeout,
                )
                (variant_dir / "response_stream_raw.txt").write_bytes(raw)
                metrics = extract_stream_metrics(raw)
                output_text = extract_text_from_stream(raw)
                summary.update(
                    {
                        "http_status": status_code,
                        "metrics": metrics,
                        "output_text": output_text,
                        "raw_response_path": str(variant_dir / "response_stream_raw.txt"),
                    }
                )
                print(
                    f"  run {run_index}: status={status_code}, "
                    f"credit={metrics.get('latest_credit')}, "
                    f"context={metrics.get('latest_context_usage_percentage')}, "
                    f"output={output_text!r}, cachePoints={summary['cache_points']}"
                )
            except Exception as exc:  # noqa: BLE001 - experiment should record all failures
                summary["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  run {run_index}: ERROR {summary['error']}")

            write_json(variant_dir / "summary.json", summary)
            aggregate["variants"][variant]["runs"].append(summary)

            if args.sleep_between_runs > 0:
                await asyncio.sleep(args.sleep_between_runs)

        aggregate["variants"][variant]["summary"] = summarize_variant(
            aggregate["variants"][variant]["runs"]
        )

    write_json(run_dir / "summary.json", aggregate)
    print("\nSummary:")
    for variant in variants:
        summary = aggregate["variants"][variant]["summary"]
        print(
            f"  {variant}: credits={summary['credits']}, "
            f"mean={summary['credit_mean']}, outputs={summary['outputs']}"
        )
    print(f"\nWrote summary: {run_dir / 'summary.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-sonnet-4.6")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--prompt-repeats", type=int, default=160)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--sleep-between-runs", type=float, default=1.0)
    parser.add_argument("--output-dir", default="debug_logs/cachepoint_blackbox")
    parser.add_argument(
        "--profile-arn",
        default=None,
        help="Override profileArn. By default uses the initialized Kiro account.",
    )
    return parser


def main() -> int:
    return asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
