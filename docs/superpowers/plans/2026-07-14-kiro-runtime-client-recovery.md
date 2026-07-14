# Kiro Runtime Client Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent a stale application-wide HTTP client from causing Kiro runtime requests to return 502.

**Architecture:** Kiro runtime requests in both public API adapters create an owned `KiroHttpClient` for every request. This matches the existing upstream streaming transport, which sends `Connection: close`, and keeps lifecycle cleanup in the route.

**Tech Stack:** Python 3.10+, FastAPI, httpx, pytest, pytest-asyncio.

## Global Constraints

- Preserve OpenAI and Anthropic request and response formats.
- Do not alter model discovery, authentication, retry counts, or timeout values.
- Use `shared_client=None` for every Kiro runtime request.
- Keep all tests network-isolated.

---

### Task 1: Lock runtime client ownership with regression tests

**Files:**
- Modify: `tests/unit/test_routes_openai.py:969-1010`
- Modify: `tests/unit/test_routes_anthropic.py:1206-1248`

**Interfaces:**
- Consumes: `KiroHttpClient(auth_manager, shared_client=None)`.
- Produces: route-level regression coverage for a non-streaming Kiro runtime request.

- [ ] **Step 1: Write failing tests**

Rename both `test_non_streaming_uses_shared_client` tests to `test_non_streaming_uses_per_request_client`, update their purpose text, and assert:

```python
assert call_args[1]["shared_client"] is None, \
    "Non-streaming Kiro runtime requests should use a per-request client"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHON_DOTENV_DISABLED=1 PROXY_API_KEY=test-key REFRESH_TOKEN=test-refresh-token /Users/gaoshan/.venvs/base/bin/python -m pytest -q tests/unit/test_routes_openai.py::TestHTTPClientUsage::test_non_streaming_uses_per_request_client tests/unit/test_routes_anthropic.py::TestHTTPClientUsage::test_non_streaming_uses_per_request_client
```

Expected: both fail because the routes still pass a shared client.

### Task 2: Make Kiro runtime clients request-owned

**Files:**
- Modify: `kiro/routes_openai.py:356-360,608-613`
- Modify: `kiro/routes_anthropic.py:421-425,739-744`

**Interfaces:**
- Consumes: the tested `KiroHttpClient` constructor behavior from Task 1.
- Produces: identical client ownership behavior for OpenAI/Anthropic and account-system/legacy route paths.

- [ ] **Step 1: Replace the branching client selection**

Replace each `if request_data.stream` / `else` client-selection block with:

```python
# Kiro runtime requests set Connection: close, so each request owns its client.
http_client = KiroHttpClient(auth_manager, shared_client=None)
```

- [ ] **Step 2: Run the focused regression tests**

Run the Task 1 pytest command. Expected: both tests pass.

- [ ] **Step 3: Run all affected unit modules**

Run:

```bash
PYTHON_DOTENV_DISABLED=1 PROXY_API_KEY=test-key REFRESH_TOKEN=test-refresh-token /Users/gaoshan/.venvs/base/bin/python -m pytest -q tests/unit/test_routes_openai.py tests/unit/test_routes_anthropic.py
```

Expected: all tests pass without live network access.

### Task 3: Verify end-to-end behavior

**Files:**
- Verify: `kiro/routes_openai.py`
- Verify: `kiro/routes_anthropic.py`

- [ ] **Step 1: Run the complete suite**

Run:

```bash
PYTHON_DOTENV_DISABLED=1 PROXY_API_KEY=test-key REFRESH_TOKEN=test-refresh-token /Users/gaoshan/.venvs/base/bin/python -m pytest -q
```

- [ ] **Step 2: Check whitespace and the focused diff**

Run `git diff --check` and inspect the focused route/test diff.

- [ ] **Step 3: Restart and probe the local gateway**

Run `bash restart.sh restart`, then submit a one-token authenticated `claude-sonnet-5` request to `http://127.0.0.1:6009/v1/messages`. Expected: HTTP 200.
