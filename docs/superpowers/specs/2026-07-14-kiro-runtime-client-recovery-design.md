# Kiro Runtime Client Recovery Design

## Problem

The gateway's long-lived shared `httpx.AsyncClient` can enter a state in which
Kiro runtime requests fail with `httpx.ConnectError`, while a newly created
client using the same endpoint and credentials succeeds. Restarting the
gateway recreates the client and restores requests.

## Decision

Kiro runtime requests will use an owned, per-request `KiroHttpClient` in both
OpenAI and Anthropic routes, for streaming and non-streaming calls. Model
catalog discovery already uses an owned client and is out of scope.

## Rationale

The request path sends upstream requests in streaming mode and adds
`Connection: close`, so retaining a long-lived pooled client does not provide
connection reuse for these Kiro runtime calls. A per-request client prevents a
stale shared client from blocking future requests and preserves the existing
retry, timeout, authentication, and response conversion behavior.

## Scope and Safety

- Do not change the OpenAI or Anthropic public API payloads.
- Do not change model discovery, authentication, timeouts, or retry counts.
- Keep the application-level shared client for unrelated users until a
  dedicated cleanup removes it with separate evidence.
- Retain the current `KiroHttpClient.close()` call so each route closes its
  owned client on success and error paths.

## Verification

Unit tests in both route test modules will assert that non-streaming Kiro
runtime requests pass `shared_client=None`, matching streaming behavior. The
full test suite and a local authenticated one-token request will verify the
regression is fixed.
