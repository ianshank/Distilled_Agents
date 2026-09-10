# ADR 0001: Replace pickle cache serialization with JSON

## Status

Accepted

## Context

L2 (Redis) and L3 (S3) caches previously used `pickle`. Untrusted pickle payloads are a remote-code-execution risk if cache storage is poisoned.

## Decision

Serialize cache values as UTF-8 JSON. Non-JSON types fail closed with a TypeError. Existing pickle blobs are invalid after this change; flush Redis/S3 cache prefixes.

## Consequences

Safer defaults, slightly narrower cached value types (dicts, lists, strings, numbers, booleans, null).
