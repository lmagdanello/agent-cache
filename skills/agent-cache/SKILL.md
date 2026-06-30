---
name: agent-cache
description: Use when working on the Agent Cache project from Codex prompts, especially prompt-first smoke tests, capsule CRUD, telemetry, and safe cache behavior.
---

# Agent Cache

Use this skill for the local `agent_cache` project.

## Default behavior

If the user gives only a natural-language prompt, do this:

1. Use the default SQLite DB at `/tmp/agent-cache.sqlite3`, or a temporary DB only when the user explicitly asks for isolation.
2. Add or choose a safe capsule that matches the prompt shape.
3. Run `ask`.
4. Report `doctor` and `telemetry`.

If the user gives capsule JSON, fixtures, or asks for explicit validation:

1. Use the CLI directly.
2. Validate with `doctor` and `telemetry`.
3. Use `replay` for fixture runs.

## Core rule

- Similarity proposes.
- Validation decides.
- Never treat a similar prompt as a valid answer by itself.

## Prompt-first mode

Use prompt-first mode for quick smoke tests:

```text
Use the agent-cache skill: Find AAA batteries under 30.
```

In this mode, infer the obvious intent, create the temp DB, and run the full cache path.

## Data-backed mode

Use data-backed mode when the response depends on real facts:

- catalog
- docs
- API
- local DB
- fixture set

The cache should help avoid repeated LLM work, but it must not invent live data.

For docs-backed workflows, first build an index with:

- `agent-cache ingest <url-or-path...> --output <index.jsonl>`
- `agent-cache docs ingest <paths...> --output <index.jsonl>`
- `agent-cache docs ingest-url <url> --output <index.jsonl>`

Typical prompt shape:

```text
Use the agent-cache skill: ingest https://pokeapi.co/docs/v2, then ask: How does PokéAPI pagination work?
```

The ingest step persists the docs index in SQLite and also writes the JSONL index file. After that, ask questions normally; you do not need to re-run ingest unless the source docs changed.
The ingest step accepts markdown, HTML, PDF, DOCX, XLSX, CSV, JSON, XML, and images when the optional extractors are available.
When `ask` returns a docs-backed hit, use the returned `response` and `source` fields directly. Do not infer the endpoint behavior from the hit alone.

For repository examples:

- Prefer the default persistent DB unless isolation is required.
- For structural reuse, create a capsule that matches the prompt template.
- For data-backed reuse, ingest a docs index or catalog first, then replay the prompt against it.

## Capsule editing

- Use `capsules show` before editing.
- Use `capsules edit` for top-level changes.
- Use `capsules import` and `capsules export` for round-trips.
- Keep dynamic fields explicit.
- Prefer MISS when a field is volatile or unvalidated.
