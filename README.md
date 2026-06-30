# Agent Cache

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![AI](https://img.shields.io/badge/AI-LLM%20Cache-0F766E)](https://github.com/)
[![LLM](https://img.shields.io/badge/LLM-Safe%20Reuse-111827)](https://github.com/)

Agent Cache is a client-side cache for LLM prompts.

It reduces token cost when users repeat the same task shape with different details.

It is not a semantic answer cache.
Similarity can propose reuse, but validation decides.

## How It Saves Tokens

- Repeated prompt structure can reuse a safe capsule.
- Docs-backed prompts can reuse a validated docs index.
- If a prompt depends on live facts, the cache must validate or miss.

## Install

Clone the repository and install the Codex skill:

```bash
git clone <your-repo-url>
cd prompt-cache
./skills/agent-cache/scripts/install_to_codex.sh
```

## Use

Initialize the cache:

```bash
python -m agent_cache.cli init
```

Ingest docs:

```bash
python -m agent_cache.cli ingest https://pokeapi.co/docs/v2 --output /tmp/pokeapi-index.jsonl
```

Ask a question:

```bash
python -m agent_cache.cli ask "How does PokéAPI pagination work?"
```

Validate cache savings:

```bash
python -m agent_cache.cli doctor
python -m agent_cache.cli telemetry
```

`doctor` shows prompt, response, capsule, and docs index counts.

`telemetry` shows:

- `hits`
- `misses`
- `candidate_accepts`
- `candidate_rejects`
- `validation_failed`
- `estimated_tokens_saved`
- `estimated_cost_saved`

These numbers are estimates, not guarantees.

## Example Run

This is what a real docs-backed run looks like in Codex:

1. Ingest the PokéAPI docs.

```bash
python3 -m agent_cache.cli docs ingest-url https://pokeapi.co/docs/v2 --output /tmp/pokeapi-index.jsonl
```

2. If the built-in fetcher is blocked, download the page with a browser-like user agent and ingest the local HTML copy instead.

3. Ask route questions against the same persisted database.

- `What does /pokemon/{id} do in PokéAPI?` -> `HIT_EXACT`
- `What does /ability/{id} do in PokéAPI?` -> `HIT_EXACT`

Observed cache state after the run:

- `prompts: 6`
- `responses: 6`
- `capsules: 2`
- `docs_indexes: 2`
- `hits: 4`
- `misses: 2`
- `estimated_tokens_saved: 112`

## Examples

Docs-backed prompt:

```text
How does PokéAPI pagination work?
```

Shopping prompt:

```text
Find AAA batteries under 30
```

## Contributing

Contributions are welcome.

- Open an issue before large changes if the behavior or API will change.
- Keep validation deterministic and explicit.
- Prefer MISS over unsafe reuse.
- Add or update tests for any cache behavior change.

## License

MIT License. See [LICENSE](LICENSE).

## CLI

- `agent-cache init`
- `agent-cache ask "prompt"`
- `agent-cache ingest <url-or-path...> --output index.jsonl`
- `agent-cache replay fixtures/*.jsonl --index index.jsonl`
- `agent-cache stats`
- `agent-cache telemetry`
- `agent-cache doctor`
- `agent-cache capsules add|show|edit|list|inspect|disable|import|export`

## Principles

- Similarity proposes.
- Validation decides.
- Prefer MISS over unsafe reuse.
- Never guess dynamic data.

By default, Agent Cache stores its SQLite database at `/tmp/agent-cache.sqlite3`.
