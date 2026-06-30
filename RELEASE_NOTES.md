# Release Notes

## v0.1.0

Initial public release of Agent Cache.

### Highlights

- Client-side prompt cache for structurally repeated prompts
- SQLite persistence for prompts, responses, capsules, and docs indexes
- Safe docs ingestion from URLs and local files
- Cached docs-backed `ask` flow with validation-first behavior
- CLI, telemetry, doctor, and replay commands
- Codex skill integration and install script

### Installation

- Local: `python -m pip install .`
- GitHub: `python -m pip install git+https://github.com/lmagdanello/agent-cache.git`

### Notes

- Default SQLite database: `/tmp/agent-cache.sqlite3`
- License: MIT
