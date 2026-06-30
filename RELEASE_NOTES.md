# Release Notes

## v0.1.1

Docs-backed cache and packaging improvements.

### Highlights

- Persistent docs indexes stored in SQLite
- `ask` returns `response`, `source`, `docs_index_id`, and `matched_title`
- Default database uses `/tmp/agent-cache.sqlite3`
- PDF and OCR Python dependencies ship with the package
- GitHub release workflow builds wheel and sdist artifacts

### Notes

- Image OCR still requires the `tesseract` system binary
- Existing `v0.1.0` remains available as the initial release

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
