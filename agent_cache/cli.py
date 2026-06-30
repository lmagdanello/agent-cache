from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

from .agent import AgentCacheAgent
from .docs import ingest_sources, ingest_url
from .llm import FakeLLMClient
from .models import AgentCacheConfig, CacheCapsule, structural_signature_from_prompt
from .telemetry import TelemetryCounters
from .store import SQLiteCacheStore


def _default_db_path() -> Path:
    override = os.environ.get("AGENT_CACHE_DB")
    if override:
        return Path(override)
    return Path("/tmp/agent-cache.sqlite3")


def _read_json_input(path: str | None, raw_json: str | None) -> dict:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    if raw_json:
        return json.loads(raw_json)
    raise ValueError("either --file or --json is required")


def _capsule_from_payload(payload: dict) -> CacheCapsule:
    metadata = dict(payload.get("metadata", {}))
    prompt_signature = payload.get("prompt_structure_signature")
    if not prompt_signature:
        prompt_template = metadata.get("prompt_template")
        if isinstance(prompt_template, str) and prompt_template.strip():
            prompt_signature = structural_signature_from_prompt(prompt_template)
        else:
            prompt_signature = structural_signature_from_prompt(payload.get("name", ""))
    return CacheCapsule(
        capsule_id=payload["capsule_id"],
        name=payload["name"],
        prompt_structure_signature=prompt_signature,
        regex_rules=tuple(payload.get("regex_rules", [])),
        parser_rules=dict(payload.get("parser_rules", {})),
        response_schema=dict(payload.get("response_schema", {})),
        allowed_dynamic_fields=tuple(payload.get("allowed_dynamic_fields", [])),
        forbidden_fields=tuple(payload.get("forbidden_fields", [])),
        ttl_seconds=int(payload.get("ttl_seconds", 0)),
        max_age_seconds=int(payload.get("max_age_seconds", payload.get("ttl_seconds", 0))),
        max_examples=int(payload.get("max_examples", 0)),
        generated_program_id=payload.get("generated_program_id"),
        validation_policy=dict(payload.get("validation_policy", {})),
        safety_flags=tuple(payload.get("safety_flags", [])),
        enabled=bool(payload.get("enabled", True)),
        metadata=metadata,
    )


def _capsule_to_payload(capsule: CacheCapsule) -> dict:
    data = asdict(capsule)
    data["regex_rules"] = list(capsule.regex_rules)
    data["allowed_dynamic_fields"] = list(capsule.allowed_dynamic_fields)
    data["forbidden_fields"] = list(capsule.forbidden_fields)
    data["safety_flags"] = list(capsule.safety_flags)
    return data


def _merge_capsule(capsule: CacheCapsule, patch: dict) -> CacheCapsule:
    data = _capsule_to_payload(capsule)
    data.update(patch)
    data["capsule_id"] = capsule.capsule_id
    data["name"] = data.get("name", capsule.name)
    data["prompt_structure_signature"] = data.get("prompt_structure_signature", capsule.prompt_structure_signature)
    data["regex_rules"] = tuple(data.get("regex_rules", capsule.regex_rules))
    data["parser_rules"] = dict(data.get("parser_rules", capsule.parser_rules))
    data["response_schema"] = dict(data.get("response_schema", capsule.response_schema))
    data["allowed_dynamic_fields"] = tuple(data.get("allowed_dynamic_fields", capsule.allowed_dynamic_fields))
    data["forbidden_fields"] = tuple(data.get("forbidden_fields", capsule.forbidden_fields))
    data["ttl_seconds"] = int(data.get("ttl_seconds", capsule.ttl_seconds))
    data["max_age_seconds"] = int(data.get("max_age_seconds", capsule.max_age_seconds))
    data["max_examples"] = int(data.get("max_examples", capsule.max_examples))
    data["generated_program_id"] = data.get("generated_program_id", capsule.generated_program_id)
    data["validation_policy"] = dict(data.get("validation_policy", capsule.validation_policy))
    data["safety_flags"] = tuple(data.get("safety_flags", capsule.safety_flags))
    data["enabled"] = bool(data.get("enabled", capsule.enabled))
    data["metadata"] = dict(data.get("metadata", capsule.metadata))
    return CacheCapsule(
        capsule_id=data["capsule_id"],
        name=data["name"],
        prompt_structure_signature=data["prompt_structure_signature"],
        regex_rules=data["regex_rules"],
        parser_rules=data["parser_rules"],
        response_schema=data["response_schema"],
        allowed_dynamic_fields=data["allowed_dynamic_fields"],
        forbidden_fields=data["forbidden_fields"],
        ttl_seconds=data["ttl_seconds"],
        max_age_seconds=data["max_age_seconds"],
        max_examples=data["max_examples"],
        generated_program_id=data["generated_program_id"],
        validation_policy=data["validation_policy"],
        safety_flags=data["safety_flags"],
        enabled=data["enabled"],
        metadata=data["metadata"],
    )


def _iter_jsonl_payloads(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_jsonl_rows(path: Path) -> list[dict]:
    return list(_iter_jsonl_payloads(path))


def _match_index_item(prompt: str, index_rows: list[dict]) -> dict | None:
    normalized = " ".join(prompt.lower().split())
    match = re.search(r"find\s+(.*?)\s+under\s+(\d+)", normalized)
    if not match:
        query_tokens = [token for token in re.findall(r"[a-z0-9]+", normalized) if len(token) > 2]
        best_row = None
        best_score = 0
        for row in index_rows:
            haystack = " ".join(
                str(row.get(field, ""))
                for field in ("name", "title", "summary", "description", "url")
            ).lower()
            haystack += " " + " ".join(
                str(item).lower()
                for item in row.get("keywords", [])
            )
            score = sum(1 for token in query_tokens if token in haystack)
            if score > best_score:
                best_row = row
                best_score = score
        return best_row

    item_query = match.group(1).strip()
    budget = int(match.group(2))
    query_tokens = [token for token in item_query.split() if token]
    best_row = None
    for row in index_rows:
        name = str(row.get("name", "")).lower()
        price = row.get("price")
        if not isinstance(price, (int, float)) or price > budget:
            continue
        if all(token in name for token in query_tokens):
            if best_row is None or float(price) < float(best_row.get("price", price)):
                best_row = row
    return best_row


def _index_answer(index_item: dict | None) -> str | None:
    if not index_item:
        return None
    if "price" in index_item:
        return f"{index_item['name']} costs {index_item['price']} and is {index_item.get('availability', 'unknown')}"
    return index_item.get("summary") or index_item.get("description") or index_item.get("title")


def _index_source(index_item: dict | None) -> str | None:
    if not index_item:
        return None
    return index_item.get("url") or index_item.get("source")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-cache")
    sub = parser.add_subparsers(dest="command", required=True)

    init_cmd = sub.add_parser("init")
    init_cmd.add_argument("--db", default=str(_default_db_path()))

    ask_cmd = sub.add_parser("ask")
    ask_cmd.add_argument("prompt")
    ask_cmd.add_argument("--db", default=str(_default_db_path()))

    stats_cmd = sub.add_parser("stats")
    stats_cmd.add_argument("--db", default=str(_default_db_path()))

    capsules_cmd = sub.add_parser("capsules")
    capsules_sub = capsules_cmd.add_subparsers(dest="capsule_command", required=True)
    capsules_add = capsules_sub.add_parser("add")
    capsules_add.add_argument("--db", default=str(_default_db_path()))
    capsules_add.add_argument("--file")
    capsules_add.add_argument("--json")
    capsules_show = capsules_sub.add_parser("show")
    capsules_show.add_argument("capsule_id")
    capsules_show.add_argument("--db", default=str(_default_db_path()))
    capsules_import = capsules_sub.add_parser("import")
    capsules_import.add_argument("--db", default=str(_default_db_path()))
    capsules_import.add_argument("--file", required=True)
    capsules_export = capsules_sub.add_parser("export")
    capsules_export.add_argument("--db", default=str(_default_db_path()))
    capsules_export.add_argument("--file")
    capsules_list = capsules_sub.add_parser("list")
    capsules_list.add_argument("--db", default=str(_default_db_path()))
    capsules_edit = capsules_sub.add_parser("edit")
    capsules_edit.add_argument("capsule_id")
    capsules_edit.add_argument("--db", default=str(_default_db_path()))
    capsules_edit.add_argument("--json")
    capsules_edit.add_argument("--file")
    capsules_inspect = capsules_sub.add_parser("inspect")
    capsules_inspect.add_argument("capsule_id")
    capsules_inspect.add_argument("--db", default=str(_default_db_path()))
    capsules_disable = capsules_sub.add_parser("disable")
    capsules_disable.add_argument("capsule_id")
    capsules_disable.add_argument("--db", default=str(_default_db_path()))

    gc_cmd = sub.add_parser("gc")
    gc_cmd.add_argument("--db", default=str(_default_db_path()))

    doctor_cmd = sub.add_parser("doctor")
    doctor_cmd.add_argument("--db", default=str(_default_db_path()))

    telemetry_cmd = sub.add_parser("telemetry")
    telemetry_cmd.add_argument("--db", default=str(_default_db_path()))
    telemetry_cmd.add_argument("--format", choices=["json", "prometheus"], default="json")

    ingest_cmd = sub.add_parser("ingest")
    ingest_cmd.add_argument("sources", nargs="+")
    ingest_cmd.add_argument("--output", required=True)
    ingest_cmd.add_argument("--db", default=str(_default_db_path()))

    docs_cmd = sub.add_parser("docs")
    docs_sub = docs_cmd.add_subparsers(dest="docs_command", required=True)
    docs_ingest = docs_sub.add_parser("ingest")
    docs_ingest.add_argument("sources", nargs="+")
    docs_ingest.add_argument("--output", required=True)
    docs_ingest.add_argument("--db", default=str(_default_db_path()))
    docs_ingest_url = docs_sub.add_parser("ingest-url")
    docs_ingest_url.add_argument("url")
    docs_ingest_url.add_argument("--output", required=True)
    docs_ingest_url.add_argument("--db", default=str(_default_db_path()))

    replay_cmd = sub.add_parser("replay")
    replay_cmd.add_argument("fixtures", nargs="+")
    replay_cmd.add_argument("--db", default=str(_default_db_path()))
    replay_cmd.add_argument("--catalog")
    replay_cmd.add_argument("--index")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        SQLiteCacheStore(args.db).close()
        print(json.dumps({"ok": True, "db": args.db}))
        return 0

    if args.command == "ingest":
        store = SQLiteCacheStore(args.db)
        try:
            rows = ingest_sources(args.sources, Path(args.output))
            index_id = Path(args.output).stem
            name = Path(args.output).stem.replace("_", " ").replace("-", " ").title() or "Docs Index"
            source = ", ".join(args.sources)
            store.save_docs_index(index_id, name=name, source=source, rows=_load_jsonl_rows(Path(args.output)), metadata={"output": args.output})
            print(json.dumps({"ok": True, "rows": rows, "output": args.output, "docs_capsule_id": f"docs::{index_id}"}))
        finally:
            store.close()
        return 0

    if args.command == "docs":
        if args.docs_command == "ingest":
            store = SQLiteCacheStore(args.db)
            try:
                rows = ingest_sources(args.sources, Path(args.output))
                index_id = Path(args.output).stem
                name = Path(args.output).stem.replace("_", " ").replace("-", " ").title() or "Docs Index"
                source = ", ".join(args.sources)
                store.save_docs_index(index_id, name=name, source=source, rows=_load_jsonl_rows(Path(args.output)), metadata={"output": args.output})
                print(json.dumps({"ok": True, "rows": rows, "output": args.output, "docs_capsule_id": f"docs::{index_id}"}))
            finally:
                store.close()
            return 0
        if args.docs_command == "ingest-url":
            store = SQLiteCacheStore(args.db)
            try:
                rows = ingest_url(args.url, Path(args.output))
                index_id = Path(args.output).stem
                name = Path(args.output).stem.replace("_", " ").replace("-", " ").title() or "Docs Index"
                store.save_docs_index(index_id, name=name, source=args.url, rows=_load_jsonl_rows(Path(args.output)), metadata={"output": args.output, "url": args.url})
                print(json.dumps({"ok": True, "rows": rows, "output": args.output, "url": args.url, "docs_capsule_id": f"docs::{index_id}"}))
            finally:
                store.close()
            return 0

    store = SQLiteCacheStore(args.db)
    agent = AgentCacheAgent(store=store, llm_client=FakeLLMClient(), config=AgentCacheConfig())

    if args.command == "ask":
        result = agent.ask(args.prompt)
        response = result.decision.response
        response_metadata = response.metadata if response else {}
        print(
            json.dumps(
                {
                    "status": result.decision.status.value,
                    "score": result.decision.score,
                    "reason": result.decision.reason,
                    "response": response.raw_response if response else None,
                    "source": response_metadata.get("source"),
                    "docs_index_id": response_metadata.get("docs_index_id"),
                    "matched_title": response_metadata.get("matched_title"),
                }
            )
        )
        return 0

    if args.command == "stats":
        print(
            json.dumps(
                {
                    "prompts": store.count_prompts(),
                    "responses": store.count_responses(),
                    "capsules": store.count_capsules(),
                    "docs_indexes": store.count_docs_indexes(),
                    "validation_logs": store.count_validation_logs(),
                }
            )
        )
        return 0

    if args.command == "doctor":
        capsules = store.iter_capsules()
        telemetry = TelemetryCounters.from_dict(store.get_meta_json("telemetry_counters"))
        capsule_health = []
        for capsule in capsules:
            summary = store.capsule_validation_summary(capsule.capsule_id)
            total = summary["total"]
            rejection_rate = (summary["failed"] / total) if total else 0.0
            hit_rate = (summary["passed"] / total) if total else 0.0
            capsule_health.append(
                {
                    "capsule_id": capsule.capsule_id,
                    "name": capsule.name,
                    "enabled": capsule.enabled,
                    "validation_total": total,
                    "validation_passed": summary["passed"],
                    "validation_failed": summary["failed"],
                    "hit_rate": hit_rate,
                    "rejection_rate": rejection_rate,
                }
            )
            payload = {
                "db": str(args.db),
                "schema_version": store.get_meta_value("schema_version"),
                "counts": {
                    "prompts": store.count_prompts(),
                    "responses": store.count_responses(),
                    "capsules": store.count_capsules(),
                    "docs_indexes": store.count_docs_indexes(),
                    "disabled_capsules": store.count_disabled_capsules(),
                    "validation_logs": store.count_validation_logs(),
                },
            "capsule_names": [capsule.name for capsule in capsules],
            "capsule_health": capsule_health,
            "telemetry": telemetry.to_dict(),
        }
        print(json.dumps(payload))
        return 0

    if args.command == "telemetry":
        telemetry = TelemetryCounters.from_dict(store.get_meta_json("telemetry_counters"))
        if args.format == "prometheus":
            print(telemetry.to_prometheus(), end="")
        else:
            print(json.dumps(telemetry.to_dict()))
        return 0

    if args.command == "capsules":
        if args.capsule_command == "add":
            payload = _read_json_input(args.file, args.json)
            capsule = _capsule_from_payload(payload)
            store.save_capsule(capsule)
            print(json.dumps({"ok": True, "capsule_id": capsule.capsule_id}))
            return 0
        if args.capsule_command == "show":
            capsule = store.get_capsule(args.capsule_id)
            print(json.dumps(_capsule_to_payload(capsule) if capsule else None, default=str))
            return 0
        if args.capsule_command == "import":
            imported = 0
            for payload in _iter_jsonl_payloads(Path(args.file)):
                store.save_capsule(_capsule_from_payload(payload))
                imported += 1
            print(json.dumps({"ok": True, "imported": imported}))
            return 0
        if args.capsule_command == "export":
            payloads = [_capsule_to_payload(capsule) for capsule in store.iter_capsules()]
            if args.file:
                path = Path(args.file)
                path.write_text(
                    "\n".join(json.dumps(payload, default=str) for payload in payloads) + ("\n" if payloads else ""),
                    encoding="utf-8",
                )
                print(json.dumps({"ok": True, "exported": len(payloads), "file": str(path)}))
            else:
                print("\n".join(json.dumps(payload, default=str) for payload in payloads))
            return 0
        if args.capsule_command == "list":
            print(
                json.dumps(
                    [
                        {
                            "capsule_id": capsule.capsule_id,
                            "name": capsule.name,
                            "enabled": capsule.enabled,
                            "ttl_seconds": capsule.ttl_seconds,
                        }
                        for capsule in store.iter_capsules()
                    ]
                )
            )
            return 0
        if args.capsule_command == "edit":
            capsule = store.get_capsule(args.capsule_id)
            if capsule is None:
                print(json.dumps({"ok": False, "error": "not_found"}))
                return 0
            patch = _read_json_input(args.file, args.json)
            updated = _merge_capsule(capsule, patch)
            store.save_capsule(updated)
            print(json.dumps({"ok": True, "capsule_id": updated.capsule_id}))
            return 0
        if args.capsule_command == "inspect":
            capsule = store.get_capsule(args.capsule_id)
            print(json.dumps(asdict(capsule) if capsule else None, default=str))
            return 0
        if args.capsule_command == "disable":
            capsule = store.get_capsule(args.capsule_id)
            if capsule:
                capsule.enabled = False
                store.save_capsule(capsule)
            print(json.dumps({"ok": capsule is not None}))
            return 0

    if args.command == "gc":
        print(json.dumps({"ok": True}))
        return 0

    if args.command == "replay":
        index_rows = _load_jsonl_rows(Path(args.index or args.catalog)) if (args.index or args.catalog) else []
        results = []
        for fixture in args.fixtures:
            for payload in _iter_jsonl_payloads(Path(fixture)):
                result = agent.ask(payload["prompt"])
                index_item = _match_index_item(payload["prompt"], index_rows) if index_rows else None
                results.append(
                    {
                        "prompt": payload["prompt"],
                        "status": result.decision.status.value,
                        "score": result.decision.score,
                        "reason": result.decision.reason,
                        "response": result.decision.response.raw_response if result.decision.response else None,
                        "index_match": index_item,
                        "index_response": _index_answer(index_item),
                        "final_answer": (
                            (
                                {
                                    "answer": _index_answer(index_item),
                                    "source": _index_source(index_item),
                                }
                                if isinstance(index_item, dict)
                                else None
                            )
                            if index_item
                            else result.decision.response.raw_response if result.decision.response else None
                        ),
                        "catalog_match": index_item,
                        "catalog_response": _index_answer(index_item),
                    }
                )
        print(json.dumps({"ok": True, "results": results}))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
