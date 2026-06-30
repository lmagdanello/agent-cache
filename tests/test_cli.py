from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from agent_cache.models import structural_signature_from_prompt


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    current_env = dict(os.environ)
    current_env["PYTHONPATH"] = str(repo_root)
    return subprocess.run(
        [sys.executable, "-m", "agent_cache.cli", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        env=current_env,
        check=True,
    )


def test_cli_end_to_end(tmp_path):
    db = tmp_path / "agent-cache.sqlite3"
    fixture = tmp_path / "fixtures.jsonl"
    fixture.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "Summarize incident INC-123 for region us-east", "expected": "summary"}),
                json.dumps({"prompt": "Find AAA batteries under 30", "expected": "shopping"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        "\n".join(
            [
                json.dumps({"sku": "AAA-001", "name": "AAA batteries", "price": 12, "availability": "in_stock"}),
                json.dumps({"sku": "USBC-002", "name": "USB-C cable", "price": 18, "availability": "in_stock"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    capsule_payload = {
        "capsule_id": "support-1",
        "name": "support",
        "prompt_structure_signature": structural_signature_from_prompt("Summarize incident INC-123 for region us-east"),
        "response_schema": {"type": "object", "required": ["text"]},
        "ttl_seconds": 60,
        "metadata": {
            "prompt_template": "Summarize incident {incident_id} for region {region}",
            "generated_program": {
                "program_id": "gp1",
                "capsule_id": "support-1",
                "kind": "template",
                "template": "Incident {incident_id} in {region}",
                "required_fields": ["incident_id", "region"],
                "allowed_fields": ["incident_id", "region"],
                "transformations": {"region": "upper"},
            },
        },
    }

    init_out = run_cli("init", "--db", str(db), cwd=tmp_path)
    assert json.loads(init_out.stdout)["ok"] is True

    add_out = run_cli("capsules", "add", "--db", str(db), "--json", json.dumps(capsule_payload), cwd=tmp_path)
    assert json.loads(add_out.stdout)["capsule_id"] == "support-1"

    list_out = run_cli("capsules", "list", "--db", str(db), cwd=tmp_path)
    listed = json.loads(list_out.stdout)
    assert listed[0]["capsule_id"] == "support-1"

    inspect_out = run_cli("capsules", "inspect", "support-1", "--db", str(db), cwd=tmp_path)
    inspected = json.loads(inspect_out.stdout)
    assert inspected["enabled"] is True

    show_out = run_cli("capsules", "show", "support-1", "--db", str(db), cwd=tmp_path)
    shown = json.loads(show_out.stdout)
    assert shown["capsule_id"] == "support-1"

    edit_out = run_cli(
        "capsules",
        "edit",
        "support-1",
        "--db",
        str(db),
        "--json",
        json.dumps({"ttl_seconds": 120}),
        cwd=tmp_path,
    )
    assert json.loads(edit_out.stdout)["ok"] is True

    edited_out = run_cli("capsules", "inspect", "support-1", "--db", str(db), cwd=tmp_path)
    edited = json.loads(edited_out.stdout)
    assert edited["ttl_seconds"] == 120

    export_file = tmp_path / "capsules.jsonl"
    export_out = run_cli("capsules", "export", "--db", str(db), "--file", str(export_file), cwd=tmp_path)
    exported = json.loads(export_out.stdout)
    assert exported["ok"] is True
    assert export_file.exists()

    imported_db = tmp_path / "imported.sqlite3"
    run_cli("init", "--db", str(imported_db), cwd=tmp_path)
    import_out = run_cli("capsules", "import", "--db", str(imported_db), "--file", str(export_file), cwd=tmp_path)
    imported = json.loads(import_out.stdout)
    assert imported["imported"] == 1

    doctor_out = run_cli("doctor", "--db", str(imported_db), cwd=tmp_path)
    doctor = json.loads(doctor_out.stdout)
    assert doctor["counts"]["capsules"] == 1
    assert doctor["schema_version"] == "2"
    assert doctor["capsule_health"][0]["capsule_id"] == "support-1"
    assert "hit_rate" in doctor["capsule_health"][0]

    ask_out = run_cli("ask", "Summarize incident INC-123 for region us-east", "--db", str(db), cwd=tmp_path)
    ask = json.loads(ask_out.stdout)
    assert ask["status"] == "HIT_GENERATED"
    assert "reason" in ask

    live_doctor_out = run_cli("doctor", "--db", str(db), cwd=tmp_path)
    live_doctor = json.loads(live_doctor_out.stdout)
    assert live_doctor["telemetry"]["hits"] >= 1

    replay_out = run_cli("replay", str(fixture), "--db", str(db), cwd=tmp_path)
    replay = json.loads(replay_out.stdout)
    assert replay["ok"] is True
    assert len(replay["results"]) == 2
    catalog_replay = run_cli("replay", str(fixture), "--catalog", str(catalog), "--db", str(db), cwd=tmp_path)
    catalog_results = json.loads(catalog_replay.stdout)
    shopping_result = next(item for item in catalog_results["results"] if "AAA batteries" in item["prompt"])
    assert shopping_result["catalog_match"]["sku"] == "AAA-001"
    assert "AAA batteries costs 12" in shopping_result["catalog_response"]

    disable_out = run_cli("capsules", "disable", "support-1", "--db", str(db), cwd=tmp_path)
    assert json.loads(disable_out.stdout)["ok"] is True

    inspect_disabled = run_cli("capsules", "inspect", "support-1", "--db", str(db), cwd=tmp_path)
    assert json.loads(inspect_disabled.stdout)["enabled"] is False

    stats_out = run_cli("stats", "--db", str(db), cwd=tmp_path)
    stats = json.loads(stats_out.stdout)
    assert stats["capsules"] == 1
    assert stats["prompts"] >= 1

    telemetry_out = run_cli("telemetry", "--db", str(db), cwd=tmp_path)
    telemetry = json.loads(telemetry_out.stdout)
    assert "misses" in telemetry

    prometheus_out = run_cli("telemetry", "--db", str(db), "--format", "prometheus", cwd=tmp_path)
    assert "agent_cache_hits" in prometheus_out.stdout


def test_cli_docs_index_replay(tmp_path):
    db = tmp_path / "agent-cache.sqlite3"
    prompts = tmp_path / "pokeapi_prompts.jsonl"
    prompts.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "How does PokéAPI pagination work?", "expected": "Resource pagination"}),
                json.dumps({"prompt": "How do I fetch a Pokémon by name?", "expected": "Pokemon endpoint"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "pagination.md").write_text(
        "# Resource pagination\n\nList endpoints are paginated and support limit/offset.\n",
        encoding="utf-8",
    )
    (docs_dir / "pokemon.html").write_text(
        "<html><body><h1>Pokemon endpoint</h1><p>Fetch Pokémon data by id or name.</p></body></html>",
        encoding="utf-8",
    )
    index = tmp_path / "pokeapi_index.jsonl"

    run_cli("init", "--db", str(db), cwd=tmp_path)
    ingest = run_cli(
        "ingest",
        str(docs_dir / "pagination.md"),
        str(docs_dir / "pokemon.html"),
        "--output",
        str(index),
        "--db",
        str(db),
        cwd=tmp_path,
    )
    ingest_payload = json.loads(ingest.stdout)
    assert ingest_payload["ok"] is True
    assert ingest_payload["docs_capsule_id"].startswith("docs::")

    ask_out = run_cli("ask", "How does PokéAPI pagination work?", "--db", str(db), cwd=tmp_path)
    ask = json.loads(ask_out.stdout)
    assert ask["status"] == "HIT_EXACT"
    assert "paginated" in ask["response"]
    assert ask["source"].endswith("pagination.md")

    out = run_cli("replay", str(prompts), "--index", str(index), "--db", str(db), cwd=tmp_path)
    result = json.loads(out.stdout)
    assert result["ok"] is True
    assert result["results"][0]["index_match"]["name"] == "Resource pagination"
    assert "paginated" in result["results"][0]["index_response"]
    assert result["results"][0]["final_answer"]["source"].endswith("pagination.md")
    assert result["results"][1]["index_match"]["name"] == "Pokemon endpoint"


def test_docs_ingest_supports_docx_xlsx_csv(tmp_path):
    from zipfile import ZipFile

    from agent_cache.docs import ingest_sources

    md = tmp_path / "api.md"
    md.write_text("# Rate limits\n\nUse limit and offset for list endpoints.\n", encoding="utf-8")

    csv_path = tmp_path / "api.csv"
    csv_path.write_text("name,summary\nPokemon endpoint,Fetch by id or name.\n", encoding="utf-8")

    html = tmp_path / "api.html"
    html.write_text("<html><body><h1>Berry endpoint</h1><p>Fetch berry data.</p></body></html>", encoding="utf-8")

    docx = tmp_path / "api.docx"
    with ZipFile(docx, "w") as zf:
        zf.writestr(
            "word/document.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:body><w:p><w:r><w:t>Docs endpoint</w:t></w:r></w:p>
              <w:p><w:r><w:t>Fetch by id or name.</w:t></w:r></w:p></w:body>
            </w:document>""",
        )

    xlsx = tmp_path / "api.xlsx"
    with ZipFile(xlsx, "w") as zf:
        zf.writestr(
            "xl/sharedStrings.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="2" uniqueCount="2">
              <si><t>Pagination</t></si>
              <si><t>limit offset</t></si>
            </sst>""",
        )
        zf.writestr(
            "xl/worksheets/sheet1.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <sheetData>
                <row r="1">
                  <c r="A1" t="s"><v>0</v></c>
                  <c r="B1" t="s"><v>1</v></c>
                </row>
              </sheetData>
            </worksheet>""",
        )

    index = tmp_path / "docs_index.jsonl"
    rows = ingest_sources([str(md), str(csv_path), str(html), str(docx), str(xlsx)], index)
    assert rows >= 5
    lines = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any("Rate limits" in row["title"] for row in lines)
    assert any("Pokemon endpoint" in row["title"] for row in lines)
    assert any("Docs endpoint" in row["title"] for row in lines)


def test_cli_ingest_accepts_file_url(tmp_path):
    docs = tmp_path / "docs.md"
    docs.write_text("# API overview\n\nThe docs expose a simple resource tree.\n", encoding="utf-8")
    index = tmp_path / "index.jsonl"

    out = run_cli("ingest", docs.as_uri(), "--output", str(index), cwd=tmp_path)
    payload = json.loads(out.stdout)
    assert payload["ok"] is True
    assert payload["rows"] >= 1
    assert index.exists()
