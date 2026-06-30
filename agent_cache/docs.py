from __future__ import annotations

import csv
import html
import io
import json
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


STOPWORDS = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "to",
    "of",
    "for",
    "in",
    "on",
    "with",
    "by",
    "from",
    "is",
    "are",
    "be",
    "as",
    "at",
    "this",
    "that",
    "it",
    "use",
    "api",
    "docs",
    "documentation",
}


@dataclass(slots=True)
class DocChunk:
    title: str
    section: str
    text: str
    source: str
    kind: str


def ingest_sources(sources: list[str], output: Path, *, base_url: str | None = None) -> int:
    rows: list[dict] = []
    for source in sources:
        source_path = Path(source)
        if _is_url(source):
            text, content_type = _load_url(source)
            title = _title_from_text(text, fallback=_title_from_source(source, content_type=content_type))
            chunks = _chunk_text(text)
            rows.extend(_chunks_to_index_rows(chunks, title=title, source=source, kind="url"))
            continue
        text = _extract_text_from_file(source_path)
        title = _title_from_text(text, fallback=_title_from_source(source))
        chunks = _chunk_text(text)
        rows.extend(_chunks_to_index_rows(chunks, title=title, source=str(source_path), kind=source_path.suffix.lower().lstrip(".")))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def ingest_url(url: str, output: Path) -> int:
    return ingest_sources([url], output)


def _chunks_to_index_rows(chunks: list[str], *, title: str, source: str, kind: str) -> list[dict]:
    rows: list[dict] = []
    for idx, chunk in enumerate(chunks):
        text = " ".join(chunk.split())
        if not text:
            continue
        keywords = _keywords(text, title=title)
        rows.append(
            {
                "name": title if idx == 0 else f"{title} section {idx + 1}",
                "title": title,
                "summary": _summary(text),
                "description": text[:400],
                "keywords": keywords,
                "url": source if _is_url(source) else None,
                "source": source,
                "kind": kind,
                "section": idx + 1,
                "content": text,
            }
        )
    return rows


def _chunk_text(text: str) -> list[str]:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                chunks.append("\n".join(current).strip())
                current = []
            continue
        if stripped.startswith("#") and current:
            chunks.append("\n".join(current).strip())
            current = [stripped]
            continue
        current.append(stripped)
    if current:
        chunks.append("\n".join(current).strip())
    if not chunks:
        chunks = [text.strip()]
    merged: list[str] = []
    index = 0
    while index < len(chunks):
        chunk = chunks[index].strip()
        if not chunk:
            index += 1
            continue
        if chunk.startswith("#") and index + 1 < len(chunks):
            next_chunk = chunks[index + 1].strip()
            if next_chunk and not next_chunk.startswith("#"):
                merged.append(f"{chunk}\n{next_chunk}".strip())
                index += 2
                continue
        merged.append(chunk)
        index += 1
    return merged


def _summary(text: str) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        return ""
    if len(normalized) <= 220:
        return normalized
    return normalized[:217].rsplit(" ", 1)[0] + "..."


def _keywords(text: str, *, title: str = "") -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", f"{title} {text}".lower())
    counts = Counter(token for token in tokens if len(token) > 2 and token not in STOPWORDS)
    return [token for token, _ in counts.most_common(8)]


def _title_from_source(source: str, *, content_type: str | None = None) -> str:
    if _is_url(source):
        parsed = urllib.parse.urlparse(source)
        stem = Path(parsed.path).stem
    else:
        stem = Path(source).stem
    return stem.replace("_", " ").replace("-", " ").title() or "Document"


def _title_from_text(text: str, *, fallback: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    if not lines:
        return fallback
    first = lines[0]
    if first.startswith("#"):
        cleaned = first.lstrip("#").strip()
        if cleaned:
            return cleaned
    if len(first) <= 96:
        return first.strip(" :-")
    return fallback


def _is_url(value: str) -> bool:
    return urllib.parse.urlparse(value).scheme in {"http", "https", "file"}


def _load_url(url: str) -> tuple[str, str | None]:
    with urllib.request.urlopen(url) as response:  # nosec - URL comes from user or local docs workflow
        data = response.read()
        content_type = response.headers.get_content_type()
        encoding = response.headers.get_content_charset() or "utf-8"
    if content_type.startswith("text/") or content_type in {"application/json", "application/xml"}:
        text = data.decode(encoding, errors="replace")
        if content_type == "text/html":
            return _extract_html(text), content_type
        return text, content_type
    return _decode_binary_document(data, url=url, content_type=content_type)


def _extract_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt", ".rst", ".toml", ".yaml", ".yml", ".ini"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".csv":
        return _extract_csv(path)
    if suffix in {".html", ".htm"}:
        return _extract_html(path.read_text(encoding="utf-8", errors="replace"))
    if suffix == ".json":
        return _flatten_json(json.loads(path.read_text(encoding="utf-8", errors="replace")))
    if suffix == ".xml":
        return _strip_xml(path.read_text(encoding="utf-8", errors="replace"))
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix == ".xlsx":
        return _extract_xlsx(path)
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff", ".bmp"}:
        return _extract_image(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _decode_binary_document(data: bytes, *, url: str, content_type: str | None = None) -> tuple[str, str | None]:
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix or ".bin", delete=True) as handle:
        handle.write(data)
        handle.flush()
        path = Path(handle.name)
        return _extract_text_from_file(path), content_type


def _extract_csv(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(", ".join(f"{key}: {value}" for key, value in row.items()))
        if rows:
            return "\n".join(rows)
    return path.read_text(encoding="utf-8", errors="replace")


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_html(text: str) -> str:
    title_match = re.search(r"(?is)<(?:h1|title)[^>]*>(.*?)</(?:h1|title)>", text)
    body = _strip_html(text)
    if title_match:
        title = _strip_html(title_match.group(1))
        if title and body and not body.startswith(title):
            return f"{title}\n{body}"
        if title:
            return title if not body else f"{title}\n{body}"
    return body


def _strip_xml(text: str) -> str:
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _flatten_json(value: object, prefix: str = "") -> str:
    parts: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            parts.append(_flatten_json(item, f"{prefix}{key}."))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            parts.append(_flatten_json(item, f"{prefix}{idx}."))
    else:
        parts.append(f"{prefix[:-1]}: {value}")
    return "\n".join(part for part in parts if part)


def _extract_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        with archive.open("word/document.xml") as document:
            xml = document.read()
    root = ET.fromstring(xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    texts = [node.text for node in root.findall(".//w:t", namespace) if node.text]
    return "\n".join(texts)


def _extract_xlsx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        shared_strings = _load_shared_strings(archive)
        sheet_names = [name for name in archive.namelist() if name.startswith("xl/worksheets/sheet")]
        rows: list[str] = []
        for sheet_name in sorted(sheet_names):
            with archive.open(sheet_name) as sheet:
                root = ET.fromstring(sheet.read())
            namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for row in root.findall(".//x:row", namespace):
                values: list[str] = []
                for cell in row.findall("x:c", namespace):
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find("x:v", namespace)
                    inline_node = cell.find("x:is/x:t", namespace)
                    if inline_node is not None and inline_node.text:
                        values.append(inline_node.text)
                    elif value_node is not None and value_node.text is not None:
                        raw = value_node.text
                        if cell_type == "s":
                            try:
                                values.append(shared_strings[int(raw)])
                            except (ValueError, IndexError):
                                values.append(raw)
                        else:
                            values.append(raw)
                if values:
                    rows.append(" | ".join(values))
    return "\n".join(rows)


def _load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    with archive.open("xl/sharedStrings.xml") as handle:
        root = ET.fromstring(handle.read())
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    return ["".join(node.itertext()) for node in root.findall(".//x:si", namespace)]


def _extract_pdf(path: Path) -> str:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception:
        pdf_to_text = shutil.which("pdftotext")
        if pdf_to_text is None:
            raise ValueError(f"PDF extraction is unavailable for {path}; install PyPDF2 or pdftotext")
        proc = subprocess.run([pdf_to_text, str(path), "-"], check=True, capture_output=True, text=True)
        return proc.stdout
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _extract_image(path: Path) -> str:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        tesseract = shutil.which("tesseract")
        if tesseract is None:
            raise ValueError(f"Image extraction is unavailable for {path}; install Pillow+pytesseract or tesseract")
        proc = subprocess.run([tesseract, str(path), "stdout"], check=True, capture_output=True, text=True)
        return proc.stdout
    try:
        import pytesseract  # type: ignore
    except Exception:
        tesseract = shutil.which("tesseract")
        if tesseract is None:
            raise ValueError(f"Image OCR is unavailable for {path}; install pytesseract or tesseract")
        proc = subprocess.run([tesseract, str(path), "stdout"], check=True, capture_output=True, text=True)
        return proc.stdout
    image = Image.open(path)
    return pytesseract.image_to_string(image)
