"""Load locally collected HTML and text files without crawling websites."""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

from src.models import PageRecord


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._hidden_depth += 1
        elif tag in {"p", "br", "div", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._hidden_depth:
            self._hidden_depth -= 1
        elif tag in {"p", "div", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.parts.append(data)


def load_web_file(path: Path, data_root: Path) -> tuple[str, str, PageRecord, str | None]:
    content = path.read_bytes()
    file_hash = hashlib.sha256(content).hexdigest()
    relative = path.resolve().relative_to(data_root.resolve()).as_posix().casefold()
    document_id = "doc_" + hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
    decoded = content.decode("utf-8-sig", errors="replace")
    if path.suffix.casefold() in {".html", ".htm"}:
        parser = _TextExtractor()
        parser.feed(decoded)
        decoded = "".join(parser.parts)
    url_file = path.with_suffix(path.suffix + ".url")
    source_url = url_file.read_text(encoding="utf-8").strip() if url_file.exists() else ""
    parsed_url = urlsplit(source_url)
    if source_url and (parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc):
        raise ValueError(f"URL nguồn không hợp lệ trong {url_file}")
    return (
        document_id,
        file_hash,
        PageRecord(document_id=document_id, page_no=1, text=decoded.strip()),
        source_url or None,
    )
