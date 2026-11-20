"""Load documents into segments (with page/section info) and split them into chunks."""
import re
from dataclasses import dataclass
from pathlib import Path

import config


MIN_CHUNK_WORDS = 8  # skip fragments too short to be useful


@dataclass
class Segment:
    source: str    # file name
    location: str  # "p. 4", "Section: Annual Leave", or "" if unknown
    text: str


@dataclass
class Chunk:
    text: str
    source: str
    location: str
    index: int  # position within the source file


def load_pdf(path: Path) -> list[Segment]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    segments = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            segments.append(Segment(path.name, f"p. {page_number}", text))
    return segments


def load_docx(path: Path) -> list[Segment]:
    from docx import Document

    doc = Document(str(path))
    segments: list[Segment] = []
    buffer: list[str] = []
    heading = ""

    def flush():
        text = "\n\n".join(buffer).strip()
        if text:
            location = f"Section: {heading}" if heading else ""
            segments.append(Segment(path.name, location, text))
        buffer.clear()

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name if para.style is not None and para.style.name else ""
        if style.startswith("Heading") or style == "Title":
            flush()
            heading = text
            buffer.append(text)  # keep the heading in the chunk text for better retrieval
        else:
            buffer.append(text)
    flush()

    heading = "Tables"
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                buffer.append(" | ".join(cells))
    flush()
    return segments


_HEADING_RE = re.compile(r"^#{1,6}\s+(.*\S)\s*$")


def load_text(path: Path) -> list[Segment]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    is_markdown = path.suffix.lower() == ".md"
    segments: list[Segment] = []
    buffer: list[str] = []
    heading = ""

    def flush():
        text = "\n".join(buffer).strip()
        if text:
            location = f"Section: {heading}" if heading else ""
            segments.append(Segment(path.name, location, text))
        buffer.clear()

    for line in raw.splitlines():
        match = _HEADING_RE.match(line) if is_markdown else None
        if match:
            flush()
            heading = match.group(1)
            buffer.append(heading)
        else:
            buffer.append(line)
    flush()
    return segments


def load_file(path: Path) -> list[Segment]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".docx":
        return load_docx(path)
    if suffix in {".txt", ".md"}:
        return load_text(path)
    raise ValueError(f"Unsupported file type: {path.name}")


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Pack paragraphs into chunks of about `size` words, carrying `overlap` words forward."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []

    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > size:  # a single huge paragraph: hard split with overlap
            if current:
                chunks.append(" ".join(current))
                current = []
            step = size - overlap
            for start in range(0, len(words), step):
                chunks.append(" ".join(words[start : start + size]))
                if start + size >= len(words):
                    break
            continue
        if current and len(current) + len(words) > size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.extend(words)

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_segments(segments: list[Segment]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for segment in segments:
        for text in chunk_text(segment.text, config.CHUNK_WORDS, config.CHUNK_OVERLAP_WORDS):
            if len(text.split()) < MIN_CHUNK_WORDS:  # e.g. a title with no body text
                continue
            chunks.append(Chunk(text, segment.source, segment.location, len(chunks)))
    return chunks
