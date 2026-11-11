"""Step 1: read the user's documents and cut them into chunks."""

from __future__ import annotations

import json
import re
from pathlib import Path

TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".html", ".htm"}


def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    if ext == ".docx":
        try:
            import docx  # python-docx
        except ImportError as e:
            raise ImportError("Install python-docx to read .docx files") from e
        return "\n\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    if ext in {".html", ".htm"}:
        text = path.read_text(errors="ignore")
        text = re.sub(r"<(script|style).*?</\1>", " ", text, flags=re.S | re.I)
        return re.sub(r"<[^>]+>", " ", text)
    return path.read_text(errors="ignore")


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> list[str]:
    """Split on paragraph boundaries, keeping chunks under max_chars."""
    text = re.sub(r"[ \t]+", " ", text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for para in paras:
        while len(para) > max_chars:  # very long paragraph: hard split
            chunks.append(para[:max_chars])
            para = para[max_chars - overlap :]
        if len(cur) + len(para) + 2 > max_chars and cur:
            chunks.append(cur)
            tail = cur[-overlap:]
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail  # don't start mid-word
            cur = tail + "\n\n" + para if overlap else para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        chunks.append(cur)
    return chunks


def ingest(docs_dir: Path, out_file: Path, max_chars: int = 1800, overlap: int = 200) -> int:
    files = sorted(p for p in docs_dir.rglob("*") if p.is_file() and not p.name.startswith("."))
    n = 0
    with out_file.open("w") as f:
        for fp in files:
            if fp.suffix.lower() not in TEXT_EXT | {".pdf", ".docx"}:
                print(f"  skip (unsupported type): {fp.name}")
                continue
            text = read_file(fp)
            for i, ch in enumerate(chunk_text(text, max_chars, overlap)):
                rec = {"id": f"{fp.stem}-{i}", "source": str(fp.relative_to(docs_dir)), "text": ch}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return n
