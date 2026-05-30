"""Разбиение документов на чанки с перекрытием и метаданными.

По заданию используется логика ``RecursiveCharacterTextSplitter`` из LangChain.
Если LangChain установлен — берём его сплиттер; иначе работает эквивалентная
реализация на чистом Python (та же идея: рекурсивно режем по separators
\\n\\n → \\n → ". " → " ", добиваем перекрытие overlap).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass
class Document:
    text: str
    source: str           # имя файла
    title: str            # заголовок (первая строка / имя)
    extra: Dict = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    title: str
    chunk_index: int

    def to_meta(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "chunk_index": self.chunk_index,
        }


def _chunk_id(source: str, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{source}:{idx}:{text}".encode("utf-8")).hexdigest()[:10]
    return f"{Path(source).stem}-{idx}-{h}"


def _split_pure(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Рекурсивный сплиттер на чистом Python (запасной вариант)."""
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _recurse(s: str, seps: Sequence[str]) -> List[str]:
        s = s.strip()
        if len(s) <= chunk_size or not s:
            return [s] if s else []
        sep = seps[0] if seps else ""
        rest = seps[1:] if len(seps) > 1 else [""]
        if sep == "":
            # режем жёстко по символам
            return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]
        parts = s.split(sep)
        pieces: List[str] = []
        for part in parts:
            if len(part) > chunk_size:
                pieces.extend(_recurse(part, rest))
            elif part:
                pieces.append(part)
        # склеиваем мелкие части до размера чанка
        merged: List[str] = []
        buf = ""
        for piece in pieces:
            candidate = (buf + sep + piece).strip(sep) if buf else piece
            if len(candidate) <= chunk_size:
                buf = candidate
            else:
                if buf:
                    merged.append(buf)
                buf = piece
        if buf:
            merged.append(buf)
        return merged

    raw = _recurse(text, separators)

    # добавляем перекрытие: к каждому чанку приклеиваем хвост предыдущего
    if overlap > 0 and len(raw) > 1:
        with_overlap: List[str] = []
        for i, ch in enumerate(raw):
            if i == 0:
                with_overlap.append(ch)
            else:
                tail = raw[i - 1][-overlap:]
                with_overlap.append((tail + " " + ch).strip())
        return with_overlap
    return raw


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return [c for c in splitter.split_text(text) if c.strip()]
    except Exception:
        return _split_pure(text, chunk_size, overlap)


def chunk_documents(
    docs: Sequence[Document], chunk_size: int, overlap: int
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in docs:
        for idx, piece in enumerate(split_text(doc.text, chunk_size, overlap)):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    id=_chunk_id(doc.source, idx, piece),
                    text=piece,
                    source=doc.source,
                    title=doc.title,
                    chunk_index=idx,
                )
            )
    return chunks


def load_documents(kb_dir: Path) -> List[Document]:
    """Читает все *.md / *.txt из папки базы знаний как отдельные документы."""
    docs: List[Document] = []
    for path in sorted(kb_dir.glob("**/*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        first_line = text.splitlines()[0].lstrip("# ").strip()
        title = first_line or path.stem
        docs.append(Document(text=text, source=path.name, title=title))
    return docs
