from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

from docx import Document
from pypdf import PdfReader


HEADING_RE = re.compile(r"^(第[一二三四五六七八九十百]+章\s*.*|\d+(?:\.\d+){0,3}\s+.*)$")
ABSTRACT_RE = re.compile(r"^(摘要|abstract)$", re.IGNORECASE)
CONCLUSION_RE = re.compile(r"^(结论|研究结论|总结与展望|结论与展望|conclusion[s]?)$", re.IGNORECASE)
REFERENCE_RE = re.compile(r"^(参考文献|references)$", re.IGNORECASE)


@dataclass
class ParagraphUnit:
    paragraph_id: int
    text: str
    page: Optional[int]
    chapter_title: str
    section_title: str
    is_heading: bool = False
    is_abstract: bool = False
    is_conclusion: bool = False
    is_reference: bool = False


@dataclass
class DocumentChunk:
    chunk_id: str
    chapter_title: str
    section_title: str
    page_start: Optional[int]
    page_end: Optional[int]
    paragraph_start: int
    paragraph_end: int
    text: str
    is_abstract: bool = False
    is_conclusion: bool = False
    is_reference: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentIndex:
    title: str
    source_path: str
    chunks: List[DocumentChunk]

    def chapter_titles(self) -> List[str]:
        titles = []
        seen = set()
        for chunk in self.chunks:
            if chunk.chapter_title not in seen:
                titles.append(chunk.chapter_title)
                seen.add(chunk.chapter_title)
        return titles

    def get_chunk(self, chunk_id: str) -> DocumentChunk:
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        raise KeyError(f"Unknown chunk_id: {chunk_id}")

    def get_chunks_by_flag(self, *, abstract: bool = False, conclusion: bool = False) -> List[DocumentChunk]:
        result = []
        for chunk in self.chunks:
            if abstract and chunk.is_abstract:
                result.append(chunk)
            if conclusion and chunk.is_conclusion:
                result.append(chunk)
        return result

    def render_chunk_catalog(self) -> str:
        lines = []
        for c in self.chunks:
            lines.append(
                f"{c.chunk_id} | {c.chapter_title} | {c.section_title} | 页码 {c.page_start}-{c.page_end} | 段落 {c.paragraph_start}-{c.paragraph_end}"
            )
        return "\n".join(lines)

    def context_for_location(self, chunk_id: str, radius: int = 1) -> str:
        idx = next((i for i, c in enumerate(self.chunks) if c.chunk_id == chunk_id), None)
        if idx is None:
            return ""
        start = max(0, idx - radius)
        end = min(len(self.chunks), idx + radius + 1)
        parts = []
        for c in self.chunks[start:end]:
            parts.append(
                f"[{c.chunk_id}] {c.chapter_title} / {c.section_title} / 页 {c.page_start}-{c.page_end} / 段 {c.paragraph_start}-{c.paragraph_end}\n{c.text}"
            )
        return "\n\n".join(parts)


class PaperLoader:
    def __init__(self, chunk_size: int = 3500, overlap: int = 400):
        if overlap >= chunk_size:
            raise ValueError("overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load(self, file_path: str) -> DocumentIndex:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        suffix = path.suffix.lower()
        if suffix == ".docx":
            paragraphs = self._load_docx(path)
        elif suffix == ".pdf":
            paragraphs = self._load_pdf(path)
        elif suffix == ".txt":
            paragraphs = self._load_txt(path)
        else:
            raise ValueError(f"暂不支持的文件类型: {suffix}")

        chunks = self._make_chunks(paragraphs)
        if not chunks:
            raise ValueError("未能从文件中提取到有效文本块。")
        return DocumentIndex(title=path.stem, source_path=str(path), chunks=chunks)

    def _normalize_heading(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _classify_heading(self, text: str) -> Dict[str, bool]:
        cleaned = self._normalize_heading(text)
        return {
            "is_abstract": bool(ABSTRACT_RE.match(cleaned)),
            "is_conclusion": bool(CONCLUSION_RE.match(cleaned)),
            "is_reference": bool(REFERENCE_RE.match(cleaned)),
        }

    def _load_docx(self, path: Path) -> List[ParagraphUnit]:
        doc = Document(str(path))
        paragraph_id = 0
        page = None
        chapter_title = "前置部分"
        section_title = "前置部分"
        in_abstract = False
        in_conclusion = False
        in_reference = False
        units: List[ParagraphUnit] = []

        for p in doc.paragraphs:
            raw = p.text.strip()
            if not raw:
                continue
            style_name = getattr(p.style, "name", "") or ""
            is_heading = "heading" in style_name.lower() or bool(HEADING_RE.match(raw))

            if is_heading:
                flags = self._classify_heading(raw)
                in_abstract = flags["is_abstract"]
                in_conclusion = flags["is_conclusion"]
                in_reference = flags["is_reference"]
                if re.match(r"^第[一二三四五六七八九十百]+章", raw):
                    chapter_title = self._normalize_heading(raw)
                    section_title = chapter_title
                else:
                    section_title = self._normalize_heading(raw)
                    if chapter_title == "前置部分":
                        chapter_title = section_title

            paragraph_id += 1
            units.append(
                ParagraphUnit(
                    paragraph_id=paragraph_id,
                    text=raw,
                    page=page,
                    chapter_title=chapter_title,
                    section_title=section_title,
                    is_heading=is_heading,
                    is_abstract=in_abstract,
                    is_conclusion=in_conclusion,
                    is_reference=in_reference,
                )
            )
        return units

    def _load_txt(self, path: Path) -> List[ParagraphUnit]:
        text = path.read_text(encoding="utf-8")
        lines = [x.strip() for x in re.split(r"\n\s*\n", text) if x.strip()]
        paragraph_id = 0
        chapter_title = "前置部分"
        section_title = "前置部分"
        in_abstract = False
        in_conclusion = False
        in_reference = False
        units: List[ParagraphUnit] = []
        for raw in lines:
            is_heading = bool(HEADING_RE.match(raw)) or len(raw) <= 30
            if is_heading:
                flags = self._classify_heading(raw)
                in_abstract = flags["is_abstract"]
                in_conclusion = flags["is_conclusion"]
                in_reference = flags["is_reference"]
                if re.match(r"^第[一二三四五六七八九十百]+章", raw):
                    chapter_title = self._normalize_heading(raw)
                    section_title = chapter_title
                else:
                    section_title = self._normalize_heading(raw)
            paragraph_id += 1
            units.append(
                ParagraphUnit(
                    paragraph_id=paragraph_id,
                    text=raw,
                    page=None,
                    chapter_title=chapter_title,
                    section_title=section_title,
                    is_heading=is_heading,
                    is_abstract=in_abstract,
                    is_conclusion=in_conclusion,
                    is_reference=in_reference,
                )
            )
        return units

    def _load_pdf(self, path: Path) -> List[ParagraphUnit]:
        reader = PdfReader(str(path))
        paragraph_id = 0
        chapter_title = "前置部分"
        section_title = "前置部分"
        in_abstract = False
        in_conclusion = False
        in_reference = False
        units: List[ParagraphUnit] = []

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
            if not blocks:
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                blocks = lines

            for raw in blocks:
                is_heading = bool(HEADING_RE.match(raw)) or (len(raw) <= 30 and not raw.endswith("。"))
                if is_heading:
                    flags = self._classify_heading(raw)
                    if flags["is_abstract"]:
                        in_abstract = True
                        in_conclusion = False
                        in_reference = False
                    elif flags["is_conclusion"]:
                        in_abstract = False
                        in_conclusion = True
                        in_reference = False
                    elif flags["is_reference"]:
                        in_abstract = False
                        in_conclusion = False
                        in_reference = True
                    elif re.match(r"^第[一二三四五六七八九十百]+章", raw):
                        chapter_title = self._normalize_heading(raw)
                        section_title = chapter_title
                        in_abstract = False
                        in_conclusion = False
                    elif re.match(r"^\d+(?:\.\d+){0,3}\s+", raw):
                        section_title = self._normalize_heading(raw)
                    elif chapter_title == "前置部分":
                        chapter_title = self._normalize_heading(raw)
                        section_title = chapter_title

                paragraph_id += 1
                units.append(
                    ParagraphUnit(
                        paragraph_id=paragraph_id,
                        text=raw,
                        page=page_idx,
                        chapter_title=chapter_title,
                        section_title=section_title,
                        is_heading=is_heading,
                        is_abstract=in_abstract,
                        is_conclusion=in_conclusion,
                        is_reference=in_reference,
                    )
                )
        return units

    def _make_chunks(self, paragraphs: List[ParagraphUnit]) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        current: List[ParagraphUnit] = []
        current_len = 0

        def flush() -> None:
            nonlocal current, current_len
            if not current:
                return
            chunk_id = f"c{len(chunks) + 1:04d}"
            text = "\n\n".join([p.text for p in current])
            page_values = [p.page for p in current if p.page is not None]
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                chapter_title=current[0].chapter_title,
                section_title=current[-1].section_title if current[-1].section_title else current[0].section_title,
                page_start=min(page_values) if page_values else None,
                page_end=max(page_values) if page_values else None,
                paragraph_start=current[0].paragraph_id,
                paragraph_end=current[-1].paragraph_id,
                text=text,
                is_abstract=any(p.is_abstract for p in current),
                is_conclusion=any(p.is_conclusion for p in current),
                is_reference=all(p.is_reference for p in current),
            )
            chunks.append(chunk)
            if self.overlap > 0:
                overlap_units: List[ParagraphUnit] = []
                overlap_len = 0
                for p in reversed(current):
                    overlap_units.insert(0, p)
                    overlap_len += len(p.text)
                    if overlap_len >= self.overlap:
                        break
                current = overlap_units
                current_len = sum(len(p.text) for p in current)
            else:
                current = []
                current_len = 0

        for para in paragraphs:
            para_len = len(para.text)
            chapter_changed = bool(current and para.chapter_title != current[0].chapter_title)
            section_changed = bool(current and para.section_title != current[-1].section_title and para.is_heading)
            if current and (chapter_changed or section_changed or current_len + para_len > self.chunk_size):
                flush()
            current.append(para)
            current_len += para_len

        flush()
        return chunks
