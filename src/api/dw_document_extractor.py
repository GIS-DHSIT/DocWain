import hashlib
import io
import logging
import re
from typing import List, Optional, Tuple, Union

import docx
import fitz
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from pptx import Presentation
try:
    import pdfplumber
except Exception:  # noqa: BLE001
    pdfplumber = None
from src.api.config import Config
from src.api.pipeline_models import ChunkCandidate, ExtractedDocument, Figure, Section, Table

logger = logging.getLogger(__name__)

try:
    import easyocr
except Exception:  # noqa: BLE001
    easyocr = None


class DocumentExtractor:
    """Layout-aware document extractor with selective OCR and structured output."""

    def __init__(self, ocr_engine: Optional[str] = None):
        self.ocr_engine = (ocr_engine or getattr(Config.Model, "OCR_ENGINE", "pytesseract")).lower()
        self._easyocr_reader = None
        self._doc_intel = DocumentIntelligence()

    # ---------- Helpers ----------
    @staticmethod
    def _is_heading(text: str) -> bool:
        if not text:
            return False
        clean = text.strip()
        if len(clean.split()) <= 1 and clean.isupper():
            return True
        heading_pattern = re.compile(r"^(\d+\.)+\s+.+|^[A-Z][A-Z0-9\s,:-]{4,}$")
        return bool(heading_pattern.match(clean))

    @staticmethod
    def _looks_like_table(text: str) -> bool:
        if not text:
            return False
        separators = text.count("|") + text.count(",") + text.count("\t")
        return separators >= 4

    @staticmethod
    def _make_section_id(title: str, page: int) -> str:
        raw = f"{title}|{page}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _detect_scanned_page(text: str, image_count: int) -> bool:
        return (not text or len(text.strip()) < 30) and image_count > 0

    def _ensure_easyocr(self):
        if easyocr and self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader(["en"], gpu=False)

    def _ocr_image(self, image: Image.Image) -> Tuple[str, Optional[float]]:
        if self.ocr_engine == "easyocr" and easyocr:
            try:
                self._ensure_easyocr()
                result = self._easyocr_reader.readtext(np.array(image), detail=1)
                texts = []
                confidences = []
                for _, text, conf in result:
                    if text:
                        texts.append(text)
                    if conf is not None:
                        confidences.append(float(conf) * 100.0)
                text_value = " ".join(texts).strip()
                avg_conf = sum(confidences) / len(confidences) if confidences else None
                return text_value, avg_conf
            except Exception as exc:  # noqa: BLE001
                logger.warning("EasyOCR failed, falling back to tesseract: %s", exc)
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            words = [w.strip() for w in data.get("text", []) if w.strip()]
            confidences = []
            for conf in data.get("conf", []):
                try:
                    conf_val = float(conf)
                except (TypeError, ValueError):
                    continue
                if conf_val >= 0:
                    confidences.append(conf_val)
            text_value = " ".join(words).strip()
            avg_conf = sum(confidences) / len(confidences) if confidences else None
            return text_value, avg_conf
        except Exception as exc:  # noqa: BLE001
            logger.error("OCR failed: %s", exc)
            return "", None

    def _finalize_section(
        self,
        sections: List[Section],
        title: str,
        section_id: str,
        start_page: int,
        end_page: int,
        buffer: List[str],
    ):
        if buffer:
            sections.append(
                Section(
                    section_id=section_id,
                    title=title,
                    level=1,
                    start_page=start_page,
                    end_page=end_page,
                    text="\n".join(buffer).strip(),
                )
            )

    @staticmethod
    def _coerce_file_like(content: object):
        """
        Normalize raw payloads into a seekable stream for libraries like python-docx/pptx.
        Handles bytes, memoryview, and already-open file-like objects.
        """
        if isinstance(content, memoryview):
            content = content.tobytes()
        if isinstance(content, (bytes, bytearray)):
            return io.BytesIO(content)
        if isinstance(content, str):
            # Treat as raw text payload, not a path; decode to bytes
            return io.BytesIO(content.encode("utf-8"))
        if hasattr(content, "read"):
            try:
                content.seek(0)
            except Exception:
                pass
            return content
        try:
            return io.BytesIO(bytes(content))
        except Exception:
            return io.BytesIO()

    # ---------- Extraction routines ----------
    def extract_text_from_pdf(self, pdf_content: bytes, filename: Optional[str] = None) -> ExtractedDocument:
        sections: List[Section] = []
        tables: List[Table] = []
        figures: List[Figure] = []
        chunk_candidates: List[ChunkCandidate] = []
        full_text_parts: List[str] = []
        errors: List[str] = []
        ocr_confidences: List[float] = []

        current_title = "Introduction"
        current_section_id = self._make_section_id(current_title, 1)
        current_start_page = 1
        section_buffer: List[str] = []
        last_page = 1

        if hasattr(fitz, "open"):
            try:
                with fitz.open(stream=pdf_content, filetype="pdf") as doc:
                    for page_index, page in enumerate(doc, start=1):
                        last_page = page_index
                        blocks = page.get_text("blocks") or []
                        page_text_parts: List[str] = []

                    for block in blocks:
                        if len(block) < 5:
                            continue
                        block_text = (block[4] or "").strip()
                        if not block_text:
                            continue

                        first_line = block_text.splitlines()[0].strip()
                        if self._is_heading(first_line):
                            self._finalize_section(
                                sections, current_title, current_section_id, current_start_page, page_index, section_buffer
                            )
                            section_buffer = []
                            current_title = first_line
                            current_section_id = self._make_section_id(first_line, page_index)
                            current_start_page = page_index
                            continue

                        section_buffer.append(block_text)
                        chunk_type = "table" if self._looks_like_table(block_text) else "text"
                        chunk_candidates.append(
                            ChunkCandidate(
                                text=block_text,
                                page=page_index,
                                section_title=current_title,
                                section_id=current_section_id,
                                chunk_type=chunk_type,
                            )
                        )
                        page_text_parts.append(block_text)
                        if chunk_type == "table":
                            tables.append(Table(page=page_index, text=block_text))

                        images = page.get_images(full=True)
                        page_text = "\n".join(page_text_parts).strip()
                        if self._detect_scanned_page(page_text, len(images)):
                            for img in images:
                                try:
                                    base_image = doc.extract_image(img[0])
                                    image = Image.open(io.BytesIO(base_image["image"]))
                                    ocr_text, ocr_conf = self._ocr_image(image)
                                    if ocr_text:
                                        figures.append(Figure(page=page_index, caption=ocr_text))
                                        chunk_candidates.append(
                                            ChunkCandidate(
                                                text=ocr_text,
                                                page=page_index,
                                                section_title=current_title,
                                                section_id=current_section_id,
                                                chunk_type="ocr_text",
                                            )
                                        )
                                        page_text_parts.append(ocr_text)
                                    if ocr_conf is not None:
                                        ocr_confidences.append(float(ocr_conf))
                                except Exception as exc:  # noqa: BLE001
                                    logger.debug("OCR image extraction failed on page %s: %s", page_index, exc)
                                    errors.append(f"image_extraction_failed: page={page_index} err={exc}")

                        for img_index, _ in enumerate(images):
                            figures.append(Figure(page=page_index, caption=f"Image_{page_index}_{img_index}"))

                        if page_text_parts:
                            full_text_parts.append(f"\n--- Page {page_index} ---\n" + "\n".join(page_text_parts))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"pdf_layout_parse_failed: {exc}")
        if pdfplumber:
            try:
                with pdfplumber.open(io.BytesIO(pdf_content)) as doc:
                    for page_index, page in enumerate(doc.pages, start=1):
                        last_page = max(last_page, page_index)
                        text = page.extract_text() or ""
                        if text.strip():
                            for para in text.split("\n"):
                                para = para.strip()
                                if not para:
                                    continue
                                if self._is_heading(para):
                                    self._finalize_section(
                                        sections, current_title, current_section_id, current_start_page, page_index, section_buffer
                                    )
                                    section_buffer = []
                                    current_title = para
                                    current_section_id = self._make_section_id(para, page_index)
                                    current_start_page = page_index
                                    continue
                                section_buffer.append(para)
                                chunk_type = "table" if self._looks_like_table(para) else "text"
                                chunk_candidates.append(
                                    ChunkCandidate(
                                        text=para,
                                        page=page_index,
                                        section_title=current_title,
                                        section_id=current_section_id,
                                        chunk_type=chunk_type,
                                    )
                                )
                                if chunk_type == "table":
                                    tables.append(Table(page=page_index, text=para))
                            full_text_parts.append(f"\n--- Page {page_index} ---\n{text}")

                        try:
                            extracted_tables = page.extract_tables()
                            if extracted_tables:
                                for tbl in extracted_tables:
                                    formatted_table = "\n".join([", ".join(row) for row in tbl])
                                    tables.append(Table(page=page_index, text=formatted_table, csv=formatted_table))
                                    chunk_candidates.append(
                                        ChunkCandidate(
                                            text=formatted_table,
                                            page=page_index,
                                            section_title=current_title,
                                            section_id=current_section_id,
                                            chunk_type="table",
                                        )
                                    )
                        except Exception as exc:  # noqa: BLE001
                            errors.append(f"pdf_table_extract_failed: page={page_index} err={exc}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"pdfplumber_open_failed: {exc}")
        else:
            errors.append("No PDF parser available: PyMuPDF/pdfplumber missing")

        self._finalize_section(
            sections, current_title, current_section_id, current_start_page, last_page, section_buffer
        )

        full_text = "\n".join(full_text_parts).strip()
        doc_type = self._doc_intel.infer_type(tables, figures, sections, full_text, filename_hint=filename or "document.pdf")
        return ExtractedDocument(
            full_text=full_text,
            sections=sections,
            tables=tables,
            figures=figures,
            chunk_candidates=chunk_candidates,
            doc_type=doc_type,
            errors=errors,
            metrics={"ocr_confidences": ocr_confidences},
        )

    def extract_text_from_docx(self, doc_content: Union[bytes, bytearray, memoryview, str, io.IOBase], filename: Optional[str] = None) -> ExtractedDocument:
        document = docx.Document(self._coerce_file_like(doc_content))
        sections: List[Section] = []
        tables: List[Table] = []
        figures: List[Figure] = []
        chunk_candidates: List[ChunkCandidate] = []
        full_text_parts: List[str] = []
        errors: List[str] = []

        current_title = "Introduction"
        current_section_id = self._make_section_id(current_title, 1)
        section_buffer: List[str] = []

        for para in document.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            style = para.style.name if para.style else ""
            if style.startswith("Heading") or self._is_heading(text):
                self._finalize_section(sections, current_title, current_section_id, 1, 1, section_buffer)
                section_buffer = []
                current_title = text
                current_section_id = self._make_section_id(current_title, 1)
                continue

            section_buffer.append(text)
            chunk_candidates.append(
                ChunkCandidate(
                    text=text,
                    page=None,
                    section_title=current_title,
                    section_id=current_section_id,
                    chunk_type="text",
                )
            )
            full_text_parts.append(text)

        # Tables
        for tbl in document.tables:
            rows = []
            for row in tbl.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    rows.append(" | ".join(row_text))
            if rows:
                csv_like = "\n".join(rows)
                tables.append(Table(page=None, text=csv_like, csv=csv_like))
                chunk_candidates.append(
                    ChunkCandidate(
                        text=csv_like,
                        page=None,
                        section_title=current_title,
                        section_id=current_section_id,
                        chunk_type="table",
                    )
                )
                full_text_parts.append(csv_like)

        self._finalize_section(sections, current_title, current_section_id, 1, 1, section_buffer)

        # Figures via document relationships (captions/alt-text if present)
        try:
            for rel in document.part.rels.values():
                if "image" in rel.target_ref:
                    caption = getattr(rel, "alt_text", None) or rel.target_ref
                    figures.append(Figure(page=None, caption=caption))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"docx_image_parse_failed: {exc}")

        try:
            for shape in getattr(document, "inline_shapes", []):
                alt_text = getattr(shape, "alt_text", "") or getattr(shape, "description", "")
                if alt_text:
                    figures.append(Figure(page=None, caption=alt_text))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"docx_inline_shapes_failed: {exc}")

        full_text = "\n".join(full_text_parts).strip()
        doc_type = self._doc_intel.infer_type(tables, figures, sections, full_text, filename_hint=filename or "document.docx")
        return ExtractedDocument(
            full_text=full_text,
            sections=sections,
            tables=tables,
            figures=figures,
            chunk_candidates=chunk_candidates,
            doc_type=doc_type,
            errors=errors,
        )

    def extract_text_from_pptx(self, ppt_content: Union[bytes, bytearray, memoryview, str, io.IOBase], filename: Optional[str] = None) -> ExtractedDocument:
        presentation = Presentation(self._coerce_file_like(ppt_content))
        sections: List[Section] = []
        tables: List[Table] = []
        figures: List[Figure] = []
        chunk_candidates: List[ChunkCandidate] = []
        full_text_parts: List[str] = []
        errors: List[str] = []

        for slide_num, slide in enumerate(presentation.slides, start=1):
            slide_title = f"Slide {slide_num}"
            section_id = self._make_section_id(slide_title, slide_num)
            slide_text_parts: List[str] = []

            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text = shape.text.strip()
                    slide_text_parts.append(text)
                    chunk_candidates.append(
                        ChunkCandidate(
                            text=text,
                            page=slide_num,
                            section_title=slide_title,
                            section_id=section_id,
                            chunk_type="text",
                        )
                    )

                if shape.has_table:
                    rows = []
                    for row in shape.table.rows:
                        row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if row_text:
                            rows.append(" | ".join(row_text))
                    if rows:
                        csv_like = "\n".join(rows)
                        tables.append(Table(page=slide_num, text=csv_like, csv=csv_like))
                        chunk_candidates.append(
                            ChunkCandidate(
                                text=csv_like,
                                page=slide_num,
                                section_title=slide_title,
                                section_id=section_id,
                                chunk_type="table",
                            )
                        )
                        slide_text_parts.append(csv_like)

                if getattr(shape, "shape_type", None) == 13 and getattr(shape, "image", None):
                    caption = getattr(shape, "name", "") or shape.image.filename or "Figure"
                    figures.append(Figure(page=slide_num, caption=caption))

            if slide_text_parts:
                full_text_parts.append(f"\n--- Slide {slide_num} ---\n" + "\n".join(slide_text_parts))
                sections.append(
                    Section(
                        section_id=section_id,
                        title=slide_title,
                        level=1,
                        start_page=slide_num,
                        end_page=slide_num,
                        text="\n".join(slide_text_parts),
                    )
                )

        full_text = "\n".join(full_text_parts).strip()
        doc_type = self._doc_intel.infer_type(tables, figures, sections, full_text, filename_hint=filename or "slides.pptx")
        return ExtractedDocument(
            full_text=full_text,
            sections=sections,
            tables=tables,
            figures=figures,
            chunk_candidates=chunk_candidates,
            doc_type=doc_type,
            errors=errors,
        )

    def extract_text_from_txt(self, text_content: Union[bytes, str], filename: Optional[str] = None) -> ExtractedDocument:
        """
        Normalize plain text into the structured ExtractedDocument used by other parsers.
        This keeps chunking/metadata consistent with richer document types.
        """
        errors: List[str] = []
        try:
            if isinstance(text_content, (bytes, bytearray)):
                text = text_content.decode("utf-8", errors="ignore")
            else:
                text = str(text_content)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"txt_decode_failed: {exc}")
            text = ""

        text = text.replace("\r\n", "\n").strip()
        if not text:
            return ExtractedDocument(
                full_text="",
                sections=[],
                tables=[],
                figures=[],
                chunk_candidates=[],
                doc_type="document",
                errors=errors or ["empty_text"],
            )

        paragraphs = [para.strip() for para in re.split(r"\n\s*\n", text) if para.strip()]
        section_title = "Text Document"
        section_id = self._make_section_id(section_title, 1)

        chunk_candidates: List[ChunkCandidate] = []
        for para in paragraphs:
            chunk_candidates.append(
                ChunkCandidate(
                    text=para,
                    page=None,
                    section_title=section_title,
                    section_id=section_id,
                    chunk_type="text",
                )
            )

        sections = [
            Section(
                section_id=section_id,
                title=section_title,
                level=1,
                start_page=1,
                end_page=1,
                text="\n\n".join(paragraphs),
            )
        ]

        full_text = "\n\n".join(paragraphs)
        doc_type = self._doc_intel.infer_type([], [], sections, full_text, filename_hint=filename or "document.txt")
        return ExtractedDocument(
            full_text=full_text,
            sections=sections,
            tables=[],
            figures=[],
            chunk_candidates=chunk_candidates,
            doc_type=doc_type,
            errors=errors,
        )

    def extract_dataframe(self, df: pd.DataFrame, sheet_name: str = "Sheet1", filename: Optional[str] = None) -> ExtractedDocument:
        """
        Build a structured ExtractedDocument from tabular data (CSV/Excel).
        Preserves headers, row-level detail, and a CSV rendition for downstream chunking.
        """
        try:
            df = df.copy()
            df = df.fillna("")
            df.columns = df.columns.astype(str)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to normalize dataframe for extraction: %s", exc)
            raise

        # Create a CSV-like text block for reference and table capture
        table_csv = df.to_csv(index=False)
        tables = [Table(page=None, text=table_csv, csv=table_csv)]

        sections: List[Section] = []
        chunk_candidates: List[ChunkCandidate] = []
        figures: List[Figure] = []
        full_text_parts: List[str] = []

        # Build a simple section representing this sheet/table
        section_id = self._make_section_id(sheet_name, 1)

        # Generate row-wise text for richer chunking
        row_texts: List[str] = []
        headers = list(df.columns)
        for idx, row in df.iterrows():
            pairs = [f"{col}: {row[col]}" for col in headers if str(row[col]).strip() != ""]
            row_text = "; ".join(pairs).strip()
            if not row_text:
                continue
            row_texts.append(row_text)
            chunk_candidates.append(
                ChunkCandidate(
                    text=row_text,
                    page=None,
                    section_title=sheet_name,
                    section_id=section_id,
                    chunk_type="table_row",
                )
            )

        # Add column-summary chunk to retain header context
        header_text = " | ".join(headers)
        if header_text:
            chunk_candidates.insert(
                0,
                ChunkCandidate(
                    text=f"Columns: {header_text}",
                    page=None,
                    section_title=sheet_name,
                    section_id=section_id,
                    chunk_type="table_header",
                ),
            )
            full_text_parts.append(f"Columns: {header_text}")

        if row_texts:
            full_text_parts.append("\n".join(row_texts))

        sections.append(
            Section(
                section_id=section_id,
                title=sheet_name,
                level=1,
                start_page=1,
                end_page=1,
                text="\n".join(row_texts) if row_texts else header_text,
            )
        )

        full_text_parts.append(table_csv)
        full_text = "\n".join([part for part in full_text_parts if part]).strip()

        doc_type = self._doc_intel.infer_type(tables, figures, sections, full_text, filename_hint=filename or sheet_name)
        return ExtractedDocument(
            full_text=full_text,
            sections=sections,
            tables=tables,
            figures=figures,
            chunk_candidates=chunk_candidates,
            doc_type=doc_type,
            errors=[],
        )
class DocumentIntelligence:
    """Lightweight classifier to tag document type from observed layout/content."""

    @staticmethod
    def infer_type(
        tables: List[Table],
        figures: List[Figure],
        sections: List[Section],
        full_text: str,
        filename_hint: Optional[str] = None,
    ) -> str:
        name = (filename_hint or "").lower()
        if name.endswith((".ppt", ".pptx")):
            return "presentation"
        if name.endswith(".docx"):
            return "document"
        if name.endswith(".pdf"):
            # Heuristics based on structure
            if len(tables) >= 3:
                return "report_table_heavy"
            if len(figures) >= 3:
                return "report_visual"
            if sections and len(sections) <= 5 and any("slide" in s.title.lower() for s in sections):
                return "presentation_pdf"
            return "report"
        if tables and not figures:
            return "table_dataset"
        if figures and not tables:
            return "visual_brief"
        # Fallback to narrative doc
        return "document"
