import hashlib
import io
import logging
import re
from typing import List, Optional

import docx
import fitz
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from pptx import Presentation
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

    def _ocr_image(self, image: Image.Image) -> str:
        if self.ocr_engine == "easyocr" and easyocr:
            try:
                self._ensure_easyocr()
                result = self._easyocr_reader.readtext(np.array(image), detail=0)
                return " ".join(result).strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("EasyOCR failed, falling back to tesseract: %s", exc)
        try:
            return pytesseract.image_to_string(image).strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("OCR failed: %s", exc)
            return ""

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

    # ---------- Extraction routines ----------
    def extract_text_from_pdf(self, pdf_content: bytes) -> ExtractedDocument:
        sections: List[Section] = []
        tables: List[Table] = []
        figures: List[Figure] = []
        chunk_candidates: List[ChunkCandidate] = []
        full_text_parts: List[str] = []

        current_title = "Introduction"
        current_section_id = self._make_section_id(current_title, 1)
        current_start_page = 1
        section_buffer: List[str] = []
        last_page = 1

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
                            ocr_text = self._ocr_image(image)
                            if ocr_text:
                                figures.append(Figure(page=page_index, caption=ocr_text))
                                chunk_candidates.append(
                                    ChunkCandidate(
                                        text=ocr_text,
                                        page=page_index,
                                        section_title=current_title,
                                        section_id=current_section_id,
                                        chunk_type="ocr",
                                    )
                                )
                                page_text_parts.append(ocr_text)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("OCR image extraction failed on page %s: %s", page_index, exc)

                if page_text_parts:
                    full_text_parts.append(f"\n--- Page {page_index} ---\n" + "\n".join(page_text_parts))

        self._finalize_section(
            sections, current_title, current_section_id, current_start_page, last_page, section_buffer
        )

        full_text = "\n".join(full_text_parts).strip()
        return ExtractedDocument(
            full_text=full_text,
            sections=sections,
            tables=tables,
            figures=figures,
            chunk_candidates=chunk_candidates,
        )

    def extract_text_from_docx(self, doc_content: bytes) -> ExtractedDocument:
        document = docx.Document(doc_content)
        sections: List[Section] = []
        tables: List[Table] = []
        figures: List[Figure] = []
        chunk_candidates: List[ChunkCandidate] = []
        full_text_parts: List[str] = []

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
        for rel in document.part.rels.values():
            if "image" in rel.target_ref:
                caption = getattr(rel, "alt_text", None) or rel.target_ref
                figures.append(Figure(page=None, caption=caption))

        full_text = "\n".join(full_text_parts).strip()
        return ExtractedDocument(
            full_text=full_text,
            sections=sections,
            tables=tables,
            figures=figures,
            chunk_candidates=chunk_candidates,
        )

    def extract_text_from_pptx(self, ppt_content: bytes) -> ExtractedDocument:
        presentation = Presentation(ppt_content)
        sections: List[Section] = []
        tables: List[Table] = []
        figures: List[Figure] = []
        chunk_candidates: List[ChunkCandidate] = []
        full_text_parts: List[str] = []

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
        return ExtractedDocument(
            full_text=full_text,
            sections=sections,
            tables=tables,
            figures=figures,
            chunk_candidates=chunk_candidates,
        )

    def extract_dataframe(self, df, MODEL):
        try:
            df = df.select_dtypes(include=[np.number, 'object']).copy()
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    df[f"Days Since {col}"] = (df[col] - df[col].min()).dt.days
                    df.drop(columns=[col], inplace=True)
                except Exception:
                    logger.warning(f"Column {col} could not be converted to datetime. Keeping it as categorical.")
                    df[col] = df[col].astype(str)

            df.columns = df.columns.astype(str)
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                text_data = df[categorical_cols].astype(str).apply(lambda x: ' '.join(x.dropna()), axis=1).tolist()
                embeddings = MODEL.encode(text_data, convert_to_numpy=True)
                return {"embeddings": embeddings, "texts": text_data}

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error processing dataframe: {e}")
            return None
