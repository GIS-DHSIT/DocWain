import pytest
from PIL import Image


def test_render_pages_returns_pil_images():
    from src.visual_intelligence.page_renderer import PageRenderer
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Hello World")
    pdf_bytes = doc.tobytes()
    doc.close()

    renderer = PageRenderer(dpi=150)
    pages = renderer.render(pdf_bytes)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert isinstance(pages[0].image, Image.Image)
    assert pages[0].width > 0
    assert pages[0].height > 0


def test_render_specific_pages():
    from src.visual_intelligence.page_renderer import PageRenderer
    import fitz
    doc = fitz.open()
    for i in range(5):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), f"Page {i+1}")
    pdf_bytes = doc.tobytes()
    doc.close()

    renderer = PageRenderer(dpi=150)
    pages = renderer.render(pdf_bytes, page_numbers=[1, 3, 5])
    assert len(pages) == 3
    assert [p.page_number for p in pages] == [1, 3, 5]


def test_render_non_pdf_returns_empty():
    from src.visual_intelligence.page_renderer import PageRenderer
    renderer = PageRenderer()
    pages = renderer.render(b"not a pdf")
    assert pages == []
