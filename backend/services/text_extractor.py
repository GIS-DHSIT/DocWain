import fitz  # PyMuPDF
import io
import docx
import csv


def extract_text_from_document(filename, content):
    text = ""

    if filename.endswith(".pdf"):
        doc = fitz.open(stream=io.BytesIO(content), filetype="pdf")
        for page in doc:
            text += page.get_text("text")

    elif filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(content))
        for para in doc.paragraphs:
            text += para.text + "\n"

    elif filename.endswith(".txt"):
        text = content.decode("utf-8")

    elif filename.endswith(".csv"):
        reader = csv.reader(io.StringIO(content.decode("utf-8")))
        text = "\n".join([" ".join(row) for row in reader])

    return text
