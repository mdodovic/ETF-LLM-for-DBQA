import pytesseract
from pdf2image import convert_from_path
from langchain_community.document_loaders import PDFMinerLoader
from utils import clean_text

def extract_text_from_pdf(pdf_path):
    try:
        loader = PDFMinerLoader(pdf_path)
        documents = loader.load()
        text = "\n".join(doc.page_content for doc in documents)
        return clean_text(text) if text.strip() else None
    except:
        return clean_text(extract_text_with_ocr(pdf_path))

def extract_text_with_ocr(pdf_path):
    text = "".join(pytesseract.image_to_string(img) for img in convert_from_path(pdf_path))
    return clean_text(text)
