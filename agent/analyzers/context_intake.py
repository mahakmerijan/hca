"""
Context Intake Module
Parses user-provided context: business idea, job description, date partner info.
Accepts plain text or uploaded PDF/DOCX files.
"""

import io
import os

# ---------- optional imports — graceful fallback ----------
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import docx  # python-docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


CONTEXT_TYPES = {
    "job":      "job_description",
    "business": "business_idea",
    "date":     "date_partner",
}


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes."""
    if not PYMUPDF_AVAILABLE:
        return "[PDF parsing unavailable — install pymupdf]"
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = [page.get_text() for page in doc]
        return "\n".join(pages).strip()
    except Exception as e:
        return f"[PDF read error: {e}]"


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract all text from a DOCX given its raw bytes."""
    if not DOCX_AVAILABLE:
        return "[DOCX parsing unavailable — install python-docx]"
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()
    except Exception as e:
        return f"[DOCX read error: {e}]"


def extract_text_from_file(filename: str, file_bytes: bytes) -> str:
    """Auto-detect file type and extract text."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(file_bytes)
    elif ext in (".txt", ".md"):
        try:
            return file_bytes.decode("utf-8", errors="replace").strip()
        except Exception as e:
            return f"[Text read error: {e}]"
    else:
        return f"[Unsupported file type: {ext}]"


class UserContext:
    """Holds the user-supplied scenario context (text + parsed docs)."""

    def __init__(self):
        self.job_description: str = ""
        self.business_idea: str = ""
        self.date_partner: str = ""

    def set_from_text(self, context_type: str, text: str):
        key = CONTEXT_TYPES.get(context_type, context_type)
        setattr(self, key, text.strip())

    def set_from_file(self, context_type: str, filename: str, file_bytes: bytes):
        text = extract_text_from_file(filename, file_bytes)
        self.set_from_text(context_type, text)

    def has_any(self) -> bool:
        return bool(self.job_description or self.business_idea or self.date_partner)

    def to_dict(self) -> dict:
        return {
            "job_description": self.job_description,
            "business_idea": self.business_idea,
            "date_partner": self.date_partner,
        }

    def to_prompt_block(self) -> str:
        """Format context for injection into the Gemini prompt."""
        blocks = []
        if self.job_description:
            blocks.append(f"JOB DESCRIPTION PROVIDED BY USER:\n{self.job_description}")
        if self.business_idea:
            blocks.append(f"BUSINESS IDEA / DEAL CONTEXT PROVIDED BY USER:\n{self.business_idea}")
        if self.date_partner:
            blocks.append(f"DATE PARTNER DESCRIPTION PROVIDED BY USER:\n{self.date_partner}")
        return "\n\n".join(blocks)
