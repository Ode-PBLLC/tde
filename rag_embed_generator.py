#!/usr/bin/env python3
import os, json, time, glob, hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from mistralai import Mistral
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

# ---- config ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment or .env")

EMBED_MODEL = "text-embedding-3-small"

PDF_DIR        = Path("data/mb_def")
INDEX_DIR      = Path("data/mb-deforest")
COLLECTION     = "mb-deforest"
MANIFEST_PATH  = INDEX_DIR / "manifest.json"
TRANSLATION = True

CHARS_PER_CHUNK = 1200
CHUNK_OVERLAP   = 200
BATCH           = 64

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

import os

# Import the Google Cloud Translation library.
from google.cloud import translate_v3

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")


def mistral_ocr_pdf(pdf_path: str) -> List[Tuple[str, int, str]]:
    """
    Use Mistral OCR API to extract text from PDF.
    Returns list of (file_path, page_number, text) tuples, same format as _read_pdfs.
    """
    
    # Initialize Mistral client
    mistral_client = None
    try:
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    except Exception as e:
        print(f"Warning: Could not initialize Mistral client: {e}")
    if not mistral_client:
        return []
    
    try:
        # Upload the PDF file
        with open(pdf_path, "rb") as f:
            uploaded_pdf = mistral_client.files.upload(
                file={
                    "file_name": os.path.basename(pdf_path),
                    "content": f,
                },
                purpose="ocr"
            )
        
        # Get signed URL for the uploaded file
        signed_url = mistral_client.files.get_signed_url(file_id=uploaded_pdf.id)
        
        # Perform OCR on the document
        ocr_response = mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            },
            include_image_base64=False  # We don't need images, just text
        )
        
        # Convert to same format as _read_pdfs: (file_path, page_number, text)
        out = []
        rel_path = os.path.relpath(pdf_path)
        for i, page in enumerate(ocr_response.pages):
            if page.markdown and page.markdown.strip():
                out.append((rel_path, i, page.markdown))
        
        return out
    
    except Exception as e:
        print(f"Mistral OCR failed for {pdf_path}: {e}")
        return []

def _read_pdfs_with_ocr(pdf_dir: Path) -> List[Tuple[str, int, str]]:
    """
    Read all PDFs in directory using Mistral OCR with pypdf fallback.
    Returns list of (file_path, page_number, text) tuples.
    """
    pdf_paths = sorted(glob.glob(str(pdf_dir / "**/*.pdf"), recursive=True))
    out = []
    for p in pdf_paths:
        # Try Mistral OCR first
        results = mistral_ocr_pdf(p)
        if results:
            print(f"[INFO] Used Mistral OCR for {p}")
            out.extend(results)
        else:
            # Fallback to pypdf using existing _read_pdfs function for single file
            print(f"[INFO] Falling back to pypdf for {p}")
            try:
                reader = PdfReader(p)
                rel_path = os.path.relpath(p)
                for i, page in enumerate(reader.pages):
                    try:
                        txt = page.extract_text() or ""
                    except Exception:
                        txt = ""
                    if txt.strip():
                        out.append((rel_path, i, txt))
            except Exception as e:
                print(f"[WARN] Failed reading {p}: {e}")
    return out

# Translate text per page
def translate_text(
    text: str = "YOUR_TEXT_TO_TRANSLATE",
    source_language_code: str = "pt-BR",
    target_language_code: str = "en-US",
) -> translate_v3.TranslationServiceClient:
    """Translate Text from a Source language to a Target language.
    Args:
        text: The content to translate.
        source_language_code: The code of the source language.
        target_language_code: The code of the target language.
            For example: "fr" for French, "es" for Spanish, etc.
            Find available languages and codes here:
            https://cloud.google.com/translate/docs/languages#neural_machine_translation_model
    """

    # Initialize Translation client
    PROJECT_ID = "tde-translate"
    client = translate_v3.TranslationServiceClient()
    parent = f"projects/{PROJECT_ID}/locations/global"

    # MIME type of the content to translate.
    # Supported MIME types:
    # https://cloud.google.com/translate/docs/supported-formats
    mime_type = "text/plain"

    # Translate text from the source to the target language.
    response = client.translate_text(
        contents=[text],
        parent=parent,
        mime_type=mime_type,
        source_language_code=source_language_code,
        target_language_code=target_language_code,
    )

    for translation in response.translations: # Print a short snippet for each page
        print(f"Translated text: {translation.translated_text[0:50]}")

    return response


def _chunk_text(text: str, source_id: str, file_path: str, page: int):
    text = " ".join(text.split())
    n, start = len(text), 0
    while start < n:
        end = min(n, start + CHARS_PER_CHUNK)
        yield {
            "id": f"{source_id}-{start}-{end}",
            "text": text[start:end],
            "meta": {"source_id": source_id, "file": file_path, "page": page, "start": start, "end": end},
        }
        if end == n:
            break
        start = max(start + CHARS_PER_CHUNK - CHUNK_OVERLAP, end)

def _build_manifest(pages) -> Dict[str, Any]:
    items = []
    for fp, page, txt in pages:
        items.append({"file": fp, "page": page, "hash": _sha1(f"{fp}|{page}|{txt[:2000]}")})
    return {"created_at": time.time(), "items": items}

def _load_manifest():
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text())
        except Exception:
            return None

def _same_manifest(a, b) -> bool:
    if not a or not b:
        return False
    ha = sorted(x["hash"] for x in a.get("items", []))
    hb = sorted(x["hash"] for x in b.get("items", []))
    return ha == hb

def main(force: bool = False):
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {PDF_DIR}")

    print("Scanning PDFs…")
    pages = _read_pdfs_with_ocr(PDF_DIR)
    if not pages:
        print("No readable PDFs found.")
        return

    new_manifest = _build_manifest(pages)
    old_manifest = _load_manifest()
    if not force and old_manifest and _same_manifest(old_manifest, new_manifest):
        print("No changes detected. Index is up-to-date.")
        return

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.Client(Settings(persist_directory=str(INDEX_DIR), is_persistent=True))

    # Recreate collection
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    oai = OpenAI(api_key=OPENAI_API_KEY)

    if TRANSLATION:  
        # Translate each page
        translated_pages = []
        for i, (fp, page, txt) in enumerate(pages):
            try:
                print(f"Translating page {page} of {fp}... ({i+1}/{len(pages)})")
                response = translate_text(
                    text=txt,
                    source_language_code="pt-BR",  # Brazilian Portuguese
                    target_language_code="en"   # English
                )
                # Get the translated text from the response
                translated_txt = response.translations[0].translated_text
                translated_pages.append((fp, page, translated_txt))
            except Exception as e:
                print(f"[WARN] Translation failed for {fp} page {page}: {e}")
                # Fall back to original text if translation fails
                translated_pages.append((fp, page, txt))
        
        # Use translated pages for chunking
        pages = translated_pages

    # Build all chunks
    chunks = []

    for fp, page, txt in pages:
        sid = _sha1(f"{fp}|{page}")
        chunks.extend(list(_chunk_text(txt, sid, fp, page)))

    total = len(chunks)
    print(f"Embedding & indexing {total} chunks from {len(pages)} PDF pages…")

    for i in range(0, total, BATCH):
        batch = chunks[i:i+BATCH]
        texts = [c["text"] for c in batch]
        ids   = [c["id"] for c in batch]
        metas = [c["meta"] for c in batch]

        # Embed
        emb = oai.embeddings.create(model=EMBED_MODEL, input=texts)
        vecs = [d.embedding for d in emb.data]

        # Upsert
        col.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=vecs)

        if (i // BATCH) % 10 == 0:
            print(f"  … {min(i+BATCH, total)}/{total}")

    MANIFEST_PATH.write_text(json.dumps(new_manifest, indent=2))
    print(f"Done. Index written to {INDEX_DIR}")

if __name__ == "__main__":
    # Use an env var FORCE_REINDEX=true to force, or tweak as needed.
    force = os.getenv("FORCE_REINDEX", "false").lower() in ("1","true","yes","y")
    main(force=force)
