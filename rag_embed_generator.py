#!/usr/bin/env python3
import os, json, time, glob, hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

# ---- config ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment or .env")

EMBED_MODEL = "text-embedding-3-small"

PDF_DIR        = Path("data/SPA_pdfs")
INDEX_DIR      = Path("data/spa_index")
COLLECTION     = "spa_pdfs"
MANIFEST_PATH  = INDEX_DIR / "manifest.json"

CHARS_PER_CHUNK = 1200
CHUNK_OVERLAP   = 200
BATCH           = 64

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _read_pdfs(pdf_dir: Path) -> List[Tuple[str, int, str]]:
    pdf_paths = sorted(glob.glob(str(pdf_dir / "**/*.pdf"), recursive=True))
    out = []
    for p in pdf_paths:
        try:
            reader = PdfReader(p)
            for i, page in enumerate(reader.pages):
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                if txt.strip():
                    out.append((os.path.relpath(p), i, txt))
        except Exception as e:
            print(f"[WARN] Failed reading {p}: {e}")
    return out

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
    pages = _read_pdfs(PDF_DIR)
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
    print("Done. Index written to data/spa_index/")

if __name__ == "__main__":
    # Use an env var FORCE_REINDEX=true to force, or tweak as needed.
    force = os.getenv("FORCE_REINDEX", "false").lower() in ("1","true","yes","y")
    main(force=force)
