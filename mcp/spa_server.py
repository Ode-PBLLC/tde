#!/usr/bin/env python3
import os, json
from pathlib import Path
from typing import List, Dict, Any

# --- optional dotenv ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from fastmcp import FastMCP
from openai import OpenAI
import chromadb
from chromadb.config import Settings

# ---- config (read-only server) ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment or .env")

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
INDEX_DIR  = Path("data/spa_index")
COLLECTION = "spa_pdfs"
TOP_K      = 6

mcp = FastMCP("spa-server")
oai = OpenAI(api_key=OPENAI_API_KEY)

def _col():
    if not INDEX_DIR.exists():
        raise RuntimeError("Index not found. Run build_spa_index.py first.")
    client = chromadb.Client(Settings(persist_directory=str(INDEX_DIR), is_persistent=True))
    return client.get_or_create_collection(COLLECTION)

def _embed_query(text: str) -> List[float]:
    emb = oai.embeddings.create(model="text-embedding-3-small", input=[text])
    return emb.data[0].embedding

def _search(query: str, k: int) -> List[Dict[str, Any]]:
    col = _col()
    qvec = _embed_query(query)
    res = col.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    out = []
    for idx, (txt, md, dist) in enumerate(zip(docs, metas, dists)):
        sim = 1.0 - float(dist) if dist is not None else 0.0
        file_path = md.get("file")
        doc_id = Path(file_path).stem if file_path else f"doc_{idx}"
        preview = (txt or "")[:240]
        out.append(
            {
                "doc_id": doc_id,
                "file": file_path,
                "page": md.get("page"),
                "text": txt,
                "preview": preview,
                "similarity": round(sim, 4),
                "chunk_id": f"{doc_id}::chunk_{idx}",
            }
        )
    return out

# --- tool: list (renamed) ---
@mcp.tool()
def AmazonAssessmentListDocs(limit: int = 50) -> List[Dict[str,Any]]:
    """
    Amazon Assessment index overview. Returns a small sample of (file, page).
    """
    mani = (INDEX_DIR / "manifest.json")
    if not mani.exists():
        return []
    data = json.loads(mani.read_text())
    items = data.get("items", [])[:max(1, limit)]
    return [{"file": it["file"], "page": it["page"]} for it in items]

# --- tool: search (renamed) ---
@mcp.tool()
def AmazonAssessmentSearch(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Vector search over the SPA (Amazon Assessment) index. Returns top-k chunks with document metadata.
    """
    return _search(query, k)

# --- tool: ask (renamed + SPA expert prompt) ---
@mcp.tool()
def AmazonAssessmentAsk(query: str, k: int = TOP_K, max_tokens: int = 800) -> Dict[str,Any]:
    """
    Answer a question as an expert from the Science Panel for the Amazon (SPA),
    using retrieved snippets. Returns model answer + compact citations.
    """
    hits = _search(query, k)
    if not hits:
        return {"answer": "No relevant snippets found in the Amazon Assessment index.", "passages": []}

    def cite(snippet):
        file_name = Path(snippet["file"]).name if snippet.get("file") else snippet["doc_id"]
        page = snippet.get("page")
        location = f" p.{page}" if page is not None else ""
        return f"[{file_name}{location}] {snippet['text'][:300]}"

    context = "\n\n".join(cite(snippet) for snippet in hits)

    system_prompt = (
        "You are responding as a subject-matter expert of the Science Panel for the Amazon (SPA). "
        "SPA synthesizes the state of knowledge on the Amazon to inform policy and practice, integrates "
        "scientific evidence with Indigenous and local knowledge, and advances sustainable development "
        "pathways for the biome. Provide precise, evidence-based, and policy-relevant answers grounded in "
        "the provided snippets from SPA’s Amazon Assessment and related SPA publications. "
        "Prioritize: (1) scientific accuracy and clear uncertainty handling; (2) actionable, Amazon-specific "
        "implications for conservation, restoration, and a sustainable/regenerative socio-bioeconomy; "
        "(3) cross-cutting lenses (One Health; climate–biodiversity–people nexus; land-use change and "
        "deforestation risks; freshwater connectivity; governance). "
        "Cite sources inline as (filename p.page). If a point is not supported by the snippets, say so briefly. "
        "Be concise and avoid speculation."
    )

    user_msg = f"Question: {query}\n\nSnippets:\n{context}"

    try:
        resp = oai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_msg}],
            temperature=0.2,
            max_tokens=max_tokens
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        return {"answer": f"LLM error: {e}", "passages": hits}

    return {
        "answer": answer,
        "passages": hits,
        "metadata": {"model": CHAT_MODEL, "top_k": k},
    }

if __name__ == "__main__":
    print("SPA MCP server ready (using existing index).")
    mcp.run()
