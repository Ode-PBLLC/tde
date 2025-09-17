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

mcp = FastMCP("spa_server")
oai = OpenAI(api_key=OPENAI_API_KEY)

def _col():
    if not INDEX_DIR.exists():
        raise RuntimeError("Index not found. Run build_spa_index.py first.")
    client = chromadb.Client(Settings(persist_directory=str(INDEX_DIR), is_persistent=True))
    return client.get_or_create_collection(COLLECTION)

def _embed_query(text: str) -> List[float]:
    emb = oai.embeddings.create(model="text-embedding-3-small", input=[text])
    return emb.data[0].embedding

def _search(query: str, k: int) -> List[Dict[str,Any]]:
    col = _col()
    qvec = _embed_query(query)
    res = col.query(query_embeddings=[qvec], n_results=k, include=["documents","metadatas","distances"])
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    out = []
    for txt, md, dist in zip(docs, metas, dists):
        sim = 1.0 - float(dist) if dist is not None else 0.0
        out.append({"text": txt, "file": md.get("file"), "page": md.get("page"), "similarity": round(sim, 4)})
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
def AmazonAssessmentSearch(query: str, k: int = TOP_K) -> List[Dict[str,Any]]:
    """
    Vector search over the SPA (Amazon Assessment) index. Returns top-k chunks with file/page/similarity.
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
        return {"answer": "No relevant snippets found in the Amazon Assessment index.", "citations": []}

    def cite(h):
        from pathlib import Path
        return f"[{Path(h['file']).name} p.{h['page']}] {h['text'][:300]}"

    context = "\n\n".join(cite(h) for h in hits)

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
        return {"answer": f"LLM error: {e}", "citations": hits}

    citations = [
        {"file": h["file"], "page": h["page"], "preview": h["text"][:180], "similarity": h["similarity"]}
        for h in hits
    ]
    return {"answer": answer, "citations": citations}

if __name__ == "__main__":
    print("SPA MCP server ready (using existing index).")
    mcp.run()
