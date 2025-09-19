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
INDEX_DIR  = Path("data/mb-deforest")
COLLECTION = "mb-deforest"
TOP_K      = 6

mcp = FastMCP("mb_deforest")
oai = OpenAI(api_key=OPENAI_API_KEY)

def _col():
    if not INDEX_DIR.exists():
        raise RuntimeError("Index not found. Run rag_embed_generator.py first.")
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
    snippets: List[Dict[str, Any]] = []
    for idx, (txt, md, dist) in enumerate(zip(docs, metas, dists)):
        sim = 1.0 - float(dist) if dist is not None else 0.0
        file_path = md.get("file")
        doc_id = Path(file_path).stem if file_path else f"doc_{idx}"
        snippets.append(
            {
                "doc_id": doc_id,
                "file": file_path,
                "page": md.get("page"),
                "text": txt,
                "preview": (txt or "")[:240],
                "similarity": round(sim, 4),
                "chunk_id": f"{doc_id}::chunk_{idx}",
            }
        )
    return snippets


def _format_context(snippets: List[Dict[str, Any]]) -> str:
    parts = []
    for snippet in snippets:
        file_name = Path(snippet["file"]).name if snippet.get("file") else snippet["doc_id"]
        page = snippet.get("page")
        location = f" p.{page}" if page is not None else ""
        parts.append(f"[{file_name}{location}] {snippet['text'][:300]}")
    return "\n\n".join(parts)

# @mcp.tool()
# def WMOReportListDocs(limit: int = 50) -> List[Dict[str,Any]]:
#     """
#     WMO Report index overview. Returns a small sample of (file, page).
#     """
#     mani = (INDEX_DIR / "manifest.json")
#     if not mani.exists():
#         return []
#     data = json.loads(mani.read_text())
#     items = data.get("items", [])[:max(1, limit)]
#     return [{"file": it["file"], "page": it["page"]} for it in items]

@mcp.tool()
def MBReportSearch(query: str, k: int = TOP_K) -> List[Dict[str,Any]]:
    """
    Vector search over the MapBiomas index. Returns top-k chunks with file/page/similarity.
    """
    return _search(query, k)

@mcp.tool()
def MBReportAsk(query: str, k: int = TOP_K, max_tokens: int = 800) -> Dict[str,Any]:
    """
    Gets context from the MapBiomas Deforestation report focused,
    using retrieved snippets. Returns model answer + citations and snippets.
    """
    hits = _search(query, k)
    if not hits:
        return {"answer": "No relevant snippets found in the WMO Report index.", "passages": []}

    context = _format_context(hits)

    system_prompt = ("""
        You are responding as a subject-matter expert drawing on the MapBiomas Annual Deforestation Report (RAD) (2024).  

        Provide concise, evidence-based, and policy-relevant answers grounded strictly in the retrieved report snippets and related WMO or IPCC publications.  

        Priorities:  
        1. **Scientific accuracy** – clearly state confidence levels, uncertainties, and attribution where reported.  
        2. **Actionable insights** – emphasize region-specific implications for adaptation, mitigation, and resilience, especially regarding governance, disaster risk reduction, and sustainable development.  
        3. **Cross-cutting perspectives** – explicitly link findings to:  
        - Climate–biodiversity–people nexus  
        - Water security and freshwater connectivity  
        - Land-use change and deforestation risks  
        - One Health (climate–health interactions)  
        - Indigenous and local knowledge integration  

        **Citation guidance**: Cite sources inline in the format (WMO-LAC2024 p.X) or (filename p.X) if another WMO source.  

        **Important rules**:  
        - If snippets do not support a claim, say so explicitly and avoid speculation.  
        - Keep responses concise, precise, and tailored to decision-makers.  
        - Do not generalize; ground every statement in the provided texts.  
        """)


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

    return {"answer": answer, "passages": hits, "metadata": {"model": CHAT_MODEL, "top_k": k}}


@mcp.tool()
def IPCCReportAsk(query: str, k: int = TOP_K, max_tokens: int = 800) -> Dict[str,Any]:
    """
    Gets context from the IPCC reports focused on state of climate globally and in Latin America and the Caribbean,
    using retrieved snippets. Returns model answer + citations and snippets.
    """
    hits = _search(query, k)
    if not hits:
        return {"answer": "No relevant snippets found in the WMO Report index.", "passages": []}

    context = _format_context(hits)

    system_prompt = ("""
        You are responding as a subject-matter expert drawing on the WMO *State of the Climate in Latin America and the Caribbean 2024* report and the IPCC reports *Weather and Climate Extreme Events in a Changing Climate* and *Central and South America*.  

        Provide concise, evidence-based, and policy-relevant answers grounded strictly in the retrieved report snippets and related WMO or IPCC publications.  

        Priorities:  
        1. **Scientific accuracy** – clearly state confidence levels, uncertainties, and attribution where reported.  
        2. **Actionable insights** – emphasize region-specific implications for adaptation, mitigation, and resilience, especially regarding governance, disaster risk reduction, and sustainable development.  
        3. **Cross-cutting perspectives** – explicitly link findings to:  
        - Climate–biodiversity–people nexus  
        - Water security and freshwater connectivity  
        - Land-use change and deforestation risks  
        - One Health (climate–health interactions)  
        - Indigenous and local knowledge integration  

        **Citation guidance**: Cite sources inline in the format (IPCC_AR6_WGI_Ch11 p.X), or (filename p.X) if another IPCC source.  

        **Important rules**:  
        - If snippets do not support a claim, say so explicitly and avoid speculation.  
        - Keep responses concise, precise, and tailored to decision-makers.  
        - Do not generalize; ground every statement in the provided texts.  
        """)


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

    return {"answer": answer, "passages": hits, "metadata": {"model": CHAT_MODEL, "top_k": k}}

if __name__ == "__main__":
    print("Deforestation MCP server ready (using existing index).")
    mcp.run()
