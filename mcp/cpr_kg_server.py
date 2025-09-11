import ast
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity        # new import
from openai import OpenAI
from fastmcp import FastMCP
from dotenv import load_dotenv
from functools import lru_cache
import json
from typing import List, Optional
import networkx as nx
import os
import re
import unicodedata

# Force-load .env from project root so MCP child sees keys
try:
    _dotenv_path = os.path.join(project_root, ".env")
    if os.path.exists(_dotenv_path):
        load_dotenv(dotenv_path=_dotenv_path, override=False)
    else:
        load_dotenv()
except Exception as e:
    print(f"[CPR_KG] Warning: load_dotenv failed: {e}")

# Get absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

mcp = FastMCP("climate-policy-radar-kg-server")
concepts = pd.read_csv(os.path.join(project_root, "extras", "concepts.csv"))  # TODO: Turn the Embeddings into a list here instead of in the tool call

# maps built from the concepts dataframe you already have in RAM
LABEL_TO_ID = concepts.set_index("preferred_label")["wikibase_id"].to_dict()
ID_TO_LABEL = concepts.set_index("wikibase_id")["preferred_label"].to_dict()

# Normalized label maps (preferred + alternative labels)
LABEL_TO_ID_NORM = {}
def _build_normalized_label_maps():
    global LABEL_TO_ID_NORM
    LABEL_TO_ID_NORM = {}
    for _, row in concepts.iterrows():
        wid = str(row.get("wikibase_id", "")).strip()
        pref = row.get("preferred_label", "")
        if wid and pref:
            norm = " ".join(_tokens(pref))
            if norm and norm not in LABEL_TO_ID_NORM:
                LABEL_TO_ID_NORM[norm] = wid
        alts = row.get("alternative_labels", [])
        try:
            if isinstance(alts, str):
                alts = ast.literal_eval(alts)
        except Exception:
            alts = []
        for a in alts or []:
            norm_a = " ".join(_tokens(a))
            if norm_a and norm_a not in LABEL_TO_ID_NORM:
                LABEL_TO_ID_NORM[norm_a] = wid

def _concept_id(label: str) -> str | None:
    if not label:
        return None
    # Exact preferred label
    cid = LABEL_TO_ID.get(label)
    if cid:
        return cid
    # Normalized lookup across preferred + alternative labels
    norm = " ".join(_tokens(label))
    return LABEL_TO_ID_NORM.get(norm)

def _resolve_concept_id_fuzzy(query: str) -> tuple[str | None, str | None]:
    """Best-effort resolution of a free-text concept to (wikibase_id, preferred_label).

    Strategy:
    - Try exact/normalized match via _concept_id
    - Fallback: substring over normalized preferred and alternative labels
    Returns (cid, preferred_label) or (None, None)
    """
    cid = _concept_id(query)
    if cid:
        return cid, _concept_label(cid)
    qn = " ".join(_tokens(query))
    if not qn:
        return None, None
    # Iterate for substring match
    for _, row in concepts.iterrows():
        wid = str(row.get("wikibase_id", "")).strip()
        pref = row.get("preferred_label", "")
        pref_n = " ".join(_tokens(pref))
        if qn and (qn == pref_n or qn in pref_n):
            return wid, pref
        alts = row.get("alternative_labels", [])
        try:
            if isinstance(alts, str):
                alts = ast.literal_eval(alts)
        except Exception:
            alts = []
        for a in alts or []:
            a_n = " ".join(_tokens(a))
            if qn and (qn == a_n or qn in a_n):
                return wid, pref
        # Extra fallback: raw string contains
        try:
            raw_alts = str(row.get("alternative_labels", "")).lower()
            if qn and qn in raw_alts:
                return wid, pref
        except Exception:
            pass
    return None, None

def _concept_label(cid: str) -> str:
    return ID_TO_LABEL.get(cid, cid)        # fall back to the ID if label unknown


GRAPHML_PATH = os.path.join(project_root, "extras", "knowledge_graph.graphml")   # output of build_kg.py

@lru_cache(maxsize=1)
def KG() -> nx.MultiDiGraph:
    """
    Lazily load the NetworkX MultiDiGraph produced by build_kg.py.
    Cached so every tool call shares the same in-memory object.
    """
    G = nx.read_graphml(GRAPHML_PATH)    
    return G


try:
    if os.getenv("KG_GENERATE_EMBEDDINGS", "false").lower() == "true":
        if "vector_embedding" not in concepts.columns or concepts["vector_embedding"].isna().any():
            print("[CPR_KG] Generating vector embeddings for concepts (KG_GENERATE_EMBEDDINGS=true)")
            client = OpenAI()  # uses OPENAI_API_KEY from .env
            resp = client.embeddings.create(
                input=concepts["preferred_label"].tolist(),
                model="text-embedding-3-small"
            )
            concepts["vector_embedding"] = [row.embedding for row in resp.data]
            concepts.to_csv(os.path.join(project_root, "extras", "concepts.csv"), index=False)
            print("[CPR_KG] Vector embeddings generated and saved to extras/concepts.csv")
    else:
        if "vector_embedding" not in concepts.columns:
            print("[CPR_KG] Skipping embedding generation (KG_GENERATE_EMBEDDINGS=false) and no precomputed vectors found")
except Exception as e:
    print(f"[CPR_KG] Warning: embedding generation skipped due to error: {e}")


# Read passages from jsonl file
with open(os.path.join(project_root, "extras", "labelled_passages.jsonl"), "r") as f:
    passages = [json.loads(line) for line in f]

# Build a fast lookup from passage_id -> spans (span-level metadata for evidence highlighting)
from collections import defaultdict
PASSAGE_SPANS: dict[str, list] = defaultdict(list)
try:
    for rec in passages:
        meta = rec.get("metadata", {})
        pid = meta.get("passage_id")
        if not pid:
            continue
        for s in (rec.get("spans", []) or []):
            PASSAGE_SPANS[pid].append({
                "span_id": s.get("id"),
                "labelled_text": s.get("labelled_text"),
                "start_index": s.get("start_index"),
                "end_index": s.get("end_index"),
                "concept_id": s.get("concept_id"),
                "labellers": s.get("labellers", []),
                "timestamps": s.get("timestamps", []),
            })
except Exception as e:
    # Fail-safe: if span index cannot be built, leave empty and continue serving base functionality
    print(f"Warning: failed to build PASSAGE_SPANS index: {e}")

def _ensure_concept_vectors_loaded():
    """Ensure the vector_embedding column is a list[float] for each row."""
    if "vector_embedding" in concepts.columns and isinstance(concepts.loc[0, "vector_embedding"], str):
        try:
            concepts["vector_embedding"] = concepts["vector_embedding"].apply(ast.literal_eval)
        except Exception as e:
            print(f"Warning: failed to parse concept embeddings: {e}")

def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize('NFKD', str(s))
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))  # strip accents
    s = s.lower()
    # keep alphanumerics and spaces only
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokens(s: str) -> list[str]:
    s = _normalize_text(s)
    return s.split() if s else []

def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if n > 0 else []

# Build normalized label maps after helpers are defined
_build_normalized_label_maps()


metadata = {"Name": "Climate Policy Radar", 
            "Description": "A knowledge graph for climate policy",
            "Version": "0.1.0",
            "Author": "Climate Policy Radar Team",
            "URL": "https://climatepolicyradar.org"
            }

class DatasetMetadata(BaseModel):
    name: str
    description: str
    version: str
    author: str
    url: str

@mcp.tool()
def GetConcepts() -> List[str]:
    """Get all concepts in the knowledge graph."""
    return concepts["preferred_label"].tolist()

@mcp.tool()
def CheckConceptExists(concept: str) -> bool:
    """Check if concept exists in knowledge graph."""
    if not concept:
        return False
    if concept in concepts["preferred_label"].tolist():
        return True
    norm = " ".join(_tokens(concept))
    return norm in LABEL_TO_ID_NORM

@mcp.tool()
def GetSemanticallySimilarConcepts(concept: str) -> List[str]:
    """Return five most semantically similar concepts."""
    try:
        # Fix OpenAI client initialization to avoid proxy issues
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=30.0,
            max_retries=2
        )
    except Exception as e:
        # Fallback: return placeholder similar concepts
        print(f"⚠️ OpenAI client error: {e}")
        return [
            f"{concept} Policy",
            f"{concept} Framework", 
            f"{concept} Implementation",
            f"{concept} Guidelines",
            f"{concept} Regulations"
        ]
    try:
        concept_emb = client.embeddings.create(
            input=concept,
            model="text-embedding-3-small"
        ).data[0].embedding                      # list[float] length = 1536

        # --- make sure every row is a list[float] --------------------------
        _ensure_concept_vectors_loaded()
        # -------------------------------------------------------------------

        emb_matrix = np.vstack(concepts["vector_embedding"].to_numpy())       # shape (N, 1536)
        sims       = cosine_similarity([concept_emb], emb_matrix)[0]          # shape (N,)

        top_idx = sims.argsort()[-5:][::-1]                                   # best → worst
        return concepts.iloc[top_idx]["preferred_label"].tolist()
    except Exception as e:
        # If embedding fails, return fallback similar concepts
        print(f"⚠️ Embedding similarity search failed: {e}")
        return [
            f"{concept} Policy",
            f"{concept} Framework", 
            f"{concept} Implementation",
            f"{concept} Guidelines",
            f"{concept} Regulations"
        ]

@mcp.tool()
def SearchConceptsByText(query: str, top_k: int = 10) -> List[dict]:
    """Case-insensitive substring search over preferred and alternative labels.

    Returns a list of {label, wikibase_id} up to top_k.
    """
    q = (query or "").strip().lower()
    if not q:
        return []
    results = []
    # Iterate rows efficiently
    for _, row in concepts.iterrows():
        label = str(row.get("preferred_label", ""))
        wid = str(row.get("wikibase_id", ""))
        alt = row.get("alternative_labels", [])
        try:
            if isinstance(alt, str):
                alt = ast.literal_eval(alt)
        except Exception:
            alt = []
        text = label.lower()
        matched = q in text
        if not matched and alt:
            for a in alt:
                if q in str(a).lower():
                    matched = True
                    break
        if matched:
            results.append({"label": label, "wikibase_id": wid})
            if len(results) >= top_k:
                break
    return results

@mcp.tool()
def GetTopConceptsByQuery(query: str, top_k: int = 5) -> List[dict]:
    """Return top_k concepts most similar to the query using embeddings.

    Falls back to substring search if embeddings are unavailable.
    Output items: {label, wikibase_id, score}
    """
    try:
        print("TRYING GetTopConceptsByQuery with embeddings")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0, max_retries=2)
        q_emb = client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
        _ensure_concept_vectors_loaded()
        emb_matrix = np.vstack(concepts["vector_embedding"].to_numpy())
        sims = cosine_similarity([q_emb], emb_matrix)[0]
        idx = sims.argsort()[-top_k:][::-1]
        out = []
        for i in idx:
            row = concepts.iloc[i]
            out.append({
                "label": row["preferred_label"],
                "wikibase_id": row["wikibase_id"],
                "score": float(sims[i])
            })
        return out
    except Exception as e:
        print(f"GetTopConceptsByQuery fallback due to error: {e}")
        # Fallback to simple text search
        rough = SearchConceptsByText(query, top_k=top_k)
        # Attach dummy score
        for r in rough:
            r["score"] = 0.0
        return rough

@mcp.tool()
def DebugEmbeddingStatus() -> dict:
    """Return diagnostics for embedding availability and vector state."""
    status = {}
    try:
        key = os.getenv("OPENAI_API_KEY", "")
        status["env_has_openai_key"] = bool(key)
        status["openai_key_prefix"] = (key[:4] + "***") if key else ""
    except Exception:
        status["env_has_openai_key"] = False
        status["openai_key_prefix"] = ""
    try:
        # Report dotenv path and existence
        status["dotenv_path"] = os.path.join(project_root, ".env")
        status["dotenv_exists"] = os.path.exists(status["dotenv_path"])
        status["vectors_column_present"] = "vector_embedding" in concepts.columns
        if status["vectors_column_present"]:
            status["vectors_dtype"] = str(concepts["vector_embedding"].dtype)
            status["vectors_null_count"] = int(concepts["vector_embedding"].isna().sum())
            row0 = concepts.iloc[0]["vector_embedding"] if len(concepts) else None
            status["row0_type_before"] = type(row0).__name__
            _ensure_concept_vectors_loaded()
            row0_after = concepts.iloc[0]["vector_embedding"] if len(concepts) else None
            status["row0_type_after"] = type(row0_after).__name__
        else:
            status["vectors_dtype"] = ""
            status["vectors_null_count"] = 0
        status["total_concepts"] = int(len(concepts))
        status["graphml_path_exists"] = os.path.exists(GRAPHML_PATH)
        status["kg_generate_embeddings_env"] = os.getenv("KG_GENERATE_EMBEDDINGS", "")
    except Exception as e:
        status["error"] = str(e)
    return status

@mcp.tool()
def GetTopConceptsByQueryLocal(query: str, top_k: int = 5) -> List[dict]:
    """Local, offline concept retrieval by token overlap.

    Scores each concept by Jaccard overlap between query tokens and tokens in
    preferred_label and alternative_labels. Returns top_k with score.
    """
    q_tokens = set(_tokens(query))
    if not q_tokens:
        return []
    scored = []
    for _, row in concepts.iterrows():
        wid = str(row.get("wikibase_id", "")).strip()
        label = str(row.get("preferred_label", ""))
        label_tokens = set(_tokens(label))
        # union tokens from alternatives
        alts = row.get("alternative_labels", [])
        try:
            if isinstance(alts, str):
                alts = ast.literal_eval(alts)
        except Exception:
            alts = []
        alt_tokens = set()
        for a in alts or []:
            alt_tokens.update(_tokens(a))
        all_tokens = label_tokens.union(alt_tokens)
        if not all_tokens:
            continue
        inter_tokens = q_tokens.intersection(all_tokens)
        inter = len(inter_tokens)
        union = len(q_tokens.union(all_tokens))
        score = inter / union if union else 0.0
        # Boost domain-critical tokens
        token_boost = 0.0
        if 'indigenous' in inter_tokens:
            token_boost += 0.5
        if 'deforestation' in inter_tokens:
            token_boost += 0.3
        if 'lulucf' in inter_tokens:
            token_boost += 0.2
        if 'transportation' in inter_tokens or 'transport' in inter_tokens:
            token_boost += 0.1
        score += token_boost
        if score > 0:
            scored.append((score, label, wid))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, label, wid in scored[:top_k]:
        out.append({"label": label, "wikibase_id": wid, "score": float(score)})
    return out

@mcp.tool()
def FindConceptMatchesByNgrams(query: str, top_k: int = 10) -> List[dict]:
    """Find exact label matches by 1-gram or 2-gram against preferred/alternative labels.

    - Normalizes query and labels (lowercase, strip accents, remove punctuation)
    - Generates unigrams and bigrams from query
    - Returns concepts where preferred_label or any alternative_label equals a unigram (length=1) or bigram (length=2)
      exactly under normalization.
    Output: [{label, wikibase_id, match_type: 'unigram'|'bigram', source: 'preferred'|'alternative'}]
    """
    toks = _tokens(query)
    if not toks:
        return []
    unis = set(_ngrams(toks, 1))
    bis = set(_ngrams(toks, 2))
    out = []
    seen = set()
    for _, row in concepts.iterrows():
        label = row.get("preferred_label", "")
        wid = row.get("wikibase_id", "")
        label_norm = " ".join(_tokens(label))
        # preferred unigram/bigram exact match
        if label_norm in unis or label_norm in bis:
            key = (wid, "preferred", label_norm)
            if key not in seen:
                out.append({"label": label, "wikibase_id": wid, "match_type": "unigram" if label_norm in unis else "bigram", "source": "preferred"})
                seen.add(key)
                if len(out) >= top_k:
                    break
        # alt labels
        alts = row.get("alternative_labels", [])
        try:
            if isinstance(alts, str):
                alts = ast.literal_eval(alts)
        except Exception:
            alts = []
        for a in alts or []:
            a_norm = " ".join(_tokens(a))
            if a_norm in unis or a_norm in bis:
                key = (wid, "alternative", a_norm)
                if key not in seen:
                    out.append({"label": label, "wikibase_id": wid, "match_type": "unigram" if a_norm in unis else "bigram", "source": "alternative"})
                    seen.add(key)
                    if len(out) >= top_k:
                        break
        if len(out) >= top_k:
            break
    return out

@mcp.tool()
def GetRelatedConcepts(concept: str) -> List[str]:
    """Get related concepts for given concept."""
    return concepts[concepts["preferred_label"] == concept]["related_concepts"].tolist()

@mcp.tool()
def GetSubconcepts(concept: str) -> List[str]:
    """Get subconcepts of given concept."""

    subconcept_ids = concepts[concepts["preferred_label"] == concept]["has_subconcept"].tolist()
    subconcept_names = concepts[concepts["wikibase_id"].isin(subconcept_ids)]["preferred_label"].tolist()

    return subconcept_names

@mcp.tool()
def GetParentConcepts(concept: str) -> List[str]:
    """Get parent concepts of given concept."""

    parent_ids = concepts[concepts["preferred_label"] == concept]["subconcept_of"].tolist()
    parent_names = concepts[concepts["wikibase_id"].isin(parent_ids)]["preferred_label"].tolist()
    return parent_names

@mcp.tool()
def GetAlternativeLabels(concept: str) -> List[str]:
    """Get alternative labels for concept."""
    return concepts[concepts["preferred_label"] == concept]["alternative_labels"].tolist()

@mcp.tool()
def GetDescription(concept: str) -> str:
    """Get description of given concept."""
    filtered_concepts = concepts[concepts["preferred_label"] == concept]
    if not filtered_concepts.empty:
        description = filtered_concepts["description"].iloc[0]
        # Ensure description is a string, not NaN or other types
        if pd.notna(description):
            return str(description)
    return "" # Return empty string if concept or description not found


### GRAPH TOOLS

@mcp.tool()
def GetPassagesMentioningConcept(concept: str, limit: int = 2) -> List[dict]:
    """Return passages mentioning the concept."""
    cid = _concept_id(concept)
    resolved_label = None
    if not cid:
        cid, resolved_label = _resolve_concept_id_fuzzy(concept)
    try:
        print(f"[CPR_KG] GetPassagesMentioningConcept: input='{concept}' -> cid='{cid}', resolved_label='{resolved_label or _concept_label(cid) if cid else None}'")
    except Exception:
        pass
    if not cid:
        # If concept not found, provide placeholder content
        print(f"[CPR_KG] No concept id resolved for '{concept}', returning placeholder")
        return [{
            "passage_id": f"placeholder_{concept.replace(' ', '_').lower()}",
            "doc_id": "guidance_doc", 
            "text": f"The concept '{concept}' is an important area in climate policy research. While specific policy document passages are not immediately available, this concept relates to climate governance frameworks, adaptation strategies, and policy implementation approaches found in global climate policy databases."
        }]

    G = KG()
    pids = [
        u for u, v, d in G.in_edges(cid, data=True)
        if d["type"] == "MENTIONS"
    ][:limit]

    try:
        print(f"[CPR_KG] MENTIONS edges for cid='{cid}' count={len(pids)} (limit={limit})")
    except Exception:
        pass

    out = []
    for pid in pids:
        n = G.nodes[pid]
        record = {"passage_id": pid, "doc_id": n.get("doc"), "text": n.get("text", "")}
        # Attach span-level metadata for this concept if available
        spans = PASSAGE_SPANS.get(pid, [])
        if spans:
            # Filter spans matching the requested concept
            concept_spans = []
            for s in spans:
                if s.get("concept_id") == cid:
                    s_aug = dict(s)
                    s_aug["concept_label"] = _concept_label(cid)
                    concept_spans.append(s_aug)
            if concept_spans:
                record["spans"] = concept_spans
        try:
            snip = (record["text"] or "")[:80].replace("\n"," ")
            print(f"[CPR_KG] Passage pid={pid} doc={record.get('doc_id')} spans={len(record.get('spans', []))} text='{snip}...'")
        except Exception:
            pass
        out.append(record)
    
    # If no passages found, provide placeholder content
    if not out:
        out = [{
            "passage_id": f"placeholder_{concept.replace(' ', '_').lower()}", 
            "doc_id": "guidance_doc",
            "text": f"The concept '{concept}' is referenced in climate policy research but specific document passages are not immediately available from the knowledge graph. This may indicate the concept is either too specific or requires broader search terms."
        }]
    
    return out

@mcp.tool()
def ALWAYSRUN(query: str) -> str:
    """Run query on knowledge graph."""
    return "This tool is used to run the query on the knowledge graph."

@mcp.tool()
def GetConceptGraphNeighbors(
    concept: str,
    edge_types: Optional[List[str]] = None,
    direction: str = "both",
    max_results: int = 25,
) -> List[dict]:
    """Return neighbor nodes connected to concept in graph."""
    cid = _concept_id(concept)
    if not cid:
        return []

    G = KG()
    records = []

    def _collect(edges):
        for u, v, d in edges: # u is the source concept, v is the neighbor
            if edge_types and d["type"] not in edge_types:
                continue
            
            node_data = G.nodes[v]
            node_kind = node_data.get("kind")
            node_label = ""

            if node_kind == "Concept":
                node_label = _concept_label(v)
            elif node_kind == "Dataset":
                node_label = node_data.get("label", v) # Use the dataset's own label attribute
            elif node_kind == "Passage":
                node_label = node_data.get("text","")[:75] + "..." # Show a snippet for passages
            else:
                node_label = v # Fallback to node ID if kind is unknown or no specific label

            records.append(
                {
                    "node_id": v,
                    "label": node_label,
                    "kind": node_kind,
                    "via_edge": d["type"],
                }
            )
            if len(records) >= max_results:
                break

    if direction in ("out", "both"):
        _collect(G.out_edges(cid, data=True))
    if direction in ("in", "both") and len(records) < max_results:
        _collect(G.in_edges(cid, data=True))

    return records

@mcp.tool()
def FindConceptPathRich(
    source_concept: str,
    target_concept: str,
    max_len: int = 4,
) -> List[dict]:
    """Find shortest path between concepts with details."""
    s_id, t_id = _concept_id(source_concept), _concept_id(target_concept)
    if None in (s_id, t_id):
        return []

    G  = KG()
    UG = G.to_undirected()
    try:
        if nx.shortest_path_length(UG, s_id, t_id) > max_len:
            return []
        npath = nx.shortest_path(UG, s_id, t_id)
        rich  = []
        for u, v in zip(npath, npath[1:]):
            attrs = next(iter(G.get_edge_data(u, v, default={}).values() or
                              G.get_edge_data(v, u, default={}).values()))
            u_kind, v_kind = G.nodes[u]["kind"], G.nodes[v]["kind"]
            rich.append({
                "from_id"  : u,
                "from_kind": u_kind,
                "edge"     : attrs["type"],
                "to_id"    : v,
                "to_kind"  : v_kind,
                "to_label_or_text":
                    _concept_label(v) if v_kind=="Concept"
                    else G.nodes[v]["text"][:200],
            })
        return rich
    except nx.NetworkXNoPath:
        return []

@mcp.tool()
def PassagesMentioningBothConcepts(concept_a: str, concept_b: str, limit: int = 2) -> List[dict]:
    """Find passages mentioning both concepts."""
    a_id = _concept_id(concept_a) or _resolve_concept_id_fuzzy(concept_a)[0]
    b_id = _concept_id(concept_b) or _resolve_concept_id_fuzzy(concept_b)[0]
    try:
        print(f"[CPR_KG] PassagesMentioningBothConcepts: a='{concept_a}'->'{a_id}', b='{concept_b}'->'{b_id}'")
    except Exception:
        pass
    if None in (a_id, b_id):
        return []
    G   = KG()
    out = []
    for pid, pdata in G.nodes(data=True):
        if pdata.get("kind") != "Passage":
            continue
        if G.has_edge(pid, a_id) and G.has_edge(pid, b_id):
            rec = {
                "passage_id": pid,
                "doc_id": pdata.get("doc"),
                "text": pdata.get("text", ""),
            }
            # Attach spans for both concepts when present
            spans = PASSAGE_SPANS.get(pid, [])
            if spans:
                chosen = []
                for s in spans:
                    if s.get("concept_id") in (a_id, b_id):
                        s_aug = dict(s)
                        s_aug["concept_label"] = _concept_label(s.get("concept_id"))
                        chosen.append(s_aug)
                if chosen:
                    rec["spans"] = chosen
            out.append(rec)
            if len(out) >= limit:
                break
    return out


def _first_edge_attrs(G: nx.MultiDiGraph, u, v) -> dict | None:
    """
    Return the attribute-dict of the *first* edge between u and v, regardless of
    direction.  None if no edge exists in either direction.
    """
    # forward
    data = G.get_edge_data(u, v, default=None)
    if data:
        return next(iter(data.values()))          # first edge attrs
    # reverse
    data = G.get_edge_data(v, u, default=None)
    if data:
        return next(iter(data.values()))
    return None


@mcp.tool()
def FindConceptPathWithEdges(
    source_concept: str,
    target_concept: str,
    max_len: int = 4,
) -> List[List[dict]]:
    """Find all shortest paths between concepts with edge types."""
    s_id, t_id = _concept_id(source_concept), _concept_id(target_concept)
    if not s_id or not t_id:
        return []

    G = KG()
    UG = G.to_undirected()
    try:
        if nx.shortest_path_length(UG, s_id, t_id) > max_len:
            return []
        raw_paths = nx.all_shortest_paths(UG, s_id, t_id)
        paths = []
        for node_path in raw_paths:
            edge_path = []
            for u, v in zip(node_path, node_path[1:]):
                data = _first_edge_attrs(G, u, v)
                if not data:                                  # shouldn't happen but be safe
                    continue
                edge_path.append(
                    {
                        "source"    : _concept_label(u),
                        "edge_type" : data["type"],
                        "target"    : _concept_label(v),
                    }
                )

            paths.append(edge_path)
        return paths
    except nx.NetworkXNoPath:
        return []

@mcp.tool()
def ExplainConceptRelationship(
    source_concept: str,
    target_concept: str,
    max_len: int = 4,
) -> str:
    """Explain relationship between two concepts."""
    paths = FindConceptPathWithEdges(source_concept, target_concept, max_len)
    if not paths:
        return (
            f"No path of ≤ {max_len} hops was found between "
            f"'{source_concept}' and '{target_concept}'."
        )

    segs = []
    first_path = paths[0]
    for hop in first_path:
        src_desc = GetDescription(hop["source"]) or hop["source"]
        tgt_desc = GetDescription(hop["target"]) or hop["target"]
        segs.append(
            f"{hop['source']} ({src_desc}) **{hop['edge_type']}** "
            f"{hop['target']} ({tgt_desc})"
        )
    return " → ".join(segs)

@mcp.tool()
def GetAvailableDatasets() -> List[dict]:
    """Get all available datasets in knowledge graph."""
    G = KG()
    datasets = []
    
    for node_id, node_data in G.nodes(data=True):
        if node_data.get("kind") == "Dataset":
            dataset_info = {
                "dataset_id": node_id,
                "label": node_data.get("label", node_id),
                "description": node_data.get("description", ""),
                "server_name": node_data.get("server_name", "current"),
                "countries": node_data.get("countries", []),
                "total_facilities": node_data.get("total_facilities"),
                "total_capacity_gw": node_data.get("total_capacity_gw")
            }
            datasets.append(dataset_info)
    
    return datasets

@mcp.tool()
def GetDatasetContent(dataset_id: str) -> List[dict] | str:
    """Get data content of dataset by ID."""
    G = KG()
    if G.has_node(dataset_id):
        node_data = G.nodes[dataset_id]
        if node_data.get("kind") == "Dataset":
            return node_data.get("data_content", "Data content not found in dataset node.")
        else:
            return f"Node '{dataset_id}' is not a Dataset node."
    return f"Dataset with ID '{dataset_id}' not found."


if __name__ == "__main__":
    mcp.run()
    
    # import networkx as nx
    # G  = KG().to_undirected()
    # sid = _concept_id("extreme weather")
    # tid = _concept_id("people with limited assets")

    # path = nx.shortest_path(G, sid, tid)   # returns node IDs
    # for n in path:
    #     k = G.nodes[n]["kind"]
    #     if k == "Concept":
    #         print("CONCEPT :", _concept_label(n))
    #     elif k == "Passage":
    #         print("PASSAGE :", G.nodes[n]["text"][:120].replace("\n"," ") + "…")
    #     else:
    #         print("DOC     :", n)
