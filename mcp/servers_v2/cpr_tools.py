"""Utilities for CPR Knowledge Graph v2 server.

This module loads the CPR policy knowledge graph data (concepts, passages,
relationships) and exposes helper functions used by the v2 MCP server.
The implementations are adapted from the legacy `cpr_kg_server` but stripped
of FastMCP-specific decorators so the new server can call them directly.
"""

from __future__ import annotations

import ast
import json
import os
import re
import unicodedata
from collections import deque
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:  # Optional dependencies
    from openai import OpenAI
except Exception:  # pragma: no cover - handled gracefully at runtime
    OpenAI = None

try:
    from rapidfuzz import fuzz, process
except Exception:  # pragma: no cover
    fuzz = process = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover
    cosine_similarity = None


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

EXTRAS_DIR = ROOT / "extras"
CONCEPTS_PATH = EXTRAS_DIR / "concepts.csv"
PASSAGES_PATH = EXTRAS_DIR / "labelled_passages.jsonl"
GRAPHML_PATH = EXTRAS_DIR / "knowledge_graph.graphml"


def _load_concepts() -> pd.DataFrame:
    concepts_df = pd.read_csv(CONCEPTS_PATH)
    for column in ["preferred_label", "description"]:
        if column in concepts_df.columns:
            concepts_df[column] = concepts_df[column].fillna("")
    return concepts_df


concepts = _load_concepts()

LABEL_TO_ID = concepts.set_index("preferred_label")["wikibase_id"].to_dict()
ID_TO_LABEL = concepts.set_index("wikibase_id")["preferred_label"].to_dict()


def _tokens(text: str) -> List[str]:
    normalized = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized.lower())
    return [tok for tok in normalized.split() if tok]


def _ngrams(tokens: List[str]) -> List[str]:
    grams = tokens[:]
    grams += [" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
    return grams


LABEL_TO_ID_NORM: Dict[str, str] = {}


def _build_normalized_label_maps() -> None:
    LABEL_TO_ID_NORM.clear()
    for _, row in concepts.iterrows():
        wid = str(row.get("wikibase_id", "")).strip()
        pref = row.get("preferred_label", "")
        if wid and pref:
            LABEL_TO_ID_NORM[" ".join(_tokens(pref))] = wid
        alts = row.get("alternative_labels", [])
        try:
            if isinstance(alts, str):
                alts = ast.literal_eval(alts)
        except Exception:
            alts = []
        for alt in alts or []:
            norm = " ".join(_tokens(alt))
            if norm and norm not in LABEL_TO_ID_NORM:
                LABEL_TO_ID_NORM[norm] = wid


_build_normalized_label_maps()


def _concept_id(label: str) -> Optional[str]:
    if not label:
        return None
    label = label.strip()
    if label in LABEL_TO_ID:
        return LABEL_TO_ID[label]
    norm = " ".join(_tokens(label))
    return LABEL_TO_ID_NORM.get(norm)


def _concept_label(wikibase_id: str) -> str:
    return ID_TO_LABEL.get(wikibase_id, wikibase_id)


def _resolve_concept_id_fuzzy(label: str) -> Tuple[Optional[str], Optional[str]]:
    if not label:
        return None, None
    cid = _concept_id(label)
    if cid:
        return cid, _concept_label(cid)
    if not process or not fuzz:
        return None, None
    candidates: List[Tuple[str, str]] = []
    for _, row in concepts.iterrows():
        wid = str(row.get("wikibase_id", "")).strip()
        pref = str(row.get("preferred_label", ""))
        if wid and pref:
            candidates.append((pref, wid))
        alts = row.get("alternative_labels", [])
        try:
            if isinstance(alts, str):
                alts = ast.literal_eval(alts)
        except Exception:
            alts = []
        for alt in alts or []:
            candidates.append((str(alt), wid))
    labels = [c[0] for c in candidates]
    matches = process.extract(label, labels, scorer=fuzz.token_set_ratio, limit=1)
    if matches:
        best_label, score, idx = matches[0]
        if score >= 70:
            wid = candidates[idx][1]
            return wid, _concept_label(wid)
    return None, None


def _search_concepts_by_text_impl(text: str, limit: int) -> List[Dict[str, Any]]:
    if not text:
        return []
    text_lower = text.lower()
    results = []
    for _, row in concepts.iterrows():
        label = row.get("preferred_label", "")
        if text_lower in label.lower():
            results.append({"label": label, "wikibase_id": row.get("wikibase_id")})
        if len(results) >= limit:
            break
    return results


def _search_concepts_fuzzy_impl(text: str, limit: int) -> List[Dict[str, Any]]:
    if not text or not process:
        return []
    labels = concepts["preferred_label"].tolist()
    matches = process.extract(text, labels, scorer=fuzz.token_set_ratio, limit=limit)
    results = []
    for label, score, _ in matches:
        if score < 60:
            continue
        wid = LABEL_TO_ID.get(label)
        if wid:
            results.append({"label": label, "wikibase_id": wid, "score": score})
    return results


def _find_concept_matches_by_ngrams_impl(text: str, top_k: int) -> List[Dict[str, Any]]:
    tokens = _tokens(text)
    grams = _ngrams(tokens)
    matches = []
    for gram in grams:
        cid = LABEL_TO_ID.get(gram)
        if cid:
            matches.append({"label": gram, "wikibase_id": cid})
        norm = " ".join(_tokens(gram))
        cid = LABEL_TO_ID_NORM.get(norm)
        if cid:
            matches.append({"label": _concept_label(cid), "wikibase_id": cid})
    unique = []
    seen = set()
    for match in matches:
        key = match["wikibase_id"]
        if key not in seen:
            seen.add(key)
            unique.append(match)
        if len(unique) >= top_k:
            break
    return unique


def _ensure_embeddings_loaded() -> Optional[np.ndarray]:
    if "vector_embedding" not in concepts.columns:
        return None
    if concepts["vector_embedding"].isna().any():
        return None
    try:
        vectors = concepts["vector_embedding"].apply(lambda x: np.array(json.loads(x) if isinstance(x, str) else x))
        matrix = np.stack(vectors.values)
        return matrix
    except Exception:
        return None


_CONCEPT_EMBEDDINGS = _ensure_embeddings_loaded()


def _get_top_concepts_by_query_impl(query: str, top_k: int) -> List[Dict[str, Any]]:
    if not query:
        return []
    if OpenAI is None or cosine_similarity is None or _CONCEPT_EMBEDDINGS is None:
        # Fall back to local method if embeddings unavailable
        return _get_top_concepts_by_query_local_impl(query, top_k)
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        embedding = client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
        embedding = np.array(embedding).reshape(1, -1)
        sims = cosine_similarity(embedding, _CONCEPT_EMBEDDINGS)[0]
        top_indices = sims.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            row = concepts.iloc[idx]
            results.append({
                "label": row.get("preferred_label"),
                "wikibase_id": row.get("wikibase_id"),
                "score": float(sims[idx]),
            })
        return results
    except Exception:
        return _get_top_concepts_by_query_local_impl(query, top_k)


def _get_top_concepts_by_query_local_impl(query: str, top_k: int) -> List[Dict[str, Any]]:
    tokens = set(_tokens(query))
    scores: List[Tuple[int, Dict[str, Any]]] = []
    for _, row in concepts.iterrows():
        label_tokens = set(_tokens(row.get("preferred_label", "")))
        score = len(tokens & label_tokens)
        if score:
            scores.append((score, {
                "label": row.get("preferred_label"),
                "wikibase_id": row.get("wikibase_id"),
                "score": score,
            }))
    scores.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scores[:top_k]]


def _concept_candidates_from_query(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    candidates = []
    candidates.extend(_find_concept_matches_by_ngrams_impl(query, top_k))
    candidates.extend(_get_top_concepts_by_query_local_impl(query, top_k))
    if OpenAI and cosine_similarity and _CONCEPT_EMBEDDINGS is not None:
        candidates.extend(_get_top_concepts_by_query_impl(query, top_k))
    seen = set()
    deduped = []
    for cand in candidates:
        cid = cand.get("wikibase_id")
        if cid and cid not in seen:
            seen.add(cid)
            deduped.append(cand)
    return deduped[:top_k]


def _load_passages() -> List[Dict[str, Any]]:
    if not PASSAGES_PATH.exists():
        return []
    passages: List[Dict[str, Any]] = []
    with PASSAGES_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                passages.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return passages


PASSAGES = _load_passages()


def _search_passages_textual(terms: List[str], limit: int = 5) -> List[Dict[str, Any]]:
    results = []
    for passage in PASSAGES:
        text = passage.get("text", "").lower()
        if all(term.lower() in text for term in terms):
            results.append(passage)
        if len(results) >= limit:
            break
    return results


def _get_passages_mentioning_concept_impl(concept: str, limit: int = 5) -> List[Dict[str, Any]]:
    if not concept:
        return []
    results = []
    for passage in PASSAGES:
        metadata = passage.get("metadata", {}) or {}
        spans = metadata.get("concept", {})
        if isinstance(spans, dict) and metadata.get("concept", {}).get("preferred_label") == concept:
            results.append(passage)
        elif concept.lower() in passage.get("text", "").lower():
            results.append(passage)
        if len(results) >= limit:
            break
    if not results:
        results = _search_passages_textual([concept], limit)
    return results


@lru_cache(maxsize=1)
def KG() -> nx.MultiDiGraph:
    if not GRAPHML_PATH.exists():
        return nx.MultiDiGraph()
    return nx.read_graphml(GRAPHML_PATH)


SEMANTIC_DEBUG_LOG: deque[str] = deque(maxlen=200)


def get_semantic_debug_log(limit: int = 50) -> Dict[str, Any]:
    return {"lines": list(SEMANTIC_DEBUG_LOG)[-limit:]}


def debug_embedding_status() -> Dict[str, Any]:
    status = {
        "embeddings_loaded": _CONCEPT_EMBEDDINGS is not None,
        "rows": len(concepts),
    }
    if OpenAI is None:
        status["note"] = "OpenAI client not available"
    return status


def get_concepts() -> List[str]:
    return concepts["preferred_label"].tolist()


def check_concept_exists(concept: str) -> bool:
    return concept in LABEL_TO_ID or " ".join(_tokens(concept)) in LABEL_TO_ID_NORM


def get_semantically_similar_concepts(concept: str) -> List[str]:
    results = _get_top_concepts_by_query_impl(concept, top_k=5)
    if not results:
        return []
    return [r.get("label") for r in results]


def search_concepts_by_text(text: str, limit: int = 10) -> List[Dict[str, Any]]:
    return _search_concepts_by_text_impl(text, limit)


def get_top_concepts_by_query(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return _get_top_concepts_by_query_impl(query, top_k)


def get_top_concepts_by_query_local(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return _get_top_concepts_by_query_local_impl(query, top_k)


def search_concepts_fuzzy(text: str, limit: int = 10) -> List[Dict[str, Any]]:
    return _search_concepts_fuzzy_impl(text, limit)


def describe_server() -> Dict[str, Any]:
    return {
        "dataset": DATASET_TITLE,
        "total_concepts": len(concepts),
        "passage_count": len(PASSAGES),
        "graph_loaded": GRAPHML_PATH.exists(),
    }


def find_concept_matches_by_ngrams(text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return _find_concept_matches_by_ngrams_impl(text, top_k)


def get_related_concepts(concept: str) -> List[Dict[str, Any]]:
    cid = _concept_id(concept)
    if not cid:
        return []
    results = []
    G = KG()
    for _, neighbor, data in G.edges(cid, data=True):
        if data.get("type") == "RELATED_TO":
            results.append({"label": _concept_label(neighbor), "edge_type": data.get("type")})
    return results


def get_subconcepts(concept: str) -> List[Dict[str, Any]]:
    cid = _concept_id(concept)
    if not cid:
        return []
    results = []
    G = KG()
    for _, neighbor, data in G.out_edges(cid, data=True):
        if data.get("type") == "HAS_SUBCONCEPT":
            results.append({"label": _concept_label(neighbor), "edge_type": data.get("type")})
    return results


def get_parent_concepts(concept: str) -> List[Dict[str, Any]]:
    cid = _concept_id(concept)
    if not cid:
        return []
    results = []
    G = KG()
    for neighbor, _, data in G.in_edges(cid, data=True):
        if data.get("type") == "HAS_SUBCONCEPT":
            results.append({"label": _concept_label(neighbor), "edge_type": data.get("type")})
    return results


def get_alternative_labels(concept: str) -> List[str]:
    row = concepts[concepts["preferred_label"] == concept]
    if row.empty:
        return []
    alts = row.iloc[0].get("alternative_labels")
    if isinstance(alts, str):
        try:
            alts = ast.literal_eval(alts)
        except Exception:
            alts = []
    return alts or []


def get_description(concept: str) -> str:
    row = concepts[concepts["preferred_label"] == concept]
    if row.empty:
        return ""
    return str(row.iloc[0].get("description", ""))


def get_passages_mentioning_concept(concept: str, limit: int = 5) -> List[Dict[str, Any]]:
    return _get_passages_mentioning_concept_impl(concept, limit)


def get_concept_graph_neighbors(concept: str, limit: int = 15) -> List[Dict[str, Any]]:
    cid = _concept_id(concept)
    if not cid:
        return []
    G = KG()
    results = []
    count = 0
    for _, neighbor, data in G.out_edges(cid, data=True):
        results.append({
            "source_label": _concept_label(cid),
            "target_label": _concept_label(neighbor),
            "edge_type": data.get("type"),
        })
        count += 1
        if count >= limit:
            break
    return results


def find_concept_path_rich(source_concept: str, target_concept: str) -> List[Dict[str, Any]]:
    source_id = _concept_id(source_concept)
    target_id = _concept_id(target_concept)
    if not source_id or not target_id:
        return []
    G = KG()
    try:
        path = nx.shortest_path(G, source=source_id, target=target_id)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    result = []
    for node in path:
        result.append({"wikibase_id": node, "label": _concept_label(node)})
    return result


def passages_mentioning_both_concepts(concept_a: str, concept_b: str, limit: int = 5) -> List[Dict[str, Any]]:
    passages_a = _get_passages_mentioning_concept_impl(concept_a, limit=limit * 2)
    passages_b = _get_passages_mentioning_concept_impl(concept_b, limit=limit * 2)
    ids_a = {p.get("metadata", {}).get("passage_id") for p in passages_a}
    results = []
    for passage in passages_b:
        pid = passage.get("metadata", {}).get("passage_id")
        if pid in ids_a:
            results.append(passage)
        if len(results) >= limit:
            break
    return results


def find_concept_path_with_edges(source_concept: str, target_concept: str, max_len: int = 5) -> List[Dict[str, Any]]:
    source_id = _concept_id(source_concept)
    target_id = _concept_id(target_concept)
    if not source_id or not target_id:
        return []
    G = KG()
    paths = []
    try:
        for path in nx.all_shortest_paths(G, source_id, target_id):
            if len(path) - 1 > max_len:
                continue
            path_edges = []
            for u, v in zip(path, path[1:]):
                data = G.get_edge_data(u, v)
                edge_type = next(iter(data.values()))["type"] if data else ""
                path_edges.append({
                    "source": _concept_label(u),
                    "target": _concept_label(v),
                    "edge_type": edge_type,
                })
            paths.append(path_edges)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    return paths


def discover_policy_context_for_query(
    query: str,
    top_concepts: int = 3,
    neighbor_limit: int = 10,
    passage_limit: int = 8,
) -> Dict[str, Any]:
    candidates = _concept_candidates_from_query(query, top_k=top_concepts * 3)
    seeds = []
    for cand in candidates:
        wid = cand.get("wikibase_id")
        if wid:
            seeds.append({"label": _concept_label(wid), "wikibase_id": wid})
        if len(seeds) >= top_concepts:
            break

    G = KG()
    neighbors: List[Dict[str, Any]] = []
    for concept in seeds:
        cid = concept["wikibase_id"]
        count = 0
        for _, neighbor, data in G.out_edges(cid, data=True):
            neighbors.append({
                "source_label": concept["label"],
                "target_label": _concept_label(neighbor),
                "edge_type": data.get("type"),
            })
            count += 1
            if count >= neighbor_limit:
                break

    passages: List[Dict[str, Any]] = []
    for concept in seeds:
        hits = _get_passages_mentioning_concept_impl(concept["label"], limit=passage_limit)
        passages.extend(hits)
        if len(passages) >= passage_limit:
            break
    if not passages:
        passages = _search_passages_textual(["solar"], limit=passage_limit)

    return {
        "concepts": seeds,
        "neighbors": neighbors,
        "passages": passages,
    }


def explain_concept_relationship(source_concept: str, target_concept: str) -> Dict[str, Any]:
    paths = find_concept_path_with_edges(source_concept, target_concept, max_len=5)
    if not paths:
        return {"paths": [], "notes": "No connection found"}
    return {"paths": paths}


def get_available_datasets() -> List[Dict[str, Any]]:
    return [
        {
            "id": "cpr_kg",
            "title": DATASET_TITLE,
            "description": "CPR knowledge graph export with Brazilian policy concepts and passages.",
        }
    ]


def get_dataset_content(dataset_id: str) -> Dict[str, Any]:
    if dataset_id != "cpr_kg":
        return {}
    return {
        "concepts_count": len(concepts),
        "passages_count": len(PASSAGES),
    }


def get_dataset_metadata() -> Dict[str, Any]:
    return {
        "dataset": DATASET_TITLE,
        "concepts": len(concepts),
        "passages": len(PASSAGES),
        "graph_nodes": KG().number_of_nodes(),
    }

