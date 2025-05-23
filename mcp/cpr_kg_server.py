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

load_dotenv()

mcp = FastMCP("climate-policy-radar-kg-server")
concepts = pd.read_csv("../extras/concepts.csv")  # TODO: Turn the Embeddings into a list here instead of in the tool call

# maps built from the concepts dataframe you already have in RAM
LABEL_TO_ID = concepts.set_index("preferred_label")["wikibase_id"].to_dict()
ID_TO_LABEL = concepts.set_index("wikibase_id")["preferred_label"].to_dict()

def _concept_id(label: str) -> str | None:
    return LABEL_TO_ID.get(label)

def _concept_label(cid: str) -> str:
    return ID_TO_LABEL.get(cid, cid)        # fall back to the ID if label unknown


GRAPHML_PATH = "../extras/knowledge_graph.graphml"   # output of build_kg.py

@lru_cache(maxsize=1)
def KG() -> nx.MultiDiGraph:
    """
    Lazily load the NetworkX MultiDiGraph produced by build_kg.py.
    Cached so every tool call shares the same in-memory object.
    """
    G = nx.read_graphml(GRAPHML_PATH)

    # --- Add dummy dataset ---
    dataset_node_id = "DUMMY_DATASET_EXTREME_WEATHER"
    dataset_label = "Extreme Weather Impact Dataset"
    
    # Sample data for the dataset
    sample_data = [
        {"event_id": "EW001", "year": 2023, "type": "Hurricane", "location": "Florida", "impact_rating": 5, "description": "Major hurricane causing widespread damage."},
        {"event_id": "EW002", "year": 2023, "type": "Wildfire", "location": "California", "impact_rating": 4, "description": "Extensive wildfires affecting multiple counties."},
        {"event_id": "EW003", "year": 2024, "type": "Flood", "location": "Germany", "impact_rating": 5, "description": "Severe flooding after heavy rainfall."},
        {"event_id": "EW004", "year": 2024, "type": "Drought", "location": "Horn of Africa", "impact_rating": 4, "description": "Prolonged drought leading to food shortages."}
    ]

    if not G.has_node(dataset_node_id):
        G.add_node(
            dataset_node_id,
            kind="Dataset",
            label=dataset_label,
            description="A sample dataset detailing impacts of various extreme weather events.",
            data_content=sample_data  # Storing data as a node attribute
        )

    # Link to "extreme weather" concept if it exists
    extreme_weather_concept_id = _concept_id("extreme weather")
    if extreme_weather_concept_id and G.has_node(extreme_weather_concept_id):
        if not G.has_edge(extreme_weather_concept_id, dataset_node_id):
            G.add_edge(extreme_weather_concept_id, dataset_node_id, type="HAS_DATASET_ABOUT")
        if not G.has_edge(dataset_node_id, extreme_weather_concept_id): # Optional: reverse link
            G.add_edge(dataset_node_id, extreme_weather_concept_id, type="DATASET_ON_TOPIC")
    # --- End of dummy dataset addition ---
    
    return G


if "vector_embedding" not in concepts.columns or concepts["vector_embedding"].isna().any():
    print("Generating vector embeddings for concepts")
    client = OpenAI()                                     # uses OPENAI_API_KEY from .env
    resp = client.embeddings.create(
        input=concepts["preferred_label"].tolist(),       # list-of-strings, one call
        model="text-embedding-3-small"
    )
    # pull the vectors out of the response
    concepts["vector_embedding"] = [row.embedding for row in resp.data]
    concepts.to_csv("../extras/concepts.csv", index=False)
    print("Vector embeddings generated and saved to concepts.csv")


# Read passages from jsonl file
with open("../extras/labelled_passages.jsonl", "r") as f:
    passages = [json.loads(line) for line in f]


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
    """
    Get all concepts in the climate policy radar knowledge graph. This takes no arguments.
    """
    return concepts["preferred_label"].tolist()

@mcp.tool()
def CheckConceptExists(concept: str) -> bool:
    """
    Check if a given concept exists in the climate policy radar knowledge graph.
    """
    return concept in concepts["preferred_label"].tolist()

@mcp.tool()
def GetSemanticallySimilarConcepts(concept: str) -> List[str]:
    """
    Return up to five concepts whose pretrained-embedding cosine similarity
    to *concept* is highest.
    """
    client = OpenAI()
    concept_emb = client.embeddings.create(
        input=concept,
        model="text-embedding-3-small"
    ).data[0].embedding                      # list[float] length = 1536

    # --- make sure every row is a list[float] --------------------------
    if isinstance(concepts.loc[0, "vector_embedding"], str):
        concepts["vector_embedding"] = concepts["vector_embedding"].apply(ast.literal_eval)
    # -------------------------------------------------------------------

    emb_matrix = np.vstack(concepts["vector_embedding"].to_numpy())       # shape (N, 1536)
    sims       = cosine_similarity([concept_emb], emb_matrix)[0]          # shape (N,)

    top_idx = sims.argsort()[-5:][::-1]                                   # best → worst
    return concepts.iloc[top_idx]["preferred_label"].tolist()

@mcp.tool()
def GetRelatedConcepts(concept: str) -> List[str]:
    """
    Get all related concepts of a given concept in the climate policy radar knowledge graph.
    """
    return concepts[concepts["preferred_label"] == concept]["related_concepts"].tolist()

@mcp.tool()
def GetSubconcepts(concept: str) -> List[str]:
    """
    Get all subconcepts of a given concept in the climate policy radar knowledge graph.
    """

    subconcept_ids = concepts[concepts["preferred_label"] == concept]["has_subconcept"].tolist()
    subconcept_names = concepts[concepts["wikibase_id"].isin(subconcept_ids)]["preferred_label"].tolist()

    return subconcept_names

@mcp.tool()
def GetParentConcepts(concept: str) -> str:
    """
    Get the parent concept of a given concept in the climate policy radar knowledge graph.
    """

    parent_ids = concepts[concepts["preferred_label"] == concept]["subconcept_of"].tolist()
    parent_names = concepts[concepts["wikibase_id"].isin(parent_ids)]["preferred_label"].tolist()
    return parent_names

@mcp.tool()
def GetAlternativeLabels(concept: str) -> List[str]:
    """
    Get all alternative labels of a given concept in the climate policy radar knowledge graph.
    """
    return concepts[concepts["preferred_label"] == concept]["alternative_labels"].tolist()

@mcp.tool()
def GetDescription(concept: str) -> str:
    """
    Get the description of a given concept in the climate policy radar knowledge graph.
    """
    filtered_concepts = concepts[concepts["preferred_label"] == concept]
    if not filtered_concepts.empty:
        description = filtered_concepts["description"].iloc[0]
        # Ensure description is a string, not NaN or other types
        if pd.notna(description):
            return str(description)
    return "" # Return empty string if concept or description not found


### GRAPH TOOLS

@mcp.tool()
def GetPassagesMentioningConcept(concept: str, limit: int = 10) -> List[dict]:
    """
    Return up to *limit* passages that MENTION the given concept.
    Each record: {passage_id, doc_id, text}.
    """
    cid = _concept_id(concept)
    if not cid:
        return []

    G = KG()
    pids = [
        u for u, v, d in G.in_edges(cid, data=True)
        if d["type"] == "MENTIONS"
    ][:limit]

    out = []
    for pid in pids:
        n = G.nodes[pid]
        out.append({"passage_id": pid, "doc_id": n["doc"], "text": n["text"]})
    return out


@mcp.tool()
def GetConceptGraphNeighbors(
    concept: str,
    edge_types: Optional[List[str]] = None,
    direction: str = "both",
    max_results: int = 25,
) -> List[dict]:
    """
    Return neighbor nodes (concepts, datasets, etc.) directly connected to *concept* in the graph.

    Parameters
    ----------
    concept:      preferred label of the starting concept
    edge_types:   list like ["RELATED_TO","SUBCONCEPT_OF"]; None = any edge
    direction:    "out", "in", or "both"
    max_results:  hard cap on number of neighbors returned

    Each record: {node_id, label, kind, via_edge}
    """
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
    """
    Return the shortest path with node details + edge types.

    Each element:
      { "from_id":…, "from_kind":…, "edge": "MENTIONS",
        "to_id":…,   "to_kind":…,   "to_label_or_text": … }
    """
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
def PassagesMentioningBoth(concept_a: str, concept_b: str, limit: int = 10) -> List[dict]:
    """
    Passages where *both* concepts are mentioned.
    """
    a_id, b_id = _concept_id(concept_a), _concept_id(concept_b)
    if None in (a_id, b_id):
        return []
    G   = KG()
    out = []
    for pid, pdata in G.nodes(data=True):
        if pdata.get("kind") != "Passage":
            continue
        if G.has_edge(pid, a_id) and G.has_edge(pid, b_id):
            out.append({
                "passage_id": pid,
                "doc_id"    : pdata["doc"],
                "text"      : pdata["text"],
            })
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
    """
    Return every shortest path (≤ max_len hops) between two concepts,
    *including* the edge type between each hop.
    Each path is a list like:
      [{"source":"Education","edge":"RELATED_TO","target":"Human capital"},
       {"source":"Human capital","edge":"HAS_SUBCONCEPT","target":"Agriculture"}]
    """
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
    """
    Produce a short narrative of how *source_concept* links to *target_concept*,
    using the first shortest path with edge labels plus each concept's description.
    """
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
def GetDatasetContent(dataset_id: str) -> List[dict] | str:
    """
    Retrieves the data content of a dataset node identified by its unique ID.
    Returns a list of records if found, or an error message string.
    Parameters
    ----------
    dataset_id: The unique ID of the dataset node (e.g., 'DUMMY_DATASET_EXTREME_WEATHER').
    """
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

