
import re
from typing import Optional, Any, List, Dict
def _build_narrative_summary_module(
        #self,
        paragraphs: List[str],
        #sequence: List[str],
        #evidence_map: Dict[str, NarrativeEvidence],
        #registry: CitationRegistry,
    ) -> Dict[str, Any]:
        pattern = re.compile(r"\[\[(F\d+)\]\]")

        # Fix grouped pattern like [[F1][F2][F7]]
        grouped_pattern = re.compile(r"\[\[(F\d+(?:\]\[F\d+)+)\]\]")

        def expand_grouped_refs(text: str) -> str:
            """
            Expands malformed grouped citations like [[F1][F2][F7]]
            into [[F1]][[F2]][[F7]].
            """
            def expand(match: re.Match[str]) -> str:
                inner = match.group(1)  # e.g. "F1][F2][F7"
                # Extract all F# inside it
                fids = re.findall(r"F\d+", inner)
                # Rebuild properly spaced [[F#]] sequence
                return "".join(f"[[{fid}]]" for fid in fids)
            
            return grouped_pattern.sub(expand, text)

        def replace_marker(match: re.Match[str]) -> str:
            fid = match.group(1)
            evidence = evidence_map.get(fid)
            if not evidence:
                return ""
            try:
                number = registry.number_for(evidence.citation)
            except KeyError:
                return ""
            return f"^{number}^"
        
        def collapse_repeated_refs(text: str) -> str:
            """
            Collapses repeated citation markers like:
            ^1^^1^^1^  → ^1^
            ^2^ ^2^    → ^2^
            ^3^   ^3^  → ^3^
            Works for any number.
            """
            # (\^\d+\^) captures a ^number^ group
            # (?:\s*\1)+ matches one or more repeats of the same marker,
            # possibly separated by spaces
            pattern = re.compile(r'(\^\d+\^)(?:\s*\1)+(?!\s*\^)')
            return pattern.sub(r'\1', text)

        def reorder_citation_groups(text: str) -> str:
            """
            Reorders any run of ^n^ markers (with optional internal spaces) into ascending order,
            preserving spacing/punctuation and working at end-of-line too.
            """
            # Match ≥2 markers possibly separated by spaces, stop before space/punct/EOL
            pattern = re.compile(r'(?P<group>(?:\^\d+\^\s*){2,})(?=(?:\s|\W|$))')

            def sort_group(match: re.Match[str]) -> str:
                group = match.group("group")
                # Extract, deduplicate, sort
                numbers = sorted({int(n) for n in re.findall(r"\^(\d+)\^", group)})
                # Rebuild normalized group
                return "".join(f"^{n}^" for n in numbers)

            return pattern.sub(sort_group, text)

        rendered: List[str] = []
        for paragraph in paragraphs:
            paragraph = expand_grouped_refs(paragraph)
            text = pattern.sub(replace_marker, paragraph)
            print(text)
            text = collapse_repeated_refs(text)
            print(text)
            text = reorder_citation_groups(text)
            rendered.append(text)

        if not rendered:
            rendered = ["No supporting facts were returned."]

        return {
            "type": "text",
            "heading": "Summary",
            "texts": rendered,
            #"metadata": {"citations": self._build_citation_metadata(sequence, evidence_map, registry)},
        }



print(_build_narrative_summary_module(["this is a test ^1^^1^^1^^1^ ^1^^1^^1^^1^.",
    "this is a test [[F1][F2][F7]].",
    "this is a test^2^^2^",
    "this is a test^5^^5^^5^",
    "this is spaced ^3^ ^3^ ^3^. ok",
    "mixed ^4^  ^3^  text",
    "no duplicates ^6^ okay",
    "multiple different ^7^^8^^8^ keep separate",
    "wrong order ^7^^1^^8^.",
    "wrong order ^7^^3^^2^^5^^5^,",
    "wrong order ^7^^3^^2^^5^^5^ end"]))