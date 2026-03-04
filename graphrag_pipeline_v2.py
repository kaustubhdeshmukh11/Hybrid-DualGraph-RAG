"""
Hybrid DualGraph RAG — Phase 2: Dual Extraction Pipeline
=========================================================
Combines SpaCy (syntactic SVO triples) + Groq LLM (semantic relations)
for a richer knowledge graph with lower LLM compute cost.

Usage:
    python graphrag_pipeline_v2.py <path_to_pdf>

Architecture:
    SpaCy  → inline Subject-Verb-Object triples  (zero LLM tokens)
    LLM    → cross-sentence, implicit, hierarchical relationships
    Both   → merged & deduplicated → MERGE into Neo4j
"""

import os
import re
import sys
import json
import time
import logging
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------------------------
# Import shared utilities from Phase 1
# ---------------------------------------------------------------------------
from graphrag_pipeline import (
    load_neo4j_credentials,
    get_neo4j_driver,
    setup_neo4j_schema,
    load_pdf,
    chunk_text,
    get_embedding_model,
    generate_embeddings,
    ingest_document,
    ingest_chunk,
    ingest_entities_and_relations,
    deduplicate_entities,
    make_chunk_id,
    LLMBenchmark,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
load_dotenv()
_GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))
benchmark = LLMBenchmark()          # Phase 2 LLM tracker


# =============================================================================
# 1.  SPACY DEPENDENCY-BASED EXTRACTION
# =============================================================================

_SPACY_NLP = None  # module-level cache


def _get_spacy_model():
    """Load (and cache) the spaCy English model."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        _SPACY_NLP = spacy.load("en_core_web_sm")
        log.info("Loaded spaCy model: en_core_web_sm")
    return _SPACY_NLP


def _expand_noun(token):
    """
    Expand a single token to its full noun phrase by collecting
    compound modifiers and adjectival modifiers attached to it.

    Example: "machine learning" instead of just "learning"
    """
    parts = []
    for child in token.children:
        if child.dep_ in ("compound", "amod", "nmod"):
            parts.append(child.text)
    parts.append(token.text)
    return " ".join(parts)


def _normalize_entity(text: str) -> str:
    """Strip punctuation and convert to Title Case to match LLM output."""
    text = re.sub(r"[^\w\s-]", "", text).strip()
    return text.title() if text else ""


def extract_with_spacy(text_chunk: str) -> dict:
    """
    Extract Subject-Verb-Object triples using spaCy dependency parsing.

    Strategy:
        1. Parse the text and iterate over every token.
        2. For each verb that is a ROOT or has verbal POS:
           - Find its nsubj / nsubjpass  (subject)
           - Find its dobj / pobj / attr  (object)
        3. Expand nouns to include compounds ("machine learning", not "learning").
        4. Normalize entity names to Title Case.

    Returns:
        dict with "entities" and "relationships" in the same schema as the LLM extractor.
    """
    nlp = _get_spacy_model()
    doc = nlp(text_chunk)

    entities_set: dict[str, str] = {}   # name -> type (dedup)
    relationships: list[dict] = []

    for token in doc:
        # Only process verbs
        if token.pos_ not in ("VERB",):
            continue

        # Collect subjects and objects attached to this verb
        subjects = []
        objects = []

        for child in token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subjects.append(child)
            elif child.dep_ in ("dobj", "pobj", "attr", "oprd"):
                objects.append(child)

        # Also check prepositional objects:  verb -> prep -> pobj
        for child in token.children:
            if child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        objects.append(grandchild)

        # Build triples for every (subject, object) pair
        for subj in subjects:
            subj_text = _normalize_entity(_expand_noun(subj))
            if not subj_text:
                continue

            # Determine entity type from spaCy NER if available
            subj_ent_type = "CONCEPT"
            if subj.ent_type_:
                subj_ent_type = subj.ent_type_
            entities_set[subj_text] = subj_ent_type

            for obj in objects:
                obj_text = _normalize_entity(_expand_noun(obj))
                if not obj_text:
                    continue

                obj_ent_type = "CONCEPT"
                if obj.ent_type_:
                    obj_ent_type = obj.ent_type_
                entities_set[obj_text] = obj_ent_type

                # Relation = lemmatized verb in SCREAMING_SNAKE_CASE
                relation = token.lemma_.upper().replace(" ", "_")
                relation = re.sub(r"[^A-Z0-9_]", "", relation)
                if not relation:
                    relation = "RELATED_TO"

                relationships.append({
                    "head": subj_text,
                    "relation": relation,
                    "tail": obj_text,
                })

    # Also capture named entities recognized by spaCy NER (even if not in a triple)
    for ent in doc.ents:
        ename = _normalize_entity(ent.text)
        if ename and ename not in entities_set:
            entities_set[ename] = ent.label_

    entities = [{"name": n, "type": t} for n, t in entities_set.items()]

    log.debug(
        "SpaCy extracted %d entities, %d relationships",
        len(entities), len(relationships),
    )
    return {"entities": entities, "relationships": relationships}


# =============================================================================
# 2.  REFINED LLM EXTRACTION  (semantic / cross-sentence only)
# =============================================================================

SEMANTIC_PROMPT = """You are an expert knowledge-graph builder working as the SECOND pass in a dual-extraction pipeline.

A dependency parser has ALREADY extracted all **inline Subject-Verb-Object triples** from this text.
Your job is to find ONLY the relationships that a simple syntactic parser CANNOT detect:

1. **Cross-sentence relationships** — connections between entities mentioned in different sentences.
2. **Implicit / inferred relationships** — e.g., "After graduating from MIT, she joined Google" implies EDUCATED_AT.
3. **Hierarchical relationships** — IS_A, PART_OF, SUBCLASS_OF, MEMBER_OF.
4. **World-knowledge relationships** — e.g., knowing that "Python" IS_A "Programming Language".
5. **Temporal or causal relationships** — PRECEDED_BY, CAUSED_BY, LED_TO.

Do NOT extract simple subject-verb-object facts (e.g., "Apple develops iPhone") — those are already covered.

Return ONLY valid JSON (no markdown, no explanation):
{{
    "entities": [
        {{"name": "Entity Name", "type": "PERSON|ORGANIZATION|CONCEPT|LOCATION|EVENT|OBJECT|TECHNOLOGY"}}
    ],
    "relationships": [
        {{"head": "Entity1", "relation": "RELATIONSHIP_TYPE", "tail": "Entity2"}}
    ]
}}

Rules:
- Normalize entity names to Title Case.
- Use SCREAMING_SNAKE_CASE for relationship types.
- Focus on quality over quantity — only extract what a syntactic parser would miss.

TEXT:
{text}
"""


def extract_semantic_llm(
    chunk_text: str,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    chunk_index: int = -1,
) -> dict:
    """
    LLM extraction focused on semantic / cross-sentence relationships
    that spaCy dependency parsing cannot capture.

    Returns:
        dict with "entities" and "relationships".
    """
    empty = {"entities": [], "relationships": []}
    try:
        t0 = time.time()
        response = _GROQ_CLIENT.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert knowledge graph builder. "
                        "Always respond with valid JSON only. "
                        "No markdown, no explanation."
                    ),
                },
                {"role": "user", "content": SEMANTIC_PROMPT.format(text=chunk_text)},
            ],
            temperature=0.1,
            max_tokens=3000,
        )
        latency = time.time() - t0
        raw = response.choices[0].message.content.strip()

        # --- Benchmark tracking ---
        usage = response.usage
        if usage:
            benchmark.record(
                prompt_tokens=usage.prompt_tokens or 0,
                completion_tokens=usage.completion_tokens or 0,
                latency=latency,
                chunk_index=chunk_index,
            )

        # Parse JSON (with fallbacks)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        if "```" in raw:
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        log.warning("Could not parse LLM response for chunk: %s…", chunk_text[:60])
        return empty

    except Exception as e:
        log.error("LLM extraction error: %s", e)
        return empty


# =============================================================================
# 3.  MERGE EXTRACTIONS
# =============================================================================

def merge_extractions(spacy_result: dict, llm_result: dict) -> dict:
    """
    Combine spaCy and LLM extraction results into a single pool.
    Deduplicates entities by lowercase name and relationships by (head, rel, tail).
    """
    # --- Merge entities (dedup by lowercase name) --------------------------
    entity_map: dict[str, dict] = {}
    for ent in spacy_result.get("entities", []) + llm_result.get("entities", []):
        key = ent["name"].strip().lower()
        if key not in entity_map:
            entity_map[key] = {
                "name": ent["name"].strip().title(),
                "type": ent.get("type", "CONCEPT").upper(),
            }

    # --- Merge relationships (dedup by head+rel+tail) ----------------------
    rel_set: set[tuple] = set()
    merged_rels: list[dict] = []
    for rel in spacy_result.get("relationships", []) + llm_result.get("relationships", []):
        head = rel.get("head", "").strip().title()
        tail = rel.get("tail", "").strip().title()
        relation = rel.get("relation", "RELATED_TO").strip().upper()
        key = (head.lower(), relation, tail.lower())
        if key not in rel_set and head and tail:
            rel_set.add(key)
            merged_rels.append({"head": head, "relation": relation, "tail": tail})

    merged = {
        "entities": list(entity_map.values()),
        "relationships": merged_rels,
    }
    return merged


# =============================================================================
# 4.  PHASE COMPARISON REPORT
# =============================================================================

def load_phase1_benchmark(path: str = "llm_benchmark_phase1.json") -> dict | None:
    """Load Phase 1 benchmark from disk for comparison."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        log.warning("Phase 1 benchmark file not found at %s", path)
        return None


def print_comparison(phase2_summary: dict, phase1_data: dict | None):
    """Print a side-by-side comparison of Phase 1 vs Phase 2 LLM usage."""
    print("\n" + "=" * 65)
    print("  📊  BENCHMARK COMPARISON — Phase 1 (LLM-Only) vs Phase 2 (Dual)")
    print("=" * 65)

    p1 = phase1_data["summary"] if phase1_data else None
    p2 = phase2_summary

    rows = [
        ("LLM calls",            "llm_calls"),
        ("Total prompt tokens",  "total_prompt_tokens"),
        ("Total completion tokens", "total_completion_tokens"),
        ("Total tokens",         "total_tokens"),
        ("Total LLM latency (s)","total_latency_s"),
        ("Avg latency / chunk",  "avg_latency_per_chunk_s"),
        ("Avg tokens / chunk",   "avg_tokens_per_chunk"),
    ]

    header = f"  {'Metric':<28} {'Phase 1':>12} {'Phase 2':>12} {'Saved':>12}"
    print(header)
    print("  " + "-" * 64)

    for label, key in rows:
        v1 = p1[key] if p1 else "N/A"
        v2 = p2[key]
        if p1 and isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            diff = v1 - v2
            pct = (diff / v1 * 100) if v1 != 0 else 0
            saved_str = f"{diff:+,.0f} ({pct:+.1f}%)"
        else:
            saved_str = "—"
        v1_str = f"{v1:>,}" if isinstance(v1, int) else (f"{v1:>.2f}" if isinstance(v1, float) else str(v1))
        v2_str = f"{v2:>,}" if isinstance(v2, int) else f"{v2:>.2f}"
        print(f"  {label:<28} {v1_str:>12} {v2_str:>12} {saved_str:>12}")

    print("=" * 65)


# =============================================================================
# 5.  MAIN PIPELINE ORCHESTRATOR (DUAL)
# =============================================================================

def run_pipeline_v2(pdf_path: str):
    """
    End-to-end Dual-Extraction GraphRAG ingestion pipeline.

    Steps:
        1. Connect to Neo4j & ensure schema
        2. Parse PDF → text chunks
        3. Generate embeddings
        4. Dual extraction (SpaCy + LLM) → merge
        5. Ingest into Neo4j
        6. Deduplicate entities
        7. Benchmark comparison
    """
    separator = "=" * 65
    print(f"\n{separator}")
    print("  GRAPHRAG DUAL-EXTRACTION PIPELINE  (Phase 2)")
    print(f"{separator}\n")

    # ---- Step 1: Neo4j connection -----------------------------------------
    log.info("[Step 1/7] Connecting to Neo4j…")
    creds = load_neo4j_credentials("neo4j.txt")
    driver = get_neo4j_driver(creds)
    setup_neo4j_schema(driver)

    # ---- Step 2: PDF parsing & chunking -----------------------------------
    log.info("[Step 2/7] Parsing PDF and creating chunks…")
    full_text = load_pdf(pdf_path)
    chunks = chunk_text(full_text, chunk_size=1000, overlap=200)
    doc_name = Path(pdf_path).stem

    # ---- Step 3: Embedding generation -------------------------------------
    log.info("[Step 3/7] Generating embeddings for %d chunks…", len(chunks))
    model = get_embedding_model()
    embeddings = generate_embeddings(chunks, model)

    # ---- Step 4: DUAL extraction ------------------------------------------
    log.info("[Step 4/7] Running DUAL extraction (SpaCy + LLM)…")
    merged_extractions: list[dict] = []

    for i, chunk in enumerate(chunks):
        log.info("  Chunk %d/%d", i + 1, len(chunks))

        # Path A: SpaCy (syntactic — zero LLM cost)
        spacy_result = extract_with_spacy(chunk)
        n_sp_ent = len(spacy_result.get("entities", []))
        n_sp_rel = len(spacy_result.get("relationships", []))
        log.info("    SpaCy  → %d entities, %d relationships", n_sp_ent, n_sp_rel)

        # Path B: LLM  (semantic — reduced prompt, cross-sentence only)
        llm_result = extract_semantic_llm(chunk, chunk_index=i)
        n_llm_ent = len(llm_result.get("entities", []))
        n_llm_rel = len(llm_result.get("relationships", []))
        log.info("    LLM    → %d entities, %d relationships", n_llm_ent, n_llm_rel)

        # Merge
        merged = merge_extractions(spacy_result, llm_result)
        merged_extractions.append(merged)
        n_m_ent = len(merged.get("entities", []))
        n_m_rel = len(merged.get("relationships", []))
        log.info("    Merged → %d entities, %d relationships", n_m_ent, n_m_rel)

    # ---- Step 5: Neo4j ingestion ------------------------------------------
    log.info("[Step 5/7] Ingesting data into Neo4j…")
    with driver.session() as session:
        session.execute_write(ingest_document, doc_name, {"source": pdf_path})

        for i, (chunk, embedding, extraction) in enumerate(
            zip(chunks, embeddings, merged_extractions)
        ):
            chunk_id = make_chunk_id(doc_name, i)
            session.execute_write(ingest_chunk, doc_name, chunk_id, chunk, embedding, i)
            session.execute_write(ingest_entities_and_relations, chunk_id, extraction)
            log.info("  Ingested chunk %d/%d", i + 1, len(chunks))

    # ---- Step 6: Post-processing ------------------------------------------
    log.info("[Step 6/7] Running entity deduplication…")
    deduplicate_entities(driver)

    # ---- Summary ----------------------------------------------------------
    with driver.session() as session:
        result = session.run(
            """
            MATCH (d:Document) WITH count(d) AS docs
            MATCH (c:Chunk)    WITH docs, count(c) AS chunks
            MATCH (e:Entity)   WITH docs, chunks, count(e) AS entities
            RETURN docs, chunks, entities
            """
        )
        record = result.single()
        if record:
            log.info("Final counts — Documents: %d  Chunks: %d  Entities: %d",
                     record["docs"], record["chunks"], record["entities"])

    driver.close()

    # ---- Step 7: Benchmark ------------------------------------------------
    log.info("[Step 7/7] Generating benchmark comparison…")
    benchmark.print_report()
    benchmark.save_to_file("llm_benchmark_phase2.json")

    phase1_data = load_phase1_benchmark("llm_benchmark_phase1.json")
    print_comparison(benchmark.summary(), phase1_data)

    print(f"\n{separator}")
    print("  ✅  DUAL-EXTRACTION PIPELINE COMPLETE — Graph loaded in Neo4j!")
    print(f"{separator}\n")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python graphrag_pipeline_v2.py <path_to_pdf>")
        print("  Dual-extraction pipeline (SpaCy + LLM) → Neo4j knowledge graph.")
        print()
        print("Setup:")
        print("  pip install spacy")
        print("  python -m spacy download en_core_web_sm")
        sys.exit(1)

    pdf_file = sys.argv[1]
    if not os.path.isfile(pdf_file):
        print(f"Error: file not found — {pdf_file}")
        sys.exit(1)

    run_pipeline_v2(pdf_file)
