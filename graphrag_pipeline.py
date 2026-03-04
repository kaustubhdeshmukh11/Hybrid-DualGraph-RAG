"""
Hybrid DualGraph RAG — Neo4j Ingestion Pipeline
=================================================
Phase 1: LLM-based extraction (Groq) + HuggingFace embeddings + Neo4j storage.

Usage:
    python graphrag_pipeline.py <path_to_pdf>

Architecture (Integrated Approach):
    Lexical Graph:  (Document) --[HAS_CHUNK]--> (Chunk)
    Domain Graph:   (Entity)   --[RELATIONSHIP]--> (Entity)
    Connection:     (Chunk)    --[HAS_ENTITY]--> (Entity)

Chunk nodes store their text AND vector embedding as properties in Neo4j.
"""

import os
import re
import sys
import json
import time
import hashlib
import logging
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# LLM USAGE BENCHMARKING
# =============================================================================

class LLMBenchmark:
    """
    Tracks LLM API usage across the pipeline for Phase 1 vs Phase 2 comparison.

    Metrics captured per call:
        - prompt_tokens, completion_tokens, total_tokens
        - latency (seconds)
    """

    def __init__(self):
        self.calls: list[dict] = []

    def record(self, prompt_tokens: int, completion_tokens: int, latency: float, chunk_index: int):
        self.calls.append({
            "chunk_index": chunk_index,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_s": round(latency, 3),
        })

    @property
    def total_prompt_tokens(self) -> int:
        return sum(c["prompt_tokens"] for c in self.calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c["completion_tokens"] for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c["total_tokens"] for c in self.calls)

    @property
    def total_latency(self) -> float:
        return sum(c["latency_s"] for c in self.calls)

    @property
    def num_calls(self) -> int:
        return len(self.calls)

    def summary(self) -> dict:
        return {
            "llm_calls": self.num_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_latency_s": round(self.total_latency, 2),
            "avg_latency_per_chunk_s": round(self.total_latency / max(self.num_calls, 1), 2),
            "avg_tokens_per_chunk": round(self.total_tokens / max(self.num_calls, 1)),
        }

    def print_report(self):
        s = self.summary()
        print("\n" + "=" * 65)
        print("  📊  LLM BENCHMARK REPORT  (Phase 1 — LLM-Only Extraction)")
        print("=" * 65)
        print(f"  LLM calls made           : {s['llm_calls']}")
        print(f"  Total prompt tokens       : {s['total_prompt_tokens']:,}")
        print(f"  Total completion tokens   : {s['total_completion_tokens']:,}")
        print(f"  Total tokens              : {s['total_tokens']:,}")
        print(f"  Total LLM latency         : {s['total_latency_s']:.2f}s")
        print(f"  Avg latency / chunk       : {s['avg_latency_per_chunk_s']:.2f}s")
        print(f"  Avg tokens / chunk        : {s['avg_tokens_per_chunk']}")
        print("=" * 65)

    def save_to_file(self, path: str = "llm_benchmark_phase1.json"):
        """Persist the benchmark data to disk for later comparison."""
        import json as _json
        data = {"summary": self.summary(), "per_chunk": self.calls}
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(data, f, indent=2)
        log.info("Benchmark saved to %s", path)


# Global benchmark instance
benchmark = LLMBenchmark()


# =============================================================================
# 1.  CONFIGURATION & NEO4J CONNECTION
# =============================================================================

def load_neo4j_credentials(path: str = "neo4j.txt") -> dict:
    """
    Read Neo4j credentials from a key=value text file.

    Expected keys: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
    """
    creds: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                creds[key.strip()] = value.strip()
    log.info("Loaded Neo4j credentials from %s", path)
    return creds


def get_neo4j_driver(creds: dict):
    """Return an authenticated Neo4j driver instance."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        creds["NEO4J_URI"],
        auth=(creds["NEO4J_USERNAME"], creds["NEO4J_PASSWORD"]),
    )
    driver.verify_connectivity()
    log.info("Connected to Neo4j at %s", creds["NEO4J_URI"])
    return driver


def setup_neo4j_schema(driver):
    """
    Create uniqueness constraints and indexes in Neo4j.
    Idempotent — safe to call multiple times.
    """
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk)    REQUIRE c.id   IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity)   REQUIRE e.name IS UNIQUE",
    ]
    with driver.session() as session:
        for cypher in constraints:
            session.run(cypher)
    log.info("Neo4j schema constraints ensured.")


# =============================================================================
# 2.  PDF PARSING & CHUNKING
# =============================================================================

def load_pdf(filepath: str) -> str:
    """Extract all text from a PDF file using PyPDF2."""
    from PyPDF2 import PdfReader

    reader = PdfReader(filepath)
    pages_text: list[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages_text.append(text)
    full_text = "\n".join(pages_text)
    log.info("Loaded PDF: %s  (%d pages, %d chars)", filepath, len(reader.pages), len(full_text))
    return full_text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks of roughly `chunk_size` characters.

    Returns a list of chunk strings.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    log.info("Created %d chunks (size=%d, overlap=%d)", len(chunks), chunk_size, overlap)
    return chunks


# =============================================================================
# 3.  EMBEDDING GENERATION (HuggingFace / sentence-transformers)
# =============================================================================

_EMBEDDING_MODEL = None  # module-level cache


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load (and cache) the sentence-transformer embedding model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer(model_name)
        log.info("Loaded embedding model: %s", model_name)
    return _EMBEDDING_MODEL


def generate_embeddings(chunks: list[str], model=None) -> list[list[float]]:
    """
    Generate vector embeddings for a list of text chunks.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if model is None:
        model = get_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    log.info("Generated embeddings: %d vectors of dimension %d", len(embeddings), len(embeddings[0]))
    return [emb.tolist() for emb in embeddings]


# =============================================================================
# 4.  LLM ENTITY & RELATIONSHIP EXTRACTION  (Phase 1 — Groq)
# =============================================================================

load_dotenv()
_GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

EXTRACTION_PROMPT = """You are an expert knowledge-graph builder.
Given the following text chunk, extract ALL meaningful entities and relationships.

Return ONLY a valid JSON object with this exact structure (no markdown, no explanation):
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
- Use SCREAMING_SNAKE_CASE for relationship types (e.g., WORKS_AT, LOCATED_IN, IS_A).
- Extract as many meaningful triples as possible.
- Every entity mentioned in a relationship MUST also appear in the entities list.

TEXT:
{text}
"""


def extract_entities_llm(
    chunk_text: str,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    chunk_index: int = -1,
) -> dict:
    """
    Use Groq LLM to extract entities and relationships from a text chunk.

    Returns:
        dict with keys "entities" (list of dicts) and "relationships" (list of triples).
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
                {"role": "user", "content": EXTRACTION_PROMPT.format(text=chunk_text)},
            ],
            temperature=0.1,
            max_tokens=4000,
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

        # Attempt direct JSON parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Fallback: extract from markdown code fence
        if "```" in raw:
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        # Fallback: find first JSON object
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


# ---------------------------------------------------------------------------
# Phase 2 placeholder — Dependency-based extraction (future scope)
# ---------------------------------------------------------------------------
# def extract_with_spacy(chunk: str) -> dict:
#     """
#     Phase 2: Use spaCy NLP pipeline (dependency parsing, NER) to extract
#     entities and relationships as a complementary extraction path.
#
#     This will create a dual-extraction architecture:
#       - Path A: LLM-based  (extract_entities_llm)   — broad, semantic
#       - Path B: NLP-based  (extract_with_spacy)      — precise, syntactic
#
#     The results from both paths will be merged and deduplicated before
#     ingestion into Neo4j.
#
#     Returns:
#         dict with keys "entities" and "relationships"
#     """
#     raise NotImplementedError("Phase 2 — spaCy extraction not yet implemented.")


# =============================================================================
# 5.  NEO4J INGESTION (Cypher MERGE queries)
# =============================================================================

def ingest_document(tx, doc_name: str, metadata: dict | None = None):
    """MERGE a Document node."""
    props = metadata or {}
    tx.run(
        """
        MERGE (d:Document {name: $name})
        SET d += $props
        """,
        name=doc_name,
        props=props,
    )


def ingest_chunk(tx, doc_name: str, chunk_id: str, text: str, embedding: list[float], chunk_index: int):
    """
    MERGE a Chunk node with its text and embedding, and link it to its Document.
    """
    tx.run(
        """
        MATCH (d:Document {name: $doc_name})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text       = $text,
            c.embedding  = $embedding,
            c.index      = $chunk_index
        MERGE (d)-[:HAS_CHUNK]->(c)
        """,
        doc_name=doc_name,
        chunk_id=chunk_id,
        text=text,
        embedding=embedding,
        chunk_index=chunk_index,
    )


def ingest_entities_and_relations(tx, chunk_id: str, extraction: dict):
    """
    MERGE Entity nodes extracted from a chunk, create inter-entity
    relationships, and create HAS_ENTITY edges from the Chunk.
    """
    # --- Create entity nodes and HAS_ENTITY edges from the chunk -----------
    for entity in extraction.get("entities", []):
        ename = entity["name"].strip()
        etype = entity.get("type", "CONCEPT").upper()
        tx.run(
            """
            MERGE (e:Entity {name: $name})
            SET e.type = $type
            WITH e
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (c)-[:HAS_ENTITY]->(e)
            """,
            name=ename,
            type=etype,
            chunk_id=chunk_id,
        )

    # --- Create inter-entity relationships (dynamic types via APOC) --------
    for rel in extraction.get("relationships", []):
        head = rel.get("head", "").strip()
        tail = rel.get("tail", "").strip()
        relation = rel.get("relation", "RELATED_TO").strip().upper()
        # Sanitize: Neo4j rel types must be valid identifiers
        relation = re.sub(r"[^A-Z0-9_]", "_", relation)
        if not head or not tail or not relation:
            continue
        # Use apoc.merge.relationship for DYNAMIC relationship types
        tx.run(
            """
            MERGE (h:Entity {name: $head})
            MERGE (t:Entity {name: $tail})
            WITH h, t
            CALL apoc.merge.relationship(h, $relation, {}, {}, t, {}) YIELD rel
            RETURN rel
            """,
            head=head,
            tail=tail,
            relation=relation,
        )


# =============================================================================
# 6.  POST-PROCESSING — Entity Deduplication
# =============================================================================

def deduplicate_entities(driver):
    """
    Merge Entity nodes whose names are identical when lowercased.

    Strategy:
    1. Find groups of entities sharing the same lowercase name.
    2. For each group, keep the first node and rewire all relationships
       from duplicates to the keeper using APOC.
    3. Delete the duplicate nodes.
    """
    # Step A: Rewire all relationships from duplicates to keeper
    rewire_cypher = """
    MATCH (e:Entity)
    WITH toLower(e.name) AS lname, collect(e) AS nodes
    WHERE size(nodes) > 1
    WITH head(nodes) AS keeper, tail(nodes) AS duplicates
    UNWIND duplicates AS dup

    // Rewire outgoing relationships (dup)-[r]->(x)  →  (keeper)-[r]->(x)
    CALL {
        WITH dup, keeper
        MATCH (dup)-[r]->(x)
        WHERE x <> keeper
        WITH keeper, x, type(r) AS rType, properties(r) AS rProps
        CALL apoc.merge.relationship(keeper, rType, rProps, {}, x, {}) YIELD rel
        RETURN count(rel) AS rewired_out
    }

    // Rewire incoming relationships (x)-[r]->(dup)  →  (x)-[r]->(keeper)
    CALL {
        WITH dup, keeper
        MATCH (x)-[r]->(dup)
        WHERE x <> keeper
        WITH keeper, x, type(r) AS rType, properties(r) AS rProps
        CALL apoc.merge.relationship(x, rType, rProps, {}, keeper, {}) YIELD rel
        RETURN count(rel) AS rewired_in
    }

    // Delete the duplicate
    DETACH DELETE dup
    RETURN count(dup) AS deleted
    """
    with driver.session() as session:
        result = session.run(rewire_cypher)
        summary = result.consume()
        log.info(
            "Entity deduplication complete. Deleted %d duplicate nodes.",
            summary.counters.nodes_deleted,
        )


# =============================================================================
# 7.  MAIN PIPELINE ORCHESTRATOR
# =============================================================================

def make_chunk_id(doc_name: str, index: int) -> str:
    """Generate a deterministic chunk ID from the document name and index."""
    raw = f"{doc_name}::chunk_{index}"
    return hashlib.md5(raw.encode()).hexdigest()


def run_pipeline(pdf_path: str):
    """
    End-to-end GraphRAG ingestion pipeline.

    Steps:
        1. Connect to Neo4j & ensure schema
        2. Parse PDF → text chunks
        3. Generate embeddings for each chunk
        4. Extract entities & relationships via LLM
        5. Ingest everything into Neo4j
        6. Deduplicate entities
    """
    separator = "=" * 65
    print(f"\n{separator}")
    print("  GRAPHRAG NEO4J INGESTION PIPELINE")
    print(f"{separator}\n")

    # ---- Step 1: Neo4j connection -----------------------------------------
    log.info("[Step 1/6] Connecting to Neo4j…")
    creds = load_neo4j_credentials("neo4j.txt")
    driver = get_neo4j_driver(creds)
    setup_neo4j_schema(driver)

    # ---- Step 2: PDF parsing & chunking -----------------------------------
    log.info("[Step 2/6] Parsing PDF and creating chunks…")
    full_text = load_pdf(pdf_path)
    chunks = chunk_text(full_text, chunk_size=1000, overlap=200)
    doc_name = Path(pdf_path).stem

    # ---- Step 3: Embedding generation -------------------------------------
    log.info("[Step 3/6] Generating embeddings for %d chunks…", len(chunks))
    model = get_embedding_model()
    embeddings = generate_embeddings(chunks, model)

    # ---- Step 4: LLM entity extraction ------------------------------------
    log.info("[Step 4/6] Extracting entities and relationships via LLM…")
    extractions: list[dict] = []
    for i, chunk in enumerate(chunks):
        log.info("  Chunk %d/%d …", i + 1, len(chunks))
        extraction = extract_entities_llm(chunk, chunk_index=i)
        extractions.append(extraction)
        n_ent = len(extraction.get("entities", []))
        n_rel = len(extraction.get("relationships", []))
        log.info("    → %d entities, %d relationships", n_ent, n_rel)

    # ---- Step 5: Neo4j ingestion ------------------------------------------
    log.info("[Step 5/6] Ingesting data into Neo4j…")
    with driver.session() as session:
        # Create Document node
        session.execute_write(ingest_document, doc_name, {"source": pdf_path})

        # Create Chunk nodes + Entity nodes + relationships
        for i, (chunk, embedding, extraction) in enumerate(
            zip(chunks, embeddings, extractions)
        ):
            chunk_id = make_chunk_id(doc_name, i)
            session.execute_write(ingest_chunk, doc_name, chunk_id, chunk, embedding, i)
            session.execute_write(ingest_entities_and_relations, chunk_id, extraction)
            log.info("  Ingested chunk %d/%d", i + 1, len(chunks))

    # ---- Step 6: Post-processing ------------------------------------------
    log.info("[Step 6/6] Running entity deduplication…")
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

    # ---- Benchmark report -------------------------------------------------
    benchmark.print_report()
    benchmark.save_to_file("llm_benchmark_phase1.json")

    print(f"\n{separator}")
    print("  ✅  PIPELINE COMPLETE — Graph loaded in Neo4j!")
    print(f"{separator}\n")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python graphrag_pipeline.py <path_to_pdf>")
        print("  Ingests a PDF into Neo4j as a connected knowledge graph.")
        sys.exit(1)

    pdf_file = sys.argv[1]
    if not os.path.isfile(pdf_file):
        print(f"Error: file not found — {pdf_file}")
        sys.exit(1)

    run_pipeline(pdf_file)
