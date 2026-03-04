# How Everything Connects — GraphRAG Architecture Explained

## The Big Picture

Imagine you have a PDF document full of knowledge. Our pipeline does three things with it:

1. **Breaks it into chunks** (small text pieces)
2. **Understands what each chunk talks about** (entities and relationships)
3. **Stores everything in a connected database** (Neo4j) where you can traverse from text → concepts → related concepts → back to text

The magic is that **the vector embeddings, the text chunks, AND the knowledge graph all live together in Neo4j** — not in separate databases. This is called the **Integrated Approach**.

---

## The Three Layers

### Layer 1: Lexical Graph (Document Structure)

This is the "filing cabinet". It tells you **where text lives**.

```
(Document: "sample1") ──HAS_CHUNK──▶ (Chunk 0: "In this paper we...")
                       ──HAS_CHUNK──▶ (Chunk 1: "The model uses...")
                       ──HAS_CHUNK──▶ (Chunk 2: "Results show...")
                       ...
```

Each **Chunk node** stores:
- `text` — the actual 1000-character text snippet
- `embedding` — a vector of 384 numbers (from the `all-MiniLM-L6-v2` model)
- `index` — its position in the document

> **Why embeddings on chunks?** When a user asks a question, we convert their question into a vector too, then find the chunks whose vectors are most similar. This is the "vector search" part — it happens right inside Neo4j, no separate vector database needed.

---

### Layer 2: Domain Graph (Knowledge)

This is the "brain". It captures **what the document knows**.

```
(Entity: "Transformer") ──BASED_ON──▶ (Entity: "Attention Mechanism")
(Entity: "BERT")         ──IS_A────▶ (Entity: "Language Model")
(Entity: "Google")       ──CREATED──▶ (Entity: "BERT")
```

Entities are things like people, organizations, technologies, concepts. Relationships are the meaningful connections between them — and in Phase 2, these come from **two sources**:

| Source | What it finds | Example |
|---|---|---|
| **SpaCy** (dependency parser) | Direct subject-verb-object facts stated in a single sentence | "Google **developed** BERT" |
| **LLM** (Groq) | Cross-sentence, implicit, hierarchical relationships | "BERT IS_A Language Model" (inferred, not stated as SVO) |

---

### Layer 3: The Connection (HAS_ENTITY)

This is the **bridge** that makes GraphRAG powerful. It links **every chunk to the entities mentioned in it**.

```
(Chunk 5: "Google developed BERT using transformer architecture...")
    ──HAS_ENTITY──▶ (Entity: "Google")
    ──HAS_ENTITY──▶ (Entity: "BERT")
    ──HAS_ENTITY──▶ (Entity: "Transformer")
```

### Why does this bridge matter?

Without it, you have two disconnected systems: a pile of text chunks and a separate knowledge graph. The `HAS_ENTITY` edge is what lets you do things like:

1. **Start from a question** → vector search finds Chunk 5
2. **Jump to the graph** → Chunk 5 `HAS_ENTITY` → "BERT"
3. **Traverse the graph** → "BERT" `IS_A` → "Language Model", "Google" `CREATED` → "BERT"
4. **Jump back to text** → "Language Model" `HAS_ENTITY` ← Chunk 12 (which has more context about language models)

This is the **RAG + Graph** combination. Regular RAG only does step 1. GraphRAG does all four — it follows connections in the knowledge graph to find related chunks that vector search alone would miss.

---

## Visual Summary

```
┌─────────────────────────────────────────────────────────┐
│                      NEO4J DATABASE                     │
│                                                         │
│  ┌──────────┐                                           │
│  │ Document │──HAS_CHUNK──▶┌───────────────────┐        │
│  │"sample1" │              │ Chunk 0           │        │
│  └──────────┘              │ text: "Google..." │        │
│       │                    │ embedding: [0.1,  │        │
│       │                    │  0.3, -0.2, ...]  │        │
│       ▼                    └────────┬──────────┘        │
│  ┌──────────┐                      │                    │
│  │ Chunk 1  │               HAS_ENTITY                  │
│  │ text:... │                      │                    │
│  │ emb:[..] │                      ▼                    │
│  └──────────┘              ┌──────────────┐             │
│       │                    │Entity: Google│──CREATED──▶ │
│       ▼                    └──────────────┘    ┌──────┐ │
│     ...                                        │ BERT │ │
│                                                └──────┘ │
│  LEXICAL GRAPH              DOMAIN GRAPH                │
│  (structure)         ◀──HAS_ENTITY──▶ (knowledge)       │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 2 — LLM Usage for Graph Creation

Every time we send a chunk to the LLM, the API reports how many **tokens** it consumed. A token ≈ ¾ of a word.

| Metric | What it means |
|---|---|
| **LLM calls** | Number of Groq API calls (= number of chunks) |
| **Prompt tokens** | Tokens sent to the LLM (chunk text + instructions) |
| **Completion tokens** | Tokens the LLM generated back (entities/relationships JSON) |
| **Total tokens** | Prompt + Completion — what counts against rate limits / cost |
| **Latency** | Wall-clock time per LLM call |

### Phase 2 Benchmark (Dual Extraction — SpaCy + LLM)

| Metric | Value |
|---|---|
| LLM calls | 31 |
| Total prompt tokens | 18,838 |
| Total completion tokens | 15,294 |
| **Total tokens** | **34,132** |
| Total LLM latency | 47.80s |
| Avg latency / chunk | 1.54s |
| Avg tokens / chunk | 1,101 |

> In Phase 2, SpaCy handles inline Subject-Verb-Object triples locally (zero LLM cost). The LLM is guided to extract **only** cross-sentence, implicit, and hierarchical relationships — producing **22% fewer completion tokens** than Phase 1 while SpaCy fills in the syntactic triples for free.

Phase 1 benchmark data is saved separately in `llm_benchmark_phase1.json`.
