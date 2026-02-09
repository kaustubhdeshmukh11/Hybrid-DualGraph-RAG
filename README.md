# Hybrid-DualGraph-RAG

A comparison of Baseline RAG vs Graph RAG approaches for enhanced retrieval-augmented generation.

## Project Overview

This project demonstrates:
1. **Baseline RAG** - Traditional vector similarity-based retrieval
2. **Knowledge Graph** - Entity-based knowledge graph construction using LLM
3. **Graph RAG** (Coming Soon) - Graph-enhanced retrieval for better context

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
Create a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

## Usage

### Baseline RAG
```bash
python baseline_rag.py
```

### Knowledge Graph
```bash
python knowledge_graph.py
```

The knowledge graph:
- Extracts entities (PERSON, ORGANIZATION, LOCATION, CONCEPT, OBJECT, EVENT)
- Identifies relationships between entities
- Visualizes the graph with color-coded nodes

## Components

| File | Description |
|------|-------------|
| `baseline_rag.py` | Traditional RAG implementation using FAISS |
| `knowledge_graph.py` | Entity-based knowledge graph with NetworkX |
| `requirements.txt` | Python dependencies |

## Tech Stack

- **LLM**: Groq (Llama 4 Scout)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Graph**: NetworkX
- **Visualization**: Matplotlib

## License

MIT
