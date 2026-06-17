# HippoRAG — Hermes Integration Analysis

**Date:** 2026-06-17  
**Repo analyzed:** [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)  
**Papers:** v1 (NeurIPS 2024) — `2405.14831`, v2 (ICML 2025) — `2502.14802`

---

## Overview

HippoRAG is a **neurobiologically-inspired Retrieval-Augmented Generation** framework that combines LLMs, Knowledge Graphs, and Personalized PageRank (PPR) for multi-hop retrieval and sense-making. Published at NeurIPS 2024 (v1) and ICML 2025 (v2).

Core analogy: mimics human hippocampal indexing — LLM acts as neocortex (perception), KG as hippocampus (indexing), PPR as memory retrieval (spreading activation).

| Feature | HippoRAG 1 | HippoRAG 2 |
|---|---|---|
| Venue | NeurIPS 2024 | ICML 2025 |
| Key innovation | LLM + KG + PPR | + deeper passage integration, dynamic LLM at query time |
| Improvement | 20% over SOTA, 10-30x cheaper vs IRCoT | +7% associativity, no factual degradation |
| PyPI | `pip install hipporag==1.0` | `pip install hipporag==2.0.0a4` |

---

## Architecture

```
src/hipporag/
├── __init__.py
├── HippoRAG.py           # Main class — index(), retrieve(), rag_qa(), delete()
├── StandardRAG.py        # DPR-based variant (no graph)
├── embedding_store.py    # Parquet-backed vector store (chunks, entities, facts)
├── embedding_model/
│   ├── base.py
│   └── ...               # NV-Embed-v2, GritLM, Contriever, OpenAI-compatible
├── evaluation/
├── information_extraction/   # OpenIE module (online vLLM, offline VLLM, Transformers)
├── llm/
│   ├── base.py
│   ├── openai_gpt.py
│   ├── bedrock_llm.py
│   └── ...
├── prompts/
│   ├── linking.py
│   ├── prompt_template_manager.py
│   └── dspy_prompts/         # DSPy reranking filters
└── utils/
    ├── config_utils.py
    └── misc_utils.py
```

---

## Key API Surface

### Instantiation

```python
from hipporag import HippoRAG

# Simplest: use OpenAI-compatible API
hipporag = HippoRAG(
    save_dir='outputs',                          # Persists graph on disk
    llm_model_name='gpt-4o-mini',
    embedding_model_name='nvidia/NV-Embed-v2'
)

# Local via vLLM
hipporag = HippoRAG(
    save_dir='outputs',
    llm_model_name='meta-llama/Llama-3.3-70B-Instruct',
    llm_base_url='http://localhost:8000/v1',
    embedding_model_name='nvidia/NV-Embed-v2'
)
```

### Core Methods

| Method | Purpose |
|---|---|
| `index(docs)` | Build KG + embeddings from documents |
| `retrieve(queries)` | Just retrieve relevant passages |
| `rag_qa(queries)` | Retrieve + answer (end-to-end QA) |
| `delete()` | Remove documents from graph |
| `add_new_nodes/edges()` | Incremental graph updates |

### Config (BaseConfig)

| Parameter | Default | Purpose |
|---|---|---|
| `retrieval_top_k` | 200 | Docs retrieved before reranking |
| `linking_top_k` | 5 | Entities linked per passage |
| `max_qa_steps` | 3 | Multi-hop reasoning steps |
| `qa_top_k` | 5 | Final answer passages |
| `graph_type` | — | KG build strategy |
| `openie_mode` | online | online/offline IE extraction |
| `force_index_from_scratch` | false | Rebuild on each run |
| `force_openie_from_scratch` | false | Re-extract facts on each run |

---

## Integration Options for Hermes

### Option 1 — Drop-in RAG Skill (no GPU needed)

Since HippoRAG supports OpenAI-compatible endpoints, it can be pointed at any Hermes-managed LLM provider (OpenRouter, local vLLM). Install in a dedicated venv, wrap as a Hermes skill. Adds **graph-based multi-hop retrieval** to the existing pipeline.

```bash
python3 -m venv ~/venvs/hipporag
source ~/venvs/hipporag/bin/activate
pip install hipporag==2.0.0a4
```

#### Sample Hermes Skill

```python
# ~/.hermes/skills/hipporag-rag/main.py
from hipporag import HippoRAG

class HippoRAGSkill:
    def __init__(self, save_dir='/home/sak/hipporag_data',
                 llm_model='gpt-4o-mini',
                 llm_base_url=None,  # Use Hermes provider endpoint
                 embedding_model='text-embedding-3-small'):
        self.rag = HippoRAG(
            save_dir=save_dir,
            llm_model_name=llm_model,
            llm_base_url=llm_base_url,
            embedding_model_name=embedding_model
        )

    def index(self, docs):
        return self.rag.index(docs)

    def query(self, question):
        return self.rag.rag_qa(question)

    def retrieve_context(self, question):
        return self.rag.retrieve(question)
```

### Option 2 — Upgrade soffos-rag (existing Qdrant setup)

Your current `soffos-rag` skill is vector-only (Qdrant). HippoRAG adds:
- **Entity linking** via OpenIE
- **Graph traversal** (PPR) for associativity
- **Multi-hop reasoning** across documents
- **Incremental updates** without full re-index

Replace/parallel the vector store for tasks needing cross-document reasoning.

### Option 3 — Cron-driven Memory Ingestion

Set up a cron job that watches a data source and calls `hipporag.index(new_docs)` on new content. The KG persists on disk in `save_dir` — incremental updates via `add_new_nodes/edges()` keep it current without full rebuilds.

### Option 4 — Hermes Agent Associative Memory

Use `hipporag.retrieve()` to pre-fetch context before agent runs. The PPR algorithm surfaces **connected facts** — giving the agent "associative memory" that pure vector search can't provide. Feed returned passages into Hermes' system prompt or context.

---

## Cost & Performance Trade-offs

| Capability | soffos-rag (Qdrant) | HippoRAG |
|---|---|---|
| Vector similarity | ✅ | ✅ |
| Multi-hop retrieval | ❌ | ✅ (PPR on KG) |
| Entity linking | ❌ | ✅ (OpenIE) |
| Fact extraction | ❌ | ✅ |
| Incremental updates | ❌ | ✅ (add/remove nodes+edges) |
| Cost efficiency | ✅ | ✅ (single-pass retrieval) |
| Associative memory | ❌ | ✅ (+7% over SOTA) |
| Continual learning | ❌ | ✅ (non-parametric) |

**Cost:** HippoRAG is 10-30x cheaper than iterative methods like IRCoT for multi-hop QA. Single-pass LLM call for indexing + single-pass at retrieval.

---

## Requirements & Constraints

| Requirement | Status | Notes |
|---|---|---|
| Python ≥3.10 | ✅ | Check Hermes env |
| GPU for local models | ❌ Optional | Use API endpoints to avoid GPU needs |
| `torch==2.5.1`, `transformers==4.45.2` | ⚠️ Heavy | Dedicated venv recommended |
| `vllm==0.6.6` | ❌ Only for local | Skip if using external APIs |
| API key for embedding/LLM | ✅ | OpenRouter / OpenAI / Azure |
| Disk for parquet store | ✅ Minimal | `save_dir` holds parquet files |

**Heads-up:** Package pins exact versions for heavy deps. Dedicated venv strongly recommended to avoid conflicts with other Hermes skills.

---

## Quick Start Test

```python
from hipporag import HippoRAG

hipporag = HippoRAG(
    save_dir='./hipporag_test',
    llm_model_name='gpt-4o-mini',
    embedding_model_name='text-embedding-3-small'
)

docs = [
    "Hippocampus is a brain region essential for memory formation.",
    "The neocortex is involved in higher-order brain functions.",
    "Personalized PageRank spreads activation through graph nodes."
]
hipporag.index(docs)

result = hipporag.rag_qa("What brain regions are involved in memory?")
print(result)
```

---

## Links

- **Repository:** https://github.com/OSU-NLP-Group/HippoRAG
- **PyPI:** https://pypi.org/project/hipporag/
- **Paper v1:** https://arxiv.org/abs/2405.14831
- **Paper v2:** https://arxiv.org/abs/2502.14802
