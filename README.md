# Agentic RAG

Agentic RAG built with **LangGraph** over two blog posts, combining three techniques in
one graph.

| Technique | Does | Nodes |
|---|---|---|
| **Adaptive RAG** | Decides if retrieval is needed | `classify_node` → `answer_direct_node` |
| **Corrective RAG** | Grades evidence, re-searches if insufficient | `grade_node` → `refine_node` |
| **Self-RAG** | Verifies the answer, repairs or abstains | `check_answer_node` → `fix_answer_node` / `give_up_node` |

## Flow

```mermaid
flowchart TD
    START([START]) --> reset[reset_node]
    reset --> classify{classify_node}

    classify -->|out_of_scope| direct[answer_direct_node]
    direct --> END1([END])

    classify -->|in_scope| research[research_node]
    research -->|tool calls| tools[tools_node]
    tools -->|results| research
    research -->|no searches left| grade{grade_node}

    grade -->|not covered| refine[refine_node]
    refine -->|retry with a new query| research
    grade -->|covered| write[write_answer_node]

    write --> check{check_answer_node}
    check -->|pass| END2([END])
    check -->|fail, repairs left| fix[fix_answer_node]
    fix -->|rewrite| write
    check -->|fail, no repairs left| giveup[give_up_node]
    giveup --> END3([END])

    classDef adaptive fill:#e8ecff,stroke:#4f46e5,color:#1e1b4b
    classDef corrective fill:#fff4e0,stroke:#ea580c,color:#431407
    classDef selfrag fill:#e9f7ec,stroke:#16a34a,color:#14532d
    classDef shared fill:#f1f5f9,stroke:#94a3b8,color:#0f172a
    classDef terminal fill:#f5f3ff,stroke:#7c3aed,color:#2e1065

    class classify,direct adaptive
    class grade,refine corrective
    class check,fix,giveup selfrag
    class reset,research,tools,write shared
    class START,END1,END2,END3 terminal
```

## How it works

1. **Classify** — in scope? which post? Out-of-scope questions get a short refusal, no retrieval.
2. **Research** — ReAct loop over `retrieve_relevant_document`: Pinecone `k=20` (optional
   `blog_post` filter), then Cohere `rerank-v3.5` → top 5. Max **4 searches** per round.
3. **Grade** — critic checks coverage; if not covered, names what's missing and proposes a
   better query. Max **2 corrective rounds**.
4. **Write** — answer only from chunks, using the author's wording for definitions.
5. **Check** — verifier checks every claim against the chunks. Max **2 repairs**, then
   declines instead of guessing.

| Limit | Value |
|---|---|
| `MAX_SEARCHES` | 4 per round |
| `MAX_CORRECTIVE_ATTEMPTS` | 2 |
| `MAX_REPAIRS` | 2 |

Worst case: 3 rounds × 4 searches = 12 searches.

## Evaluation with DeepEval

`evals/` scores every graph run with [DeepEval](https://github.com/confident-ai/deepeval).

### Goldens — `evals/goldens.json`

20 cases, each with an `input` and `expected_output`:

| Bucket | Cases |
|---|---|
| Answerable — `harness-engineering` | 6 |
| Answerable — `diffusion-models` | 7 |
| Refusal — detail absent from corpus | 5 |
| Refusal — false cross-post premise | 1 |
| Refusal — out of scope | 1 |

Refusal cases use `"DECLINE"` as their `expected_output`.

### Metrics

Built from graph state (`answer` → `actual_output`, `chunks` → `retrieval_context`),
judged by `gpt-4o-mini`:

| Metric | Threshold | Catches |
|---|---|---|
| Faithfulness | 0.80 | claims the chunks do not support |
| Answer Relevancy | 0.70 | answers that dodge the question |
| Contextual Recall | 0.70 | missing evidence in retrieval |
| Contextual Precision | 0.60 | relevant chunks ranked below noise |
| Contextual Relevancy | 0.50 | off-topic retrieval |

Faithfulness and Answer Relevancy test generation; the Contextual metrics test retrieval.

### Running

```bash
uv run python -m evals.test_rag
```

`-m` is required so `from graph import graph` resolves. The runner prints score, cost and
failure reason per metric, and currently scores only the first 4 goldens
(`dataset.goldens[0:4]`) — drop the slice for all 20. It is a script, not a pytest
module, so `deepeval test run` collects nothing.

### Reading results

| Symptom | Fix |
|---|---|
| Faithfulness low | `WRITE_ANSWER_SYSTEM` |
| Answer Relevancy low | `WRITE_ANSWER_SYSTEM`, `CLASSIFY_SYSTEM` |
| Contextual Recall low | chunking, `k`, rerank `top_n`, `refine_node` |
| Contextual Precision low | `k`, metadata filter, duplicate chunks |
| Contextual Relevancy low | query wording, `blog_post` filter |

Each case costs one graph run plus five judge calls; per-metric cost is printed.

## Corpus

`ingestion.py`: scrape → chunk `1000/200` → embed → Pinecone (1536-dim, cosine,
`aws/us-east-1`).

- **harness-engineering** — *Harness Engineering for Self-Improvement* (Lilian Weng, 2026)
- **diffusion-models** — *Diffusion Models for Video Generation* (Lilian Weng, 2024)

## Stack

Python 3.12 · LangGraph · LangChain · Pinecone · Cohere `rerank-v3.5` · DeepEval ·
OpenAI `gpt-4o-mini` / `text-embedding-3-small` · uv

## Layout

| File | Role |
|---|---|
| `graph.py` | Compiles the `StateGraph` |
| `state.py` | `AgenticGraphState` (14 keys) |
| `nodes.py` | 11 nodes + structured-output schemas |
| `edges.py` | Routing functions and loop limits |
| `prompts.py` | All system prompts |
| `tools.py` | Pinecone search (`k=20`) + Cohere rerank (`top_n=5`) |
| `ingestion.py` | Builds the index |
| `llm_model.py` | Shared `gpt-4o-mini` client |
| `evals/goldens.json` | 20 golden cases |
| `evals/test_rag.py` | DeepEval runner |
| `docs/graph_annotated.png` | Architecture diagram |

## Quick start

```bash
uv sync
cp .env.example .env             # add keys
uv run python ingestion.py       # build the index
uv run langgraph dev             # LangGraph Studio
uv run python -m evals.test_rag  # DeepEval
```

From Python:

```python
from graph import graph

print(graph.invoke({"question": "What are the three components of ACE?"})["answer"])
```

## Config

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Chat + embeddings |
| `PINECONE_API_KEY` | Vector store |
| `INDEX_NAME` | Index name (default `agentic-rag-index`) |
| `COHERE_API_KEY` | Reranking |
| `LANGSMITH_*` | Optional tracing |

> Re-running `ingestion.py` deletes the index contents.
