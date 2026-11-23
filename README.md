# Agentic RAG

An agentic retrieval-augmented generation system built with **LangGraph**, implementing
three RAG techniques in a single graph over a fixed corpus of two blog posts.

| Technique | What it does | Nodes |
|---|---|---|
| **Adaptive RAG** | Decides whether the question needs retrieval at all | `classify_node` → `answer_direct_node` |
| **Corrective RAG** | Judges whether the retrieved evidence is sufficient, and re-searches with a better query if not | `grade_node` → `refine_node` |
| **Self-RAG** | Verifies the final answer against the retrieved chunks, repairs it, or abstains | `check_answer_node` → `fix_answer_node` / `give_up_node` |

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

Every gate is a separate node, so each step is independently observable and
traceable via LangSmith.

## How it works

1. **Classify** — an LLM decides if the question is in scope and which post it targets
   (`harness-engineering` or `diffusion-models`). Out-of-scope questions skip retrieval
   entirely and get a short "outside the indexed posts" answer.
2. **Research** — a ReAct loop where the LLM calls `retrieve_relevant_document`
   (Pinecone semantic search, `k=5`, optional per-post metadata filter), up to
   **4 searches per round**.
3. **Grade** — a critic LLM checks whether the gathered evidence actually covers the
   question and, if not, names what is missing and proposes one better query.
   Up to **2 corrective rounds**, then it writes the best answer it can.
4. **Write** — the answer is composed strictly from the retrieved chunks, using the
   author's own wording for definitions.
5. **Check** — a verifier LLM confirms every claim in the answer is supported by the
   chunks (and catches claims credited to the wrong post). Up to **2 repairs**, then the
   system **declines** rather than guessing.

### Loop budgets

| Limit | Value | Resets when |
|---|---|---|
| `MAX_SEARCHES` | 4 | `refine_node` starts a new round |
| `MAX_CORRECTIVE_ATTEMPTS` | 2 | — |
| `MAX_REPAIRS` | 2 | — |

Worst case: 3 research rounds × 4 searches = 12 search turns.

## Corpus

Two posts, ingested by `ingestion.py` (scrape → chunk `1000/200` → embed → Pinecone,
1536-dim, cosine, `aws/us-east-1`):

- **harness-engineering** — *Harness Engineering for Self-Improvement* (Lilian Weng, 2026)
- **diffusion-models** — *Diffusion Models for Video Generation* (Lilian Weng, 2024)

## Stack

Python 3.12 · LangGraph · LangChain · Pinecone · OpenAI (`gpt-4o-mini` for
reasoning, `text-embedding-3-small` for retrieval) · uv

## Project layout

| File | Role |
|---|---|
| `graph.py` | Compiles the `StateGraph`; exported as `graph` for `langgraph.json` |
| `state.py` | `AgenticGraphState` — 15 keys, `messages` uses the `add_messages` reducer |
| `nodes.py` | All 11 nodes + Pydantic schemas for structured output |
| `edges.py` | The 4 routing functions and the loop limits |
| `prompts.py` | Every system prompt (`CLASSIFY`, `RESEARCH`, `GRADE`, `WRITE_ANSWER`, `CHECK_ANSWER`, `ANSWER_DIRECT`, `GIVE_UP_MESSAGE`) |
| `tools.py` | `retrieve_relevant_document` — Pinecone semantic search tool |
| `ingestion.py` | Standalone script that builds the index |
| `llm_model.py` | The shared `gpt-4o-mini` client |
| `docs/graph_annotated.png` | Annotated architecture diagram |

## Quick start

```bash
uv sync
cp .env.example .env        # add your keys
uv run python ingestion.py  # scrape, chunk and embed the two posts
uv run langgraph dev        # open LangGraph Studio
```

To run it from Python instead:

```python
from graph import graph

result = graph.invoke({"question": "What are the three components of ACE?"})
print(result["answer"])
```

## Configuration

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Chat completions and embeddings |
| `PINECONE_API_KEY` | Vector store |
| `INDEX_NAME` | Pinecone index name (default `agentic-rag-index`) |
| `LANGSMITH_*` | Optional tracing |

> `ingestion.py` **deletes the existing index contents** when re-run.
