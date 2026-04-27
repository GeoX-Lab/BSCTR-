# Bidirectional Semantic Complementary Tool
Retrieval for Remote Sensing Agents

A tool retrieval agent framework that addresses **bidirectional semantic incompleteness** in tool retrieval — enriching semantics from both the query side (via task decomposition) and the tool side (via graph-based information aggregation).

---

## Core Idea

Standard dense retrieval suffers from semantic gaps in two directions:

1. **Query-side incompleteness**: A user query is often too short or ambiguous to match tool descriptions precisely. This is addressed by decomposing the query into structured sub-tasks with explicit `action` and `tool_search` fields, enriching the retrieval signal.

2. **Tool-side incompleteness**: A tool's description alone may not capture its relational context (e.g., which tools are typically used together). This is addressed by aggregating neighborhood information from a tool dependency graph using **Simplified Graph Convolution (SGC)**, blending raw embeddings with graph-propagated context.

---

## Architecture

```
User Query
    │
    ▼
[Task Decomposition]  ← LLM decomposes query into structured sub-tasks
    │                    (query-side semantic enrichment)
    ▼
[Tool Retrieval]      ← Retrieves candidate tools per sub-task
    │
    ├── Dense          raw embedding cosine similarity
    ├── BM25           keyword-based retrieval
    ├── SGC            graph-aggregated embedding retrieval  ← tool-side enrichment
    └── Graph Fusion   dense + trajectory graph DFS expansion
    │
    ▼
[RRF Fusion]          ← Reciprocal Rank Fusion merges multi-task candidates
    │
    ▼
[Tool Pool]           ← Top-ranked tools passed to the execution agent
    │
    ▼
[ReAct Execution]     ← Thought → Action → Observation loop
    │
    ▼
[Graph Update]        ← Successful tool chains update the dependency graph
```

---

## Retrieval Strategies

All strategies are in [Strategy/](Strategy/) and share the `BaseRetriever` interface.

| Strategy | File | Description |
|---|---|---|
| Dense | [Strategy/Dense.py](Strategy/Dense.py) | Cosine similarity over normalized embeddings |
| BM25 | [Strategy/BM25.py](Strategy/BM25.py) | Keyword-based retrieval via `rank_bm25` |
| SGC | [Strategy/SGCRetriever.py](Strategy/SGCRetriever.py) | Graph-convolution-enhanced embeddings |
| Graph Fusion | [Strategy/Graph.py](Strategy/Graph.py) | Dense retrieval + DFS expansion on trajectory graph |

### SGC Retriever

The SGC retriever enriches tool embeddings by aggregating neighbor information from the tool dependency graph:

```
z = (1 - α) · H_raw  +  α · Â^k · H_raw
```

where `Â` is the row-normalized adjacency matrix (with self-loops) and `α = 0.5` by default.

Three aggregation modes are supported:

- `f` (forward): parent tools aggregate into child tools
- `b` (backward): child tools aggregate into parent tools  
- `s` (symmetric): bidirectional aggregation (default in experiments)

### Graph Fusion Retriever

Combines dense retrieval with a data-driven tool graph built from historical trajectories. Initial dense candidates are expanded via DFS traversal, with a decay factor applied to dependency scores.

---

## Project Structure

```
AgentToolmem/
├── Agents/
│   ├── Base.py           # BaseAgent: LLM calling, tool execution, data persistence
│   ├── agent.py          # ToolAgent: task decomposition, tool pool building, re-planning
│   ├── run.py            # Wrapper: experiment entry point, mode-based retriever init
│   └── ReAct.py          # ReActAgent: Thought-Action-Observation loop
├── Strategy/
│   ├── Base.py           # Abstract BaseRetriever
│   ├── Dense.py          # Dense retriever
│   ├── BM25.py           # BM25 retriever
│   ├── SGCRetriever.py   # SGC graph-enhanced retriever
│   └── Graph.py          # Graph Fusion retriever
├── GraphManager.py       # Tool dependency graph (adjacency matrix, SGC adj)
├── Graph_RAG_Tool_Fusion.py  # NetworkX-based trajectory graph for Graph Fusion
├── Toolregistry.py       # Tool registration, schema inference, unified info API
├── Working_memory.py     # Agent execution state tracking
├── model.py              # LLM wrapper (OpenAI-compatible async API)
├── Prompt/               # Prompt templates (decompose, act, replan)
├── config.yaml           # LLM and embedding model configuration
└── evaluate/             # Evaluation scripts and benchmark results
```

---

## Experiment Modes

The `Wrapper` class in [Agents/run.py](Agents/run.py) supports the following modes:

| Mode | Planning | Retriever |
|---|---|---|
| `dense` | No | Dense |
| `bm25` | No | BM25 |
| `sgc` | No | SGC |
| `graph` | No | Graph Fusion |
| `plan` | Yes | Dense |
| `plan_sgc` | Yes | SGC |

**Planning** mode decomposes the query into sub-tasks before retrieval, using the structured `action` + `tool_search` fields as enriched query vectors.

---

## Key Components

### Task Decomposition (Query-Side Enrichment)

The LLM decomposes a user query into a list of structured steps:

```json
[
  {"step": 1, "action": "retrieve", "tool_search": "search satellite imagery", "query": "..."},
  {"step": 2, "action": "analyze", "tool_search": "compute NDVI index", "query": "..."}
]
```

Each step's `action + tool_search` is embedded and used as the retrieval query, providing richer semantic signal than the raw user query.

### RRF Fusion

When multiple sub-tasks each retrieve `top_k` candidates, results are merged using **Reciprocal Rank Fusion**:

```
score(tool) = Σ  1 / (k + rank_i)
```

where `k = 60` by default. This produces a unified, rank-stable tool pool.

### Trajectory-Based Graph Learning

After successful task execution, the ordered sequence of tools used is recorded as a trajectory edge chain in `GraphManager`. These edges inform future SGC aggregation, making the retriever improve with usage.

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for embedding inference)
- [Ollama](https://ollama.com) running locally for the embedding model

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd AgentToolmem
pip install -r requirements.txt
```

For PyTorch with CUDA support, install it separately first (adjust the CUDA version to match your driver):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### 2. Configure LLM backends

Edit [config.yaml](config.yaml) and fill in your API key and base URL:

```yaml
deepseek-v3.2:
  use_responses_api: true
  client_kwargs:
    api_key: "YOUR_API_KEY"
    base_url: "api_url"
  generation:
    max_tokens: 16000
    temperature: 0.0
    stream: true

ollama:
  embedding_url: "embeddings_url"
  model_name: "nomic-embed-text:latest"
  embedding_dim: 768
```

Any OpenAI-compatible endpoint works (DeepSeek, GPT-4o, Qwen, Gemini via proxy, etc.). Add additional model blocks following the same structure.

### 3. Install benchmark-specific dependencies (optional)

**geoplan-bench** requires geospatial libraries:

```bash
pip install rasterio scikit-image scipy
```

`rasterio` depends on GDAL. If the pip install fails, use conda:

```bash
conda install -c conda-forge rasterio
```

**API-Bank** has no extra dependencies beyond the base `requirements.txt`.

---

## Benchmarks & Evaluation

The framework is evaluated on three benchmarks. Each benchmark has its own `main.py` entry point that registers tools, loads questions, and runs the agent in a specified mode. Outputs are saved per-mode under `outputs/<model_name>/<mode>/`.

### Benchmarks Overview

| Benchmark | Domain | Tool Count | Test Split | GT Format |
|---|---|---|---|---|
| [geoplan-bench](geoplan-bench/) | Geospatial planning | 100+ | `data/test_tasks/*.json` | `data/gt_with_difficulty.json` |
| [API-Bank](API-Bank/) | General API calls | 50+ | `test-data/level-3-api.json` | `test-data/level-gt.json` |

---

### Step 1 — Run the Agent

Each benchmark's `main.py` accepts a `mode` argument. Example for geoplan-bench:

```bash
cd geoplan-bench
python main.py
```

Inside `main.py`, set the `mode` and `model_name` variables before running:

```python
modes = ["dense", "bm25", "sgc", "graph", "plan", "plan_sgc"]
model_name = "deepseek-v3.2"
```

The agent will:
1. Register all benchmark tools into `ToolRegistry`
2. Load test questions from the benchmark's test split
3. Initialize the retriever according to the selected mode (optionally loading trajectory data for SGC/Graph modes)
4. For each question, run the full ReAct loop (Decompose → Retrieve → Execute)
5. Save two output files per mode:
   - `outputs/<model>/<mode>/tools_pool.json` — the retrieved tool pool per question (used for retrieval evaluation)
   - `outputs/<model>/<mode>/tool_call.json` — the actual tools called during execution (used for execution evaluation)

---

### Step 2 — Evaluate Retrieval Quality

Script: [evaluate/evaluate_recall_and_percision.py](evaluate/evaluate_recall_and_percision.py)

This script measures how well the retrieved tool pool covers the ground-truth tools, **before** execution. It computes per-question Recall, Precision, F1, and NDCG@k, then aggregates by difficulty level (Simple / Medium / Complex).

**Metrics:**

| Metric | Description |
|---|---|
| Recall@k | Fraction of GT tools found in the top-k retrieved pool |
| Precision@k | Fraction of top-k retrieved tools that are in GT |
| F1 | Harmonic mean of Recall and NDCG |
| NDCG@k | Normalized Discounted Cumulative Gain — rewards ranking GT tools higher |

**Usage — evaluate a single baseline:**

```python
from evaluate.evaluate_recall_and_percision import calculate_retrieval_metrics

calculate_retrieval_metrics(
    gt_file_path="...",
    retrieved_file_path="...",
    top_k= ...
)
```

**Usage — compare all baselines in a folder:**

```python
from evaluate.evaluate_recall_and_percision import evaluate_baselines_in_folder

evaluate_baselines_in_folder(
    base_folder="...",
    gt_path="...",
    json_filename="...",
    top_k= ...
)
```

This iterates over every subdirectory (one per mode) and prints a unified comparison table across all baselines and difficulty levels.

**GT file format** (`gt_with_difficulty.json`):

```json
{
  "0": {"tools": ["ToolA", "ToolB"], "difficulty": "Simple"},
  "1": {"tools": ["ToolC", "ToolD", "ToolE"], "difficulty": "Complex"}
}
```

---

### Step 3 — Evaluate Execution Quality

Script: [evaluate/tools_call.py](evaluate/tools_call.py)

This script measures the quality of the actual tool call sequence produced by the agent during execution, comparing it against the ground-truth tool chain. It supports three complementary metrics:

| Metric | Description |
|---|---|
| Any Order | Set intersection recall — fraction of GT tools actually called, regardless of order |
| In Order | Subsequence match — fraction of GT tools called in the correct relative order |
| Levenshtein | Edit-distance similarity — `1 - distance / max_len`, tolerates insertions/deletions/substitutions |

**Usage — evaluate a single prediction file:**

```python
from evaluate.tools_call import evaluate_tools_by_difficulty

evaluate_tools_by_difficulty(
    gt_file="...",
    pred_file="..."
)
```

**Usage — compare all baselines in a folder:**

```python
from evaluate.tools_call import evaluate_baselines_by_folder

evaluate_baselines_by_folder(
    base_folder="...",
    gt_file="..."
)
```

Results are broken down by difficulty (Simple / Medium / Complex / Overall) and saved alongside the prediction file as `tool_call_metrics.json`.

**GT file format** (`level-gt.json`):

```json
{
  "0": {"tools": ["login", "search_product", "add_to_cart"], "difficulty": "Medium"},
  "1": {"tools": ["get_weather"], "difficulty": "Simple"}
}
```

---

### Benchmark-Specific Notes

**geoplan-bench**
- Trajectory data (`data/train_trajectory.json`) is pre-loaded for SGC and Graph Fusion modes to warm-start the tool dependency graph.
- GT file: `data/gt_with_difficulty.json` — includes difficulty labels derived from the number of required tools.
- Retrieval evaluation uses `top_k=15`; execution evaluation uses the full predicted call sequence.

**API-Bank**
- Tests level-3 tasks (multi-step, multi-API calls).
- Tools are loaded dynamically from `api_bank/apis/*.py` via `inspect.getmembers`.
- GT file: `test-data/level-gt.json`.
- No pre-built trajectory; SGC graph is initialized from scratch and updated online.
---