# From Digital Twins to Autonomous Local Agents: Rethinking AI for Smart Water Networks

> **IEEE Internet of Things Journal** — Special Issue on Effective, Efficient, and Trustworthy AI Agents for the Internet of Things

This repository contains the code for the paper *"From Digital Twins to Autonomous Local Agents: Rethinking AI for Smart Water Networks"*, accepted in the **IEEE Internet of Things Journal**.

The system is a fully local, multi-agent LLM pipeline that automatically generates, executes, self-repairs, and evaluates [WNTR](https://usepa.github.io/WNTR/) (Water Network Tool for Resilience) Python code from natural language task descriptions. It is designed to run entirely on local hardware without relying on external cloud APIs.

---

## Overview

The agent operates in two modes:

| Mode | Description |
|------|-------------|
| **RAG** | Retrieves relevant WNTR documentation from a local vector store to augment code generation |
| **0-shot** | Generates code directly from the LLM's parametric knowledge, without retrieval |

Both pipelines share a common self-healing loop: generated code is executed inside a sandboxed Docker container, and any runtime error triggers an automated diagnosis-and-fix cycle (up to 5 iterations). A final QA validator assesses whether the produced code correctly addresses the original user request.

---

## Agent Pipeline

### RAG Mode

```
User Prompt
    │
    ▼
[Query Generator]          ← Decomposes prompt into targeted search queries
    │
    ▼
[Document Retriever]       ← Parent-document retriever over ChromaDB (WNTR docs)
    │
    ▼
[Context Distiller]        ← Filters & summarizes retrieved chunks per query
    │
    ▼
[Context Synthesizer]      ← Merges all distilled contexts into a unified context
    │
    ▼
[Code Generator]           ← Generates imports + code from context + user request
    │
    ▼
[Code Executor]            ← Runs code in an isolated Docker sandbox
    │
    ├── success ──────────► [QA Validator]  ← Compliance check: FULL / PARTIAL / EXCESSIVE / FAILED
    │
    └── failure ──────────► [Error Diagnoser]
                                │
                                ├── generic_python ──────► [Code Healer]
                                │                               │
                                └── library_specific ──► [Doc Retriever] → [Code Healer]
                                                              (loop, up to 5 iterations)
```

### 0-shot Mode

The same pipeline without the retrieval stages — the LLM generates code from parametric knowledge alone, and self-repairs using diagnosis + fix nodes only.

---

## Benchmark

The system is evaluated on **WNTR coding tasks** organized by difficulty.

All tasks use the **Net3** network (`wdnets/net3.inp`) as the benchmark target.

---

## Water Networks

The repository includes 8 standard benchmark networks from the water distribution literature:

| Network | File |
|---------|------|
| Net1 | `wdnets/net1.inp` |
| Net3 | `wdnets/net3.inp` |
| Hanoi | `wdnets/hanoi.inp` |
| D-Town | `wdnets/dtown.inp` |
| C-Town | `wdnets/ctown.inp` |
| Modena | `wdnets/modena.inp` |
| Fossolo | `wdnets/fossolo.inp` |
| Twins | `wdnets/twins.inp` |

---

## Models Evaluated

The following locally-served models were benchmarked via [Ollama](https://ollama.com/):

- `hf.co/mistralai/Devstral-Small-2507_gguf:BF16`
- `qwen3-coder:30b-a3b-fp16`
- `phi4:14b-fp16`
- `gemma3:27b-it-fp16`

OpenAI-compatible providers are also supported via the `--model_provider` flag.

---

## Requirements

### System

- [Docker](https://www.docker.com/) (used to sandbox code execution)
- [Ollama](https://ollama.com/) (used to serve local LLMs and embedding models)
- Python 3.10+

### Python dependencies

Install via pip:

```bash
pip install langchain langgraph langchain-community langchain-ollama langchain-chroma \
            langchain-huggingface sentence-transformers docker \
            psutil pynvml torch numpy
```

The Docker sandbox image is built automatically on first run from `docker-env/`. It installs:

```
numpy, pandas, scikit-learn, requests, matplotlib, wntr
```

---

## Usage

### Basic run (all modes, all default LLMs)

```bash
python main.py
```

### Run a specific mode and model

```bash
python main.py --mode rag --llm qwen3-coder:30b-a3b-fp16
python main.py --mode 0-shot --llm phi4:14b-fp16
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | both | `rag` or `0-shot` |
| `--llm` | all defaults | LLM model name |
| `--model_provider` | `ollama` | `ollama` or `openai` |
| `--embedding_model` | `qwen3-embedding:8b-fp16` | Ollama embedding model for RAG |
| `--chunk_size` | `1500` | Parent chunk size for document splitting |
| `--chunk_overlap` | `300` | Chunk overlap for splitting |
| `--rag_k` | `4` | Number of documents to retrieve |
| `--temperature` | `0.0` | LLM temperature |
| `--top_p` | `1.0` | LLM top-p |
| `--db_rebuild` | `False` | Force rebuild of the vector database |
| `--reranker_model` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker model |
| `--reranker_top_n` | `3` | Documents to keep after reranking |
| `--save_graph_png` | `False` | Save LangGraph diagram as PNG |

---

## Output

Results are saved to `results/<llm_name>/<mode>/` as JSON files containing the full agent state, including:

- Generated code (all iterations)
- Execution traces and tracebacks
- Retrieved documents and distilled contexts
- Token usage per node
- Per-node inference times
- Hardware power metrics
- Final QA compliance report (`FULL` / `PARTIAL` / `EXCESSIVE` / `FAILED`)

---

## Power Profiler

Every LLM invocation is wrapped by `power_profiler.py`, which runs background monitoring threads to collect hardware metrics at 100 ms intervals. It supports:

- **Multi-GPU** power draw, utilization, and VRAM usage via [NVML](https://developer.nvidia.com/management-library-nvml) (`pynvml`)
- **CPU** per-process RAM usage and estimated power based on a configurable TDP value (`cpu_tdp` parameter in Watts)
- **Energy** in Joules and Wh computed via numerical integration over the collected power samples

Metrics are stored per-node in the agent state under `power_hardware_metrics` and serialized to the output JSON. To profile on a CPU-only machine, NVML initialization gracefully falls back with a warning.

> **Note:** Set `cpu_tdp` in `graph.py` → `invoke_with_metadata()` to match your processor's TDP for accurate CPU energy estimates.

---

## Project Structure

```
wdn-agents/
├── main.py              # Entry point and argument parsing
├── graph.py             # LangGraph pipeline definitions (RAG and 0-shot)
├── prompts.py           # All system prompts and benchmark task list
├── schemas.py           # Pydantic structured output schemas
├── utils.py             # LLM loading, RAG setup, Docker utilities, I/O helpers
├── power_profiler.py    # Hardware profiler (CPU + multi-GPU power, VRAM, utilization)
├── wdnets/              # EPANET .inp files for 8 benchmark networks
└── docker-env/          # Dockerfile and requirements for the code sandbox
    ├── Dockerfile
    ├── requirements.txt
    └── wdnets/          # Copy of network files available inside the container
```

---

## Citation

Citation to be released.

---

## License

This project is licensed under the [MIT License](LICENSE).