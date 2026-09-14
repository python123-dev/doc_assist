# Document Assistant (RAG)

A simple Retrieval-Augmented Generation (RAG) app: upload a PDF, ask questions about it, get answers grounded in the document's actual content.

Built step by step with LangChain, OpenAI, and Pinecone, using the cheapest suitable models to keep API costs near-zero for personal/learning use.

## How it works

```
PDF  --load & chunk-->  text chunks
                              |
                              v
                    embed (OpenAI) & store (Pinecone)
                              |
Question  --embed (OpenAI)--> similarity search (Pinecone)
                              |
                  overfetched candidate chunks
                              |
                              v
                  rerank (Pinecone hosted reranker)
                              |
                    top-n most relevant chunks
                              |
                              v
              prompt (context + question) --> chat model (OpenAI) --> Answer
```

- **Chunking** breaks the PDF into small, focused pieces so each one embeds a single idea instead of a blurry average of the whole document.
- **Embeddings** turn text into vectors (lists of numbers) that capture meaning, so "similar meaning" can be found by vector distance.
- **Pinecone** stores those vectors and does the similarity search.
- **Reranking** takes a larger pool of candidate chunks (found by vector similarity) and re-scores them against the actual question text, since vector distance alone is an imperfect relevance signal. Only the top few survivors get sent to the LLM.
- **The chat model** reads only the reranked chunks (not the whole PDF) and answers from them — this keeps cost low and reduces made-up answers.

## Project files

| File | Purpose |
|---|---|
| `config.py` | Loads `.env` and holds all shared settings (model names, chunk size, index name, etc.) |
| `ingest.py` | Loads a PDF, splits it into chunks, embeds them, and stores them in Pinecone |
| `query.py` | Retrieves relevant chunks for a question and generates an answer with the chat model |
| `agent.py` | A tool-using agent that decides between searching the document or the live web |
| `app.py` | Streamlit UI: upload a PDF, ask questions, see answers |
| `requirements.txt` | Python package dependencies |
| `.env` | API keys (OpenAI, Pinecone) — **never commit this file** |

## Models used (chosen for low cost)

| Purpose | Model | Cost |
|---|---|---|
| Embeddings | `text-embedding-3-small` | $0.02 / 1M tokens |
| Reranking | `bge-reranker-v2-m3` (Pinecone hosted) | $2.00 / 1,000 rerank units (1 unit ≈ 1 query reranked against up to 100 short chunks) |
| Answering | `gpt-5-nano` | $0.05 / 1M input tokens, $0.40 / 1M output tokens |
| Agent tool-routing | `gpt-5-mini` | $0.25 / 1M input tokens, $2.00 / 1M output tokens (agent mode only) |

For a typical small PDF and a handful of questions, total cost is a fraction of a cent to a few cents. Embedding is a one-time cost per document; reranking and answering are billed per question, and reranking a small candidate set costs a small fraction of one rerank unit.

## Setup

1. Make sure `.env` in the project root has:
   ```
   OPENAI_API_KEY=...
   PINECONE_API_KEY=...
   ```
2. Create/activate a virtual environment and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Running it

### Option A: Command line (useful for testing/debugging each stage)

```
python config.py    # confirms your API keys load correctly
python ingest.py    # loads 22-promptengg.pdf, chunks it, embeds it, stores it in Pinecone
python query.py     # asks a sample question and prints the answer
```

### Option B: Web app

```
streamlit run app.py
```

Then open the URL it prints (usually http://localhost:8501), upload a PDF, and ask questions in the text box.

## Agent mode (optional web search fallback)

Checking **"Allow web search (agent mode)"** in the app switches from the plain RAG pipeline to a genuine tool-using agent (`agent.py`). Given a question, the agent itself — not hardcoded logic — decides whether to:

1. Call `search_document` (our existing retrieve + rerank, returning raw relevant chunk text), and/or
2. Call `search_web` (a live web search via Tavily, using the `TAVILY_API_KEY` already in `.env`)

It's built with `create_agent` (LangChain's tool-calling agent loop): the LLM reads each tool's description, calls what it needs, and keeps going until it can give a final answer — which always states whether it came from your document or the web. Try it from the terminal:

```
python agent.py
```

This asks one question answerable from the PDF and one clearly outside it, printing which tool got called for each so you can see the routing decision happen.

Agent mode uses a different (slightly stronger, still cheap) model, `gpt-5-mini`, than the plain RAG path — see `AGENT_MODEL` in `config.py` — since deciding *whether and which* tool to use is a harder judgment call than just answering from given context.

## Using a different PDF

- **Command line:** change `PDF_PATH` at the top of `ingest.py`, then re-run `python ingest.py`.
- **Web app:** just upload a different file — `app.py` handles chunking/embedding/storing automatically and won't re-process a file it's already indexed in the same session.

## Adjusting cost vs. quality

All the relevant knobs live in `config.py`:

- `CHUNK_SIZE` / `CHUNK_OVERLAP` — bigger chunks = more context per chunk but higher embedding/answering cost.
- `INITIAL_RETRIEVAL_K` — how many candidate chunks are fetched by vector similarity before reranking. More candidates give the reranker a better pool to choose from, at a small extra Pinecone read cost.
- `RERANK_TOP_N` — how many chunks survive reranking and actually get sent to the LLM. Fewer = cheaper and faster, but may miss relevant context.
- `RERANK_MODEL` — Pinecone's hosted reranker in use. `pinecone-rerank-v0` is a same-price alternative worth trying.
- `CHAT_MODEL` — swap `gpt-5-nano` for `gpt-5-mini` if answers feel too shallow (higher cost, better quality).
- `AGENT_MODEL` — model used only by agent mode's tool-routing decisions (separate from `CHAT_MODEL` so the plain RAG path's cost is unaffected).

## Notes

- `.env` contains real secret keys — it's covered by `.gitignore` so it won't be committed if you turn this into a git repo. Never share or paste its contents anywhere.
- Pinecone index name (`doc-assist`, set in `config.py`) is created automatically the first time you run ingestion, and reused after that.
