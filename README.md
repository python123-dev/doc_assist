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
                    top-k relevant chunks
                              |
                              v
              prompt (context + question) --> chat model (OpenAI) --> Answer
```

- **Chunking** breaks the PDF into small, focused pieces so each one embeds a single idea instead of a blurry average of the whole document.
- **Embeddings** turn text into vectors (lists of numbers) that capture meaning, so "similar meaning" can be found by vector distance.
- **Pinecone** stores those vectors and does the similarity search.
- **The chat model** reads only the retrieved chunks (not the whole PDF) and answers from them — this keeps cost low and reduces made-up answers.

## Project files

| File | Purpose |
|---|---|
| `config.py` | Loads `.env` and holds all shared settings (model names, chunk size, index name, etc.) |
| `ingest.py` | Loads a PDF, splits it into chunks, embeds them, and stores them in Pinecone |
| `query.py` | Retrieves relevant chunks for a question and generates an answer with the chat model |
| `app.py` | Streamlit UI: upload a PDF, ask questions, see answers |
| `requirements.txt` | Python package dependencies |
| `.env` | API keys (OpenAI, Pinecone) — **never commit this file** |

## Models used (chosen for low cost)

| Purpose | Model | Cost |
|---|---|---|
| Embeddings | `text-embedding-3-small` | $0.02 / 1M tokens |
| Answering | `gpt-5-nano` | $0.05 / 1M input tokens, $0.40 / 1M output tokens |

For a typical small PDF and a handful of questions, total cost is a fraction of a cent to a few cents. Embedding is a one-time cost per document; answering is billed per question.

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

## Using a different PDF

- **Command line:** change `PDF_PATH` at the top of `ingest.py`, then re-run `python ingest.py`.
- **Web app:** just upload a different file — `app.py` handles chunking/embedding/storing automatically and won't re-process a file it's already indexed in the same session.

## Adjusting cost vs. quality

All the relevant knobs live in `config.py`:

- `CHUNK_SIZE` / `CHUNK_OVERLAP` — bigger chunks = more context per chunk but higher embedding/answering cost.
- `RETRIEVAL_K` — how many chunks are retrieved per question. Fewer = cheaper and faster, but may miss relevant context.
- `CHAT_MODEL` — swap `gpt-5-nano` for `gpt-5-mini` if answers feel too shallow (higher cost, better quality).

## Notes

- `.env` contains real secret keys — it's covered by `.gitignore` so it won't be committed if you turn this into a git repo. Never share or paste its contents anywhere.
- Pinecone index name (`doc-assist`, set in `config.py`) is created automatically the first time you run ingestion, and reused after that.
