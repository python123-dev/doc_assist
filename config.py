"""
Central place for settings shared across the project.

Why a separate config file? So model names, chunk sizes, etc. live in ONE
place. Later, when we want to tune cost/quality, we only change values here
instead of hunting through every script.
"""

import os
from dotenv import load_dotenv

# load_dotenv() reads the .env file in this folder and copies its key=value
# pairs into the environment, so os.getenv() below can see them.
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# --- Model choices (kept cheap on purpose) ---
EMBEDDING_MODEL = "text-embedding-3-small"   # $0.02 / 1M tokens
CHAT_MODEL = "gpt-5-nano"                    # $0.05 / 1M input, $0.40 / 1M output

# Used only by the agent (agent.py), not the plain RAG path above. Deciding
# WHETHER and WHICH tool to call is a harder judgment call than "answer from
# the given context," so the agent gets a slightly stronger (still cheap)
# model. This keeps the plain RAG path's cost exactly as it was.
AGENT_MODEL = "gpt-5-mini"                   # $0.25 / 1M input, $2.00 / 1M output

# --- Chunking settings (used in Stage 1) ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Pinecone settings (used from Stage 2 onward) ---
PINECONE_INDEX_NAME = "doc-assist"
EMBEDDING_DIMENSION = 1536   # must match EMBEDDING_MODEL's output size

# --- Retrieval settings (used from Stage 3 onward) ---
INITIAL_RETRIEVAL_K = 15  # how many candidate chunks to fetch by vector similarity

# --- Reranking settings ---
# We overfetch INITIAL_RETRIEVAL_K candidates above, then use Pinecone's
# hosted reranker to re-score them against the question and keep only the
# best RERANK_TOP_N. This is cheap (reranking short text is a fraction of
# a cent) and improves relevance vs. trusting raw vector distance alone.
RERANK_MODEL = "bge-reranker-v2-m3"   # Pinecone hosted rerank model, $2 / 1k rerank units
RERANK_TOP_N = 4  # how many chunks survive reranking and get sent to the LLM


if __name__ == "__main__":
    # Sanity check: run "python config.py" to confirm your keys loaded,
    # WITHOUT ever printing the actual secret values.
    print("OPENAI_API_KEY loaded:", bool(OPENAI_API_KEY))
    print("PINECONE_API_KEY loaded:", bool(PINECONE_API_KEY))
