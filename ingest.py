"""
Stage 1: Load a PDF and split it into chunks.
Stage 2: Embed those chunks and store them in a Pinecone index.

Why chunk at all? An embedding model turns text into a single vector that
represents its *overall* meaning. If we embedded the whole PDF as one chunk,
the vector would be a blurry average of everything in the document, and
retrieval would basically always return "the whole document" instead of the
specific paragraph that answers a question. Splitting into smaller chunks
lets each vector represent one focused idea, so we can retrieve just the
relevant piece later.

Stage 1 (loading + splitting) is free. Stage 2 (embedding + upserting) is
the first step in this project that spends real OpenAI money — a fraction
of a cent for a document this size, since embeddings only bill for input
tokens at $0.02 / 1M tokens.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSION,
)

PDF_PATH = "22-promptengg.pdf"


def load_and_split(pdf_path: str):
    # PyPDFLoader reads the PDF and returns one LangChain Document per page.
    # Each Document has .page_content (the text) and .metadata (e.g. page number).
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # RecursiveCharacterTextSplitter breaks that text into smaller pieces.
    # chunk_size = max characters per chunk (roughly proportional to tokens,
    #   which is what OpenAI actually bills you for).
    # chunk_overlap = characters repeated between consecutive chunks, so an
    #   idea/sentence that falls near a chunk boundary isn't lost entirely
    #   in either chunk.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    return chunks


def ensure_index_exists(pc: Pinecone, index_name: str, dimension: int):
    """Create the Pinecone index if it doesn't already exist.

    Checking first makes this safe to call every time you run ingestion —
    without it, re-running the script would try to create the same index
    twice and error out.

    dimension MUST match the embedding model's output size exactly
    (text-embedding-3-small -> 1536). A mismatch here is the most common
    beginner Pinecone error.
    """
    existing_names = [index["name"] for index in pc.list_indexes()]
    if index_name in existing_names:
        print(f"Pinecone index '{index_name}' already exists — reusing it.")
        return

    print(f"Creating Pinecone index '{index_name}' (dimension={dimension})...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        # ServerlessSpec = Pinecone's pay-as-you-go hosting, cheapest option
        # for a small personal project like this one.
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


def embed_and_store(chunks):
    """Turn each chunk's text into a vector and upload it to Pinecone."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index_exists(pc, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION)

    # This app only ever answers questions about ONE document at a time.
    # from_documents() below always assigns fresh random IDs to each vector,
    # so without clearing first, every new upload would just add its vectors
    # on top of whatever's already there — old and new documents' chunks
    # would mix together (or the same document would end up duplicated if
    # re-ingested). Wiping the index here keeps it holding exactly one
    # document's worth of vectors, matching how the app is actually used.
    #
    # Guard: deleting from a namespace with zero vectors raises a "not
    # found" error on Pinecone serverless, so only clear if there's
    # something there (e.g. skip this on the very first ever ingestion).
    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    if stats.get("total_vector_count", 0) > 0:
        index.delete(delete_all=True)

    # OpenAIEmbeddings sends text to OpenAI and gets back a list of numbers
    # (the vector) that represents that text's meaning.
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # from_documents does two things in one call: embeds every chunk, then
    # upserts (inserts/updates) the resulting vectors into the named index.
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )
    print(f"Stored {len(chunks)} chunks in Pinecone index '{PINECONE_INDEX_NAME}'.")


if __name__ == "__main__":
    chunks = load_and_split(PDF_PATH)
    print(f"Loaded and split into {len(chunks)} chunks.\n")

    embed_and_store(chunks)
