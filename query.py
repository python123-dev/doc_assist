"""
Stage 3: Retrieval — given a question, fetch the most relevant chunks
from the Pinecone index we filled in Stage 2.
Stage 4: Feed those chunks + the question to a cheap chat model and get
back an actual answer.

We deliberately keep retrieval and generation as two separate, visible
steps (rather than one black-box "RAG chain" function) so you can check
retrieval quality on its own. If the LLM's answer is ever wrong or vague,
printing the retrieved chunks tells you immediately whether the problem is
"retrieval found the wrong text" or "the LLM misread good text" — two very
different bugs.

Cost note: embedding a question is tiny (a handful of tokens). Answering
with gpt-5-nano costs based on how much context we stuff into the prompt
(the retrieved chunks) plus the length of the answer — still a fraction of
a cent per question for a small document like this.
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone

from config import (
    EMBEDDING_MODEL,
    CHAT_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    INITIAL_RETRIEVAL_K,
    RERANK_MODEL,
    RERANK_TOP_N,
)

# The prompt template is the instruction we send to the LLM alongside the
# retrieved context. Telling it to answer ONLY from the context (and admit
# when it can't) is what keeps a RAG app from "hallucinating" answers that
# aren't actually in your document.
PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.
If the answer isn't contained in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {question}

Answer:"""
)


def get_vector_store():
    """Reconnect to the existing Pinecone index (no re-embedding of the
    document happens here — we're just pointing at what Stage 2 already
    built)."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)


def retrieve_chunks(question: str, k: int = INITIAL_RETRIEVAL_K):
    """Embed the question and return the k most similar chunks stored in
    Pinecone, ranked by vector similarity (closest meaning first).

    We now overfetch a larger candidate pool (k=INITIAL_RETRIEVAL_K) than we
    actually want to send to the LLM, because vector similarity alone is an
    imperfect relevance signal — rerank_chunks() below narrows this pool
    down to the truly best matches."""
    vector_store = get_vector_store()
    return vector_store.similarity_search(question, k=k)


def rerank_chunks(question: str, chunks, top_n: int = RERANK_TOP_N):
    """Re-score candidate chunks against the question using Pinecone's
    hosted reranker, and keep only the top_n most relevant ones.

    Why this helps: similarity_search ranks by embedding distance, which is
    a decent but imperfect proxy for "actually answers this question." A
    reranker looks at the question and each chunk's actual text together
    and scores relevance more precisely — too slow/expensive to run over an
    entire index, but cheap over the small candidate set we already have.

    Gotcha: the rerank API returns each result's .index (its position in
    the `documents` list we sent) and .score, NOT the original chunk
    object. We map that index back into our own `chunks` list so we keep
    the LangChain Document's metadata (like page number) intact.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)

    texts = [chunk.page_content for chunk in chunks]

    result = pc.inference.rerank(
        model=RERANK_MODEL,
        query=question,
        documents=texts,
        top_n=top_n,
    )

    return [chunks[item.index] for item in result.data]


def generate_answer(question: str, chunks) -> str:
    """Join the retrieved chunks into one text block, plug it + the
    question into the prompt template, and ask the chat model to answer."""
    context = "\n\n".join(chunk.page_content for chunk in chunks)

    llm = ChatOpenAI(model=CHAT_MODEL)

    # The "|" pipes the formatted prompt straight into the LLM. This is
    # LangChain's LCEL syntax: prompt.invoke(...) then llm.invoke(...) in
    # one step, so the data flow reads left-to-right just like it happens.
    chain = PROMPT_TEMPLATE | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content


def answer_question(question: str, k: int = INITIAL_RETRIEVAL_K, top_n: int = RERANK_TOP_N) -> str:
    """The full pipeline: overfetch candidates, rerank down to the best
    few, then generate an answer from just those."""
    candidates = retrieve_chunks(question, k=k)
    chunks = rerank_chunks(question, candidates, top_n=top_n)
    return generate_answer(question, chunks)


if __name__ == "__main__":
    question = "What is prompt engineering?"

    candidates = retrieve_chunks(question)
    print(f"Retrieved {len(candidates)} candidates (before rerank):")
    for i, chunk in enumerate(candidates):
        print(f"  {i}. page {chunk.metadata.get('page')}: {chunk.page_content[:80]!r}")

    reranked = rerank_chunks(question, candidates)
    print(f"\nTop {len(reranked)} after reranking:")
    for chunk in reranked:
        print(f"  page {chunk.metadata.get('page')}: {chunk.page_content[:80]!r}")

    answer = generate_answer(question, reranked)
    print(f"\nQuestion: {question}\n")
    print(f"Answer: {answer}")
