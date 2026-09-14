"""
Agent: a genuine tool-using agent, not a hardcoded if/else.

The difference between this and everything in query.py: in query.py, WE
decide the order of operations (retrieve, then rerank, then generate) in
plain Python. Here, the LLM itself decides — on every question — which
tool(s) to call, by reading each tool's docstring/description and a system
prompt. create_agent() runs that decision loop (a "ReAct" agent): call the
LLM, see if it wants to use a tool, run the tool, feed the result back to
the LLM, repeat until it gives a final answer.

Why this specific agent: it can answer questions the plain RAG pipeline
can't — anything outside the uploaded document — by falling back to a live
web search, while still preferring the document when possible (cheaper,
and grounded in what you actually uploaded).
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch

from config import AGENT_MODEL
from query import retrieve_chunks, rerank_chunks


@tool
def search_document(question: str) -> str:
    """Search the uploaded document for text relevant to the question.
    Use this FIRST for any question that might be answered by the uploaded PDF."""
    candidates = retrieve_chunks(question)
    chunks = rerank_chunks(question, candidates)
    return "\n\n".join(
        f"(page {c.metadata.get('page')}) {c.page_content}" for c in chunks
    )


# TavilySearch reads TAVILY_API_KEY from the environment automatically —
# no need to pass it in. Giving it a clear name/description matters: the
# agent's LLM reads this description to decide when to reach for it.
search_web = TavilySearch(
    max_results=3,
    name="search_web",
    description=(
        "Search the live web. Use this ONLY if search_document did not "
        "contain enough information to answer the question."
    ),
)

AGENT_SYSTEM_PROMPT = (
    "You answer questions using the provided tools. Always try search_document "
    "first. Only call search_web if search_document's results don't answer the "
    "question. Your final answer must state whether it came from the document "
    "or the web."
)

# create_agent wires the model + tools + system prompt into a ready-to-run
# ReAct loop. Nothing here tells it WHEN to use which tool beyond the
# descriptions/prompt above — that decision happens inside the LLM itself.
agent = create_agent(
    model=f"openai:{AGENT_MODEL}",
    tools=[search_document, search_web],
    system_prompt=AGENT_SYSTEM_PROMPT,
)


def run_agent(question: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    # One question the document can answer, one it clearly can't — so you
    # can SEE the agent make two different routing decisions, not just
    # trust that it did.
    demo_questions = [
        "What is prompt engineering?",
        "Who won the most recent Ballon d'Or?",
    ]

    for question in demo_questions:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})

        print(f"\nQuestion: {question}")
        for msg in result["messages"]:
            if getattr(msg, "tool_calls", None):
                print("  Tool(s) called:", [tc["name"] for tc in msg.tool_calls])
        print("Answer:", result["messages"][-1].content)
