"""Self-RAG LangGraph node functions — each function is one step in the workflow.

Workflow: Contextualize → Decide → Retrieve → Grade → Rewrite → Web Search
          → Generate → Support Check → Revise → Usefulness → Commit Memory
"""

import time
import logging
from typing import List, Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tavily import TavilyClient

from src.config import get_settings
from src.vectorstore import get_retriever
from src.llm import get_llm, get_fast_llm
from src.rag_state import (
    RAGState,
    BatchRelevanceDecision,
    SupportDecision,
    UsefulnessDecision,
    QueryRewrite,
)


logger = logging.getLogger(__name__)


# ─── Helper Functions ────────────────────────────────────────────────


def _trace(state: RAGState, item: str):
    """Append a trace entry to the workflow audit trail."""
    return [*(state.get("trace") or []), item]


def _format_context(docs: List[Document]) -> str:
    """Format retrieved documents into a labeled text block for the LLM prompt."""
    blocks = []

    # Web evidence: keep only the top 3 results
    # and cap each result to 1500 characters.
    if docs and all(
        (d.metadata or {}).get("source_type") == "web"
        for d in docs
    ):
        docs = docs[:3]

    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}

        if meta.get("source_type") == "web":
            head = (
                f"[WEB {i}] "
                f"{meta.get('title', '')} | "
                f"{meta.get('url', '')}"
            )
            content = d.page_content[:1500]
        else:
            head = (
                f"[INTERNAL {i}] "
                f"{meta.get('title') or meta.get('document_name') or meta.get('source', '')}"
            )

            if meta.get("page") is not None:
                head += f" | page {int(meta['page']) + 1}"

            # Internal documents remain completely unchanged.
            content = d.page_content

        blocks.append(f"{head}\n{content}")

    return "\n\n---\n\n".join(blocks)


def _memory_text(state: RAGState, limit: int = 4) -> str:
    """Return the last N conversation memory entries as text."""
    items = state.get("memory") or []

    return (
        "\n\n".join(items[-limit:])
        if items
        else "No previous conversation context."
    )


# ─── Node: Contextualize Question ───────────────────────────────────


def contextualize_question(state: RAGState):
    """Rewrite the user's question using conversation history if needed."""
    t0 = time.perf_counter()
    history = _memory_text(state)

    user_question = (
        state.get("user_question")
        or state.get("question", "")
    )

    if not state.get("memory"):
        dt = time.perf_counter() - t0
        return {
            "question": user_question,
            "trace": _trace(
                state,
                f"Memory: new incident session ({dt:.2f}s)"
            )
        }

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rewrite the newest user message as a standalone "
            "cloud-operations question using the previous "
            "conversation only when needed. Preserve service "
            "names, symptoms, errors, and constraints. If the "
            "message already stands alone, return it unchanged. "
            "Do not answer the question."
        ),
        (
            "human",
            "Previous conversation:\n{history}\n\n"
            "Newest message:\n{question}"
        ),
    ])

    chain = prompt | get_llm() | StrOutputParser()
    raw_question = chain.invoke(
        {
            "history": history,
            "question": user_question,
        }
    )
    question = raw_question.strip().strip("\"'")
    dt = time.perf_counter() - t0

    return {
        "question": question,
        "trace": _trace(
            state,
            f"Memory contextualized question: {question} ({dt:.2f}s)"
        )
    }


# ─── Node: Commit Memory ────────────────────────────────────────────


def commit_memory(state: RAGState):
    """Save the current Q&A turn into the conversation memory."""
    t0 = time.perf_counter()
    user_question = (
        state.get("user_question")
        or state.get("question", "")
    )

    answer = state.get("answer", "")
    route = state.get("source_mode", "none")

    entry = (
        f"User: {user_question}\n"
        f"Assistant ({route}): {answer}"
    )
    dt = time.perf_counter() - t0

    return {
        "memory": [entry],
        "trace": _trace(
            state,
            f"SQLite memory checkpoint updated ({dt:.2f}s)"
        )
    }


# ─── Node: Decide Retrieval ─────────────────────────────────────────


def decide_retrieval(state: RAGState):
    """Ask the LLM whether this question needs retrieval or can be answered directly."""
    t0 = time.perf_counter()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a retrieval router.

Decide whether the question needs retrieval.

Return exactly one word:
TRUE
or
FALSE

Return TRUE for:
- company-specific information
- internal policies
- runbooks
- production incidents
- troubleshooting
- deployment issues
- current or specific technical information
- questions requiring evidence

Return FALSE for:
- simple arithmetic
- basic definitions
- generic explanations
- casual/general questions that do not require evidence

Examples:

Question: what is 2+2
TRUE/FALSE: FALSE

Question: what is Python
TRUE/FALSE: FALSE

Question: Our checkout API is returning 502 errors after deployment.
TRUE/FALSE: TRUE

Question: What is our company's refund policy?
TRUE/FALSE: TRUE

Do not answer the question."""
        ),
        (
            "human",
            "Question: {question}"
        ),
    ])

    response = get_fast_llm().invoke(
        prompt.format_messages(
            question=state["question"]
        )
    )

    raw = response.content.strip().upper()

    should_retrieve = raw.startswith("TRUE")
    dt = time.perf_counter() - t0

    return {
        "need_retrieval": should_retrieve,
        "trace": _trace(
            state,
            f"Retrieval decision: {should_retrieve} ({dt:.2f}s)"
        )
    }


def route_after_decide(
    state: RAGState
) -> Literal["direct", "retrieve"]:
    """Router: send to direct answer or retrieval pipeline."""
    return (
        "retrieve"
        if state.get("need_retrieval", True)
        else "direct"
    )


# ─── Node: Generate Direct ──────────────────────────────────────────


def generate_direct(state: RAGState):
    """Answer from general knowledge without retrieval."""
    t0 = time.perf_counter()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer briefly from general technical knowledge only. "
            "Do not invent organization-specific infrastructure, "
            "runbooks, credentials, incident history, or deployment "
            "procedures."
        ),
        (
            "human",
            "{question}"
        ),
    ])

    ans = get_llm().invoke(
        prompt.format_messages(
            question=state["question"]
        )
    ).content
    dt = time.perf_counter() - t0

    return {
        "answer": ans,
        "source_mode": "direct",
        "trace": _trace(
            state,
            f"Generated direct answer ({dt:.2f}s)"
        )
    }


# ─── Node: Retrieve Internal ────────────────────────────────────────


def retrieve_internal(state: RAGState):
    """Fetch relevant chunks from the Pinecone vector store."""
    t0 = time.perf_counter()
    q = (
        state.get("retrieval_query")
        or state["question"]
    )

    docs = get_retriever().invoke(q)

    for d in docs:
        d.metadata = {
            **(d.metadata or {}),
            "source_type": "internal"
        }
    dt = time.perf_counter() - t0

    return {
        "docs": docs,
        "relevant_docs": [],
        "source_mode": "internal",
        "trace": _trace(
            state,
            f"Internal retrieval: {len(docs)} chunks ({dt:.2f}s)"
        )
    }


# ─── Node: Grade Relevance ──────────────────────────────────────────


def grade_relevance(state: RAGState):
    """Batch-grade all retrieved documents for relevance to the question."""
    t0 = time.perf_counter()
    docs = state.get("docs", [])
    mode = state.get("source_mode", "internal")

    if not docs:
        dt = time.perf_counter() - t0
        return {
            "relevant_docs": [],
            "trace": _trace(
                state,
                f"Relevance grade ({mode}, batched): 0/0 relevant ({dt:.2f}s)"
            )
        }

    formatted_docs = "\n\n".join(
        f"[DOC {i}]\n{d.page_content[:2000]}"
        for i, d in enumerate(docs, 1)
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Judge relevance at the topic/evidence level for each retrieved document. "
            "A document is relevant when it contains information useful for answering the user's question. "
            "Do not require the exact final answer. Be strict about unrelated content. "
            "Return exactly one relevance judgment for every document labeled [DOC i]."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Retrieved Documents:\n{documents}"
        ),
    ])

    grader = get_fast_llm().with_structured_output(
        BatchRelevanceDecision,
        method="function_calling"
    )

    try:
        decision = grader.invoke(
            prompt.format_messages(
                question=state["question"],
                documents=formatted_docs
            )
        )

        expected_indices = set(range(1, len(docs) + 1))
        returned_indices = [g.doc_index for g in decision.results]

        if (
            len(returned_indices) != len(docs)
            or set(returned_indices) != expected_indices
        ):
            raise ValueError(
                f"Invalid batch relevance indices: expected {sorted(expected_indices)}, "
                f"got {returned_indices}"
            )

        grade_map = {
            g.doc_index: g.is_relevant
            for g in decision.results
        }

        relevant = [
            d
            for i, d in enumerate(docs, 1)
            if grade_map.get(i, False)
        ]

        dt = time.perf_counter() - t0
        return {
            "relevant_docs": relevant,
            "trace": _trace(
                state,
                f"Relevance grade ({mode}, batched): "
                f"{len(relevant)}/{len(docs)} relevant ({dt:.2f}s)"
            )
        }

    except Exception as e:
        logger.warning("Batch relevance grading failed: %s", e)
        dt = time.perf_counter() - t0
        return {
            "relevant_docs": [],
            "trace": _trace(
                state,
                f"Batch relevance grading failed ({dt:.2f}s): {e}"
            )
        }


def route_after_relevance(
    state: RAGState
) -> Literal[
    "generate",
    "rewrite_internal",
    "rewrite_web",
    "no_answer"
]:
    """Router: decide what to do based on relevance grading results."""
    if state.get("relevant_docs"):
        return "generate"

    s = get_settings()

    if state.get("source_mode") == "web":

        if (
            state.get("web_rewrites", 0)
            < s.max_web_rewrites
        ):
            return "rewrite_web"

        return "no_answer"

    if (
        state.get("retrieval_rewrites", 0)
        < s.max_retrieval_rewrites
    ):
        return "rewrite_internal"

    return "rewrite_web"


# ─── Node: Rewrite Queries ──────────────────────────────────────────


def rewrite_internal_query(state: RAGState):
    """Rewrite the retrieval query to improve Pinecone search results."""
    t0 = time.perf_counter()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rewrite the operations question for semantic vector "
            "retrieval over internal cloud runbooks, SOPs, "
            "postmortems, architecture notes, and troubleshooting "
            "documents. Use 6-18 words, preserve service names "
            "and error symptoms, add useful operations keywords, "
            "remove filler, and do not answer."
        ),
        (
            "human",
            "Question: {question}\n"
            "Previous query: {previous}"
        ),
    ])

    out = get_llm().with_structured_output(
        QueryRewrite,
        method="function_calling"
    ).invoke(
        prompt.format_messages(
            question=state["question"],
            previous=state.get(
                "retrieval_query",
                ""
            )
        )
    )
    dt = time.perf_counter() - t0

    return {
        "retrieval_query": out.query,
        "retrieval_rewrites": (
            state.get("retrieval_rewrites", 0) + 1
        ),
        "docs": [],
        "relevant_docs": [],
        "trace": _trace(
            state,
            f"Rewrote internal query: {out.query} ({dt:.2f}s)"
        )
    }


def rewrite_web_query(state: RAGState):
    """Rewrite the question into a concise internet search query."""
    t0 = time.perf_counter()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rewrite the question into a concise internet "
            "search query of 6-14 words. Preserve important "
            "entities. Add recency wording when the question "
            "asks for latest/current/today. Do not answer."
        ),
        (
            "human",
            "Question: {question}\n"
            "Previous web query: {previous}"
        ),
    ])

    out = get_llm().with_structured_output(
        QueryRewrite,
        method="function_calling"
    ).invoke(
        prompt.format_messages(
            question=state["question"],
            previous=state.get(
                "web_query",
                ""
            )
        )
    )
    dt = time.perf_counter() - t0

    return {
        "web_query": out.query,
        "web_rewrites": (
            state.get("web_rewrites", 0) + 1
        ),
        "docs": [],
        "relevant_docs": [],
        "trace": _trace(
            state,
            f"Prepared internet search query: {out.query} ({dt:.2f}s)"
        )
    }


# ─── Node: Web Search ───────────────────────────────────────────────


def web_search(state: RAGState):
    """Search the internet via Tavily when internal evidence is insufficient."""
    t0 = time.perf_counter()
    s = get_settings()

    if not s.tavily_api_key:
        dt = time.perf_counter() - t0
        return {
            "docs": [],
            "source_mode": "web",
            "used_web_search": True,
            "trace": _trace(
                state,
                f"Internet search unavailable: TAVILY_API_KEY missing ({dt:.2f}s)"
            )
        }

    client = TavilyClient(
        api_key=s.tavily_api_key
    )

    q = (
        state.get("web_query")
        or state["question"]
    )

    response = client.search(
        query=q,
        search_depth="advanced",
        max_results=5,
        include_answer=False
    )

    docs = []

    for r in response.get("results", []):
        content = r.get("content", "")

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source_type": "web",
                    "source": r.get("url", ""),
                    "url": r.get("url", ""),
                    "title": r.get("title", "")
                },
            )
        )
    dt = time.perf_counter() - t0

    return {
        "docs": docs,
        "source_mode": "web",
        "used_web_search": True,
        "trace": _trace(
            state,
            f"Internet search: {len(docs)} results ({dt:.2f}s)"
        )
    }


# ─── Node: Generate from Context ────────────────────────────────────


def generate_from_context(state: RAGState):
    """Generate an answer grounded in the retrieved evidence."""
    t0 = time.perf_counter()
    context = _format_context(
        state.get("relevant_docs", [])
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are OpsGuard, an enterprise cloud "
            "operations and incident-response copilot. Answer "
            "using only the supplied evidence. Prefer private "
            "runbooks, SOPs, architecture notes, and postmortems "
            "when present. If the evidence comes from the web, "
            "clearly label it as external guidance and never "
            "present it as an organization-specific procedure. "
            "Do not invent infrastructure facts, credentials, "
            "commands, or incident history. Provide concise, "
            "actionable troubleshooting guidance and preserve "
            "any cautions contained in the evidence."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Evidence:\n{context}"
        ),
    ])

    ans = get_llm().invoke(
        prompt.format_messages(
            question=state["question"],
            context=context
        )
    ).content
    dt = time.perf_counter() - t0

    return {
        "answer": ans,
        "context": context,
        "support_retries": 0,
        "trace": _trace(
            state,
            f"Generated answer from "
            f"{state.get('source_mode', '')} evidence ({dt:.2f}s)"
        )
    }


# ─── Node: Support Check ────────────────────────────────────────────


def check_support(state: RAGState):
    """Verify whether the answer's claims are supported by the evidence."""
    t0 = time.perf_counter()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Verify whether every meaningful claim in the answer "
            "is supported by the supplied evidence. Return "
            "fully_supported only when all important claims are "
            "grounded; partially_supported when some claims are "
            "grounded but some are not; no_support when key claims "
            "are unsupported. Evidence excerpts should be short."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Answer:\n{answer}\n\n"
            "Evidence:\n{context}"
        ),
    ])

    out = get_llm().with_structured_output(
        SupportDecision,
        method="function_calling"
    ).invoke(
        prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", "")
        )
    )
    dt = time.perf_counter() - t0

    return {
        "support_status": out.status,
        "evidence": out.evidence,
        "trace": _trace(
            state,
            f"Support check: {out.status} ({dt:.2f}s)"
        )
    }


def route_after_support(
    state: RAGState
) -> Literal["usefulness", "revise"]:
    """Router: if fully supported go to usefulness, else revise."""
    if state.get("support_status") == "fully_supported":
        return "usefulness"

    if (
        state.get("support_retries", 0)
        >= get_settings().max_support_retries
    ):
        return "usefulness"

    return "revise"


# ─── Node: Revise Answer ────────────────────────────────────────────


def revise_answer(state: RAGState):
    """Rewrite the answer to remove unsupported claims."""
    t0 = time.perf_counter()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rewrite the answer so every factual claim is "
            "directly supported by the provided evidence. "
            "Remove unsupported interpretation and speculation. "
            "Still answer the question naturally; do not mention "
            "this verification process."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Current answer:\n{answer}\n\n"
            "Evidence:\n{context}"
        ),
    ])

    ans = get_llm().invoke(
        prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", "")
        )
    ).content
    dt = time.perf_counter() - t0

    return {
        "answer": ans,
        "support_retries": (
            state.get("support_retries", 0) + 1
        ),
        "trace": _trace(
            state,
            f"Revised answer for grounding ({dt:.2f}s)"
        )
    }


# ─── Node: Usefulness Check ─────────────────────────────────────────


def check_usefulness(state: RAGState):
    """Judge whether the answer actually addresses the user's question."""
    t0 = time.perf_counter()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Judge only whether the answer directly addresses "
            "the user's question. Do not re-grade factual "
            "grounding. Return useful or not_useful and a "
            "one-line reason."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Answer:\n{answer}"
        ),
    ])

    out = get_fast_llm().with_structured_output(
        UsefulnessDecision,
        method="json_schema"
    ).invoke(
        prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", "")
        )
    )
    dt = time.perf_counter() - t0

    return {
        "usefulness": out.status,
        "use_reason": out.reason,
        "trace": _trace(
            state,
            f"Usefulness check: {out.status} ({dt:.2f}s)"
        )
    }


def route_after_usefulness(
    state: RAGState
) -> Literal[
    "end",
    "rewrite_internal",
    "rewrite_web",
    "no_answer"
]:
    """Router: end if useful, otherwise try rewriting or give up."""
    if state.get("usefulness") == "useful":
        return "end"

    s = get_settings()

    if state.get("source_mode") == "internal":

        if (
            state.get("retrieval_rewrites", 0)
            < s.max_retrieval_rewrites
        ):
            return "rewrite_internal"

        return "rewrite_web"

    if (
        state.get("source_mode") == "web"
        and state.get("web_rewrites", 0)
        < s.max_web_rewrites
    ):
        return "rewrite_web"

    return "no_answer"


# ─── Node: No Answer ────────────────────────────────────────────────


def no_answer(state: RAGState):
    """Return a safe fallback when no reliable answer could be found."""
    return {
        "answer": (
            "I could not find enough reliable runbook or "
            "external evidence to recommend a safe "
            "troubleshooting action."
        ),
        "source_mode": "none",
        "trace": _trace(
            state,
            "Stopped: no reliable answer found"
        )
    }
