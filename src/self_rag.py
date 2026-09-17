from typing import List, TypedDict, Literal, Annotated
import operator
import sqlite3
import time
import re
import logging
import asyncio
from pathlib import Path

import groq
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from tavily import TavilyClient

from src.config import get_settings
from src.vectorstore import get_retriever


class RAGState(TypedDict, total=False):
    user_question: str
    question: str
    memory: Annotated[List[str], operator.add]
    retrieval_query: str
    web_query: str
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    support_status: Literal[
        "fully_supported",
        "partially_supported",
        "no_support",
        ""
    ]
    evidence: List[str]
    usefulness: Literal["useful", "not_useful", ""]
    use_reason: str
    support_retries: int
    retrieval_rewrites: int
    web_rewrites: int
    source_mode: Literal["internal", "web", "direct", "none"]
    used_web_search: bool
    trace: List[str]


class RetrieveDecision(BaseModel):
    should_retrieve: bool


class RelevanceDecision(BaseModel):
    is_relevant: bool


class SupportDecision(BaseModel):
    status: Literal[
        "fully_supported",
        "partially_supported",
        "no_support"
    ]
    evidence: List[str] = Field(default_factory=list)


class UsefulnessDecision(BaseModel):
    status: Literal["useful", "not_useful"]
    reason: str


class QueryRewrite(BaseModel):
    query: str


logger = logging.getLogger(__name__)


class GroqRateLimitExhaustedError(RuntimeError):
    """Raised when Groq API rate limit is reached and bounded retries are exhausted."""
    pass


def is_rate_limit_error(e: Exception) -> bool:
    """Detect Groq HTTP 429 / rate-limit errors."""
    if isinstance(e, groq.RateLimitError):
        return True
    if getattr(e, "status_code", None) == 429:
        return True
    resp = getattr(e, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True
    cause = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
    if cause is not None and cause is not e and is_rate_limit_error(cause):
        return True
    msg = str(e).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg or "too many requests" in msg


def _parse_duration_seconds(val: str, is_ms: bool = False) -> float | None:
    """Safely parse duration string like '2.5', '2s', '500ms', '1m' into float seconds."""
    if not val:
        return None
    val_str = str(val).strip().lower()
    if is_ms:
        try:
            return float(val_str) / 1000.0
        except ValueError:
            pass
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(ms|s|m)?$", val_str)
    if m:
        num = float(m.group(1))
        unit = (m.group(2) or "s").lower()
        if unit == "ms":
            return num / 1000.0
        elif unit == "m":
            return num * 60.0
        return num
    try:
        return float(val_str)
    except ValueError:
        return None


def _get_retry_after(e: Exception, default: float = 2.0) -> float:
    """Extract retry-after duration in seconds from Groq response headers or error message."""
    response = getattr(e, "response", None)
    if response is not None and hasattr(response, "headers"):
        headers = response.headers
        ra = headers.get("retry-after")
        if ra:
            sec = _parse_duration_seconds(ra)
            if sec is not None:
                return max(0.5, min(sec, 10.0))
        ra_ms = headers.get("retry-after-ms")
        if ra_ms:
            sec = _parse_duration_seconds(ra_ms, is_ms=True)
            if sec is not None:
                return max(0.5, min(sec, 10.0))
        ra_reset = headers.get("x-ratelimit-reset-requests")
        if ra_reset:
            sec = _parse_duration_seconds(ra_reset)
            if sec is not None:
                return max(0.5, min(sec, 10.0))

    msg = str(e)
    m = re.search(r"(?:try again in|retry after|wait)\s+(\d+(?:\.\d+)?)\s*(ms|s|m)?", msg, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        unit = (m.group(2) or "s").lower()
        if unit == "ms":
            sec = val / 1000.0
        elif unit == "m":
            sec = val * 60.0
        else:
            sec = val
        return max(0.5, min(sec, 10.0))
    return default


class RateLimitedChatGroq(ChatGroq):
    """ChatGroq with bounded retry handling for HTTP 429 RateLimitErrors."""

    max_rate_limit_retries: int = 2

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        attempts = 0
        while True:
            try:
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                if not is_rate_limit_error(e):
                    raise
                attempts += 1
                if attempts > self.max_rate_limit_retries:
                    logger.warning(
                        "Groq rate limit (HTTP 429) retries exhausted after %d attempts: %s",
                        attempts,
                        e,
                    )
                    raise GroqRateLimitExhaustedError(
                        f"Groq API rate limit reached after {attempts} attempts. {e}"
                    ) from e

                wait_sec = _get_retry_after(e, default=2.0 * attempts)
                logger.info(
                    "Groq rate limit (HTTP 429) hit. Waiting %.2fs before retry %d/%d...",
                    wait_sec,
                    attempts,
                    self.max_rate_limit_retries,
                )
                time.sleep(wait_sec)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        attempts = 0
        while True:
            try:
                return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                if not is_rate_limit_error(e):
                    raise
                attempts += 1
                if attempts > self.max_rate_limit_retries:
                    logger.warning(
                        "Groq rate limit (HTTP 429) retries exhausted after %d attempts: %s",
                        attempts,
                        e,
                    )
                    raise GroqRateLimitExhaustedError(
                        f"Groq API rate limit reached after {attempts} attempts. {e}"
                    ) from e

                wait_sec = _get_retry_after(e, default=2.0 * attempts)
                logger.info(
                    "Groq rate limit (HTTP 429) hit. Waiting %.2fs before retry %d/%d...",
                    wait_sec,
                    attempts,
                    self.max_rate_limit_retries,
                )
                await asyncio.sleep(wait_sec)


def _llm():
    s = get_settings()

    if not s.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    return RateLimitedChatGroq(
        api_key=s.groq_api_key,
        model=s.groq_model,
        temperature=0,
    )


def _trace(state: RAGState, item: str):
    return [*(state.get("trace") or []), item]


def _format_context(docs: List[Document]) -> str:
    blocks = []

    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}

        if meta.get("source_type") == "web":
            head = (
                f"[WEB {i}] "
                f"{meta.get('title', '')} | "
                f"{meta.get('url', '')}"
            )
        else:
            head = (
                f"[INTERNAL {i}] "
                f"{meta.get('title') or meta.get('document_name') or meta.get('source', '')}"
            )

            if meta.get("page") is not None:
                head += f" | page {int(meta['page']) + 1}"

        blocks.append(f"{head}\n{d.page_content}")

    return "\n\n---\n\n".join(blocks)


def _memory_text(state: RAGState, limit: int = 4) -> str:
    items = state.get("memory") or []

    return (
        "\n\n".join(items[-limit:])
        if items
        else "No previous conversation context."
    )


def contextualize_question(state: RAGState):
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

    out = _llm().with_structured_output(
        QueryRewrite,
        method="json_schema"
    ).invoke(
        prompt.format_messages(
            history=history,
            question=user_question
        )
    )
    dt = time.perf_counter() - t0

    return {
        "question": out.query,
        "trace": _trace(
            state,
            f"Memory contextualized question: {out.query} ({dt:.2f}s)"
        )
    }


def commit_memory(state: RAGState):
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


def decide_retrieval(state: RAGState):
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

    response = _llm().invoke(
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

    return (
        "retrieve"
        if state.get("need_retrieval", True)
        else "direct"
    )


def generate_direct(state: RAGState):
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

    ans = _llm().invoke(
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


def retrieve_internal(state: RAGState):
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


def grade_relevance(state: RAGState):
    t0 = time.perf_counter()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Judge relevance at the topic/evidence level. "
            "A document is relevant when it contains information "
            "useful for answering the user's question. "
            "Do not require the exact final answer. "
            "Be strict about unrelated content."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Document:\n{document}"
        ),
    ])

    grader = _llm().with_structured_output(
        RelevanceDecision,
        method="json_schema"
    )

    relevant = []

    for d in state.get("docs", []):
        try:
            decision = grader.invoke(
                prompt.format_messages(
                    question=state["question"],
                    document=d.page_content[:7000]
                )
            )

            if decision.is_relevant:
                relevant.append(d)

        except Exception:
            continue

    mode = state.get("source_mode", "internal")
    dt = time.perf_counter() - t0

    return {
        "relevant_docs": relevant,
        "trace": _trace(
            state,
            f"Relevance grade ({mode}): "
            f"{len(relevant)}/{len(state.get('docs', []))} relevant ({dt:.2f}s)"
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


def rewrite_internal_query(state: RAGState):
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

    out = _llm().with_structured_output(
        QueryRewrite,
        method="json_schema"
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

    out = _llm().with_structured_output(
        QueryRewrite,
        method="json_schema"
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


def web_search(state: RAGState):
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


def generate_from_context(state: RAGState):
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

    ans = _llm().invoke(
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


def check_support(state: RAGState):
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

    out = _llm().with_structured_output(
        SupportDecision,
        method="json_schema"
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

    if state.get("support_status") == "fully_supported":
        return "usefulness"

    if (
        state.get("support_retries", 0)
        >= get_settings().max_support_retries
    ):
        return "usefulness"

    return "revise"


def revise_answer(state: RAGState):
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

    ans = _llm().invoke(
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


def check_usefulness(state: RAGState):
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

    out = _llm().with_structured_output(
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


def no_answer(state: RAGState):
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


def build_graph():

    g = StateGraph(RAGState)

    g.add_node(
        "contextualize",
        contextualize_question
    )

    g.add_node(
        "decide_retrieval",
        decide_retrieval
    )

    g.add_node(
        "direct",
        generate_direct
    )

    g.add_node(
        "retrieve",
        retrieve_internal
    )

    g.add_node(
        "grade",
        grade_relevance
    )

    g.add_node(
        "rewrite_internal",
        rewrite_internal_query
    )

    g.add_node(
        "rewrite_web",
        rewrite_web_query
    )

    g.add_node(
        "web_search",
        web_search
    )

    g.add_node(
        "generate",
        generate_from_context
    )

    g.add_node(
        "support",
        check_support
    )

    g.add_node(
        "revise",
        revise_answer
    )

    g.add_node(
        "usefulness",
        check_usefulness
    )

    g.add_node(
        "no_answer",
        no_answer
    )

    g.add_node(
        "commit_memory",
        commit_memory
    )

    # -----------------------------
    # Graph flow
    # -----------------------------

    g.add_edge(
        START,
        "contextualize"
    )

    g.add_edge(
        "contextualize",
        "decide_retrieval"
    )

    g.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {
            "direct": "direct",
            "retrieve": "retrieve"
        }
    )

    g.add_edge(
        "direct",
        "commit_memory"
    )

    g.add_edge(
        "retrieve",
        "grade"
    )

    g.add_conditional_edges(
        "grade",
        route_after_relevance,
        {
            "generate": "generate",
            "rewrite_internal": "rewrite_internal",
            "rewrite_web": "rewrite_web",
            "no_answer": "no_answer"
        }
    )

    g.add_edge(
        "rewrite_internal",
        "retrieve"
    )

    g.add_edge(
        "rewrite_web",
        "web_search"
    )

    g.add_edge(
        "web_search",
        "grade"
    )

    g.add_edge(
        "generate",
        "support"
    )

    g.add_conditional_edges(
        "support",
        route_after_support,
        {
            "usefulness": "usefulness",
            "revise": "revise"
        }
    )

    g.add_edge(
        "revise",
        "support"
    )

    g.add_conditional_edges(
        "usefulness",
        route_after_usefulness,
        {
            "end": "commit_memory",
            "rewrite_internal": "rewrite_internal",
            "rewrite_web": "rewrite_web",
            "no_answer": "no_answer"
        }
    )

    g.add_edge(
        "no_answer",
        "commit_memory"
    )

    g.add_edge(
        "commit_memory",
        END
    )

    # -----------------------------
    # SQLite persistent memory
    # -----------------------------

    db_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "langgraph_memory.sqlite"
    )

    db_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(
        str(db_path),
        check_same_thread=False
    )

    checkpointer = SqliteSaver(conn)

    return g.compile(
        checkpointer=checkpointer
    )


def _sources(docs: List[Document]):

    seen = set()
    out = []

    for d in docs or []:

        m = d.metadata or {}

        typ = (
            "web"
            if m.get("source_type") == "web"
            else "internal"
        )

        key = (
            typ,
            m.get("url") or m.get("source"),
            m.get("page")
        )

        if key in seen:
            continue

        seen.add(key)

        item = {
            "type": typ,
            "title": (
                m.get("title")
                or m.get("document_name")
                or ""
            ),
            "source": (
                m.get("source")
                or ""
            ),
            "url": (
                m.get("url")
                if typ == "web"
                else None
            ),
            "page": (
                int(m["page"]) + 1
                if (
                    typ == "internal"
                    and m.get("page") is not None
                )
                else None
            ),
        }

        out.append(item)

    return out


_graph = None


def run_self_rag(
    question: str,
    thread_id: str
) -> dict:
    t_start = time.perf_counter()

    global _graph

    if _graph is None:
        _graph = build_graph()

    initial: RAGState = {

        "user_question": question,

        "question": question,

        "memory": [],

        "retrieval_query": question,

        "web_query": "",

        "docs": [],

        "relevant_docs": [],

        "context": "",

        "answer": "",

        "support_status": "",

        "evidence": [],

        "usefulness": "",

        "use_reason": "",

        "support_retries": 0,

        "retrieval_rewrites": 0,

        "web_rewrites": 0,

        "source_mode": "internal",

        "used_web_search": False,

        "trace": [],
    }

    try:
        result = _graph.invoke(
            initial,
            config={
                "configurable": {
                    "thread_id": thread_id
                },
                "recursion_limit": 60
            }
        )
    except (GroqRateLimitExhaustedError, groq.RateLimitError):
        total_elapsed = time.perf_counter() - t_start
        return {
            "answer": (
                "OpsGuard is temporarily experiencing high traffic with the AI model provider (Groq rate limit reached). "
                "Please wait a moment and submit your incident question again."
            ),
            "route": "Rate Limit Exceeded",
            "used_web_search": False,
            "support_status": "rate_limited",
            "usefulness": "rate_limited",
            "sources": [],
            "trace": [
                "Groq API rate limit reached (HTTP 429). Maximum retries exhausted.",
                f"Total Self-RAG latency: {total_elapsed:.2f}s",
            ],
            "thread_id": thread_id,
            "memory_turns": 0,
        }
    except Exception as e:
        if is_rate_limit_error(e):
            total_elapsed = time.perf_counter() - t_start
            return {
                "answer": (
                    "OpsGuard is temporarily experiencing high traffic with the AI model provider (Groq rate limit reached). "
                    "Please wait a moment and submit your incident question again."
                ),
                "route": "Rate Limit Exceeded",
                "used_web_search": False,
                "support_status": "rate_limited",
                "usefulness": "rate_limited",
                "sources": [],
                "trace": [
                    "Groq API rate limit reached (HTTP 429). Maximum retries exhausted.",
                    f"Total Self-RAG latency: {total_elapsed:.2f}s",
                ],
                "thread_id": thread_id,
                "memory_turns": 0,
            }
        raise

    total_elapsed = time.perf_counter() - t_start
    final_trace = list(result.get("trace", []))
    final_trace.append(f"Total Self-RAG latency: {total_elapsed:.2f}s")

    mode = result.get(
        "source_mode",
        "none"
    )

    route = {
        "internal": "Private Runbooks",
        "web": "Internet Search",
        "direct": "General Knowledge",
        "none": "No Reliable Evidence"
    }.get(
        mode,
        mode
    )

    return {
        "answer": result.get(
            "answer",
            ""
        ),

        "route": route,

        "used_web_search": bool(
            result.get("used_web_search")
        ),

        "support_status": result.get(
            "support_status",
            ""
        ),

        "usefulness": result.get(
            "usefulness",
            ""
        ),

        "sources": _sources(
            result.get(
                "relevant_docs",
                []
            )
        ),

        "trace": final_trace,

        "thread_id": thread_id,

        "memory_turns": len(
            result.get(
                "memory",
                []
            )
        ),
    }