"""Self-RAG graph builder and entry-point runner."""

import time
import logging
from typing import List
from pathlib import Path

import groq
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import get_settings
from src.llm import GroqRateLimitExhaustedError, is_rate_limit_error
from src.rag_state import RAGState
from src.graph_nodes import (
    contextualize_question,
    commit_memory,
    decide_retrieval,
    route_after_decide,
    generate_direct,
    retrieve_internal,
    grade_relevance,
    route_after_relevance,
    rewrite_internal_query,
    rewrite_web_query,
    web_search,
    generate_from_context,
    check_support,
    route_after_support,
    revise_answer,
    check_usefulness,
    route_after_usefulness,
    no_answer,
)


logger = logging.getLogger(__name__)


# ─── Source Extraction ───────────────────────────────────────────────


def _sources(docs: List[Document]):
    """Extract deduplicated source citations from relevant documents."""
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


# ─── Graph Builder ───────────────────────────────────────────────────


def build_graph():
    """Build and compile the Self-RAG LangGraph with SQLite persistence."""

    g = StateGraph(RAGState)

    # Register all nodes
    g.add_node("contextualize", contextualize_question)
    g.add_node("decide_retrieval", decide_retrieval)
    g.add_node("direct", generate_direct)
    g.add_node("retrieve", retrieve_internal)
    g.add_node("grade", grade_relevance)
    g.add_node("rewrite_internal", rewrite_internal_query)
    g.add_node("rewrite_web", rewrite_web_query)
    g.add_node("web_search", web_search)
    g.add_node("generate", generate_from_context)
    g.add_node("support", check_support)
    g.add_node("revise", revise_answer)
    g.add_node("usefulness", check_usefulness)
    g.add_node("no_answer", no_answer)
    g.add_node("commit_memory", commit_memory)

    # Graph flow
    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "decide_retrieval")

    g.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {"direct": "direct", "retrieve": "retrieve"}
    )

    g.add_edge("direct", "commit_memory")
    g.add_edge("retrieve", "grade")

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

    g.add_edge("rewrite_internal", "retrieve")
    g.add_edge("rewrite_web", "web_search")
    g.add_edge("web_search", "grade")
    g.add_edge("generate", "support")

    g.add_conditional_edges(
        "support",
        route_after_support,
        {"usefulness": "usefulness", "revise": "revise"}
    )

    g.add_edge("revise", "support")

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

    g.add_edge("no_answer", "commit_memory")
    g.add_edge("commit_memory", END)

    # SQLite persistent memory (Fix 7: thread-safe connection)
    db_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "langgraph_memory.sqlite"
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = SqliteSaver.from_conn_string(str(db_path))

    return g.compile(checkpointer=checkpointer)


# Build once at module level instead of using a mutable global.
_graph = build_graph()


# ─── Rate-limit Response Helper ──────────────────────────────────────


def _rate_limit_response(thread_id: str, elapsed: float) -> dict:
    """Build a standardized response when Groq rate limits are exhausted."""
    return {
        "answer": (
            "OpsGuard is temporarily experiencing high traffic with the AI model provider "
            "(Groq rate limit reached). Please wait a moment and submit your incident question again."
        ),
        "route": "Rate Limit Exceeded",
        "used_web_search": False,
        "support_status": "rate_limited",
        "usefulness": "rate_limited",
        "sources": [],
        "trace": [
            "Groq API rate limit reached (HTTP 429). Maximum retries exhausted.",
            f"Total Self-RAG latency: {elapsed:.2f}s",
        ],
        "thread_id": thread_id,
        "memory_turns": 0,
    }


# ─── Entry Point ─────────────────────────────────────────────────────


def run_self_rag(question: str, thread_id: str) -> dict:
    """Run the full Self-RAG pipeline for a single question."""
    t_start = time.perf_counter()

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
        return _rate_limit_response(thread_id, time.perf_counter() - t_start)
    except Exception as e:
        if is_rate_limit_error(e):
            return _rate_limit_response(thread_id, time.perf_counter() - t_start)
        raise

    total_elapsed = time.perf_counter() - t_start
    final_trace = list(result.get("trace", []))
    final_trace.append(f"Total Self-RAG latency: {total_elapsed:.2f}s")

    mode = result.get("source_mode", "none")

    route = {
        "internal": "Private Runbooks",
        "web": "Internet Search",
        "direct": "General Knowledge",
        "none": "No Reliable Evidence"
    }.get(mode, mode)

    return {
        "answer": result.get("answer", ""),
        "route": route,
        "used_web_search": bool(result.get("used_web_search")),
        "support_status": result.get("support_status", ""),
        "usefulness": result.get("usefulness", ""),
        "sources": _sources(result.get("relevant_docs", [])),
        "trace": final_trace,
        "thread_id": thread_id,
        "memory_turns": len(result.get("memory", [])),
    }