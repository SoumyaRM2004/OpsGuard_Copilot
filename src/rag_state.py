"""Pydantic models and TypedDict state for the Self-RAG workflow."""

from typing import List, TypedDict, Literal, Annotated
import operator

from pydantic import BaseModel, Field
from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    """State that flows through every node in the Self-RAG LangGraph."""
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


class DocumentGrade(BaseModel):
    doc_index: int = Field(
        description="1-based index matching [DOC i]"
    )
    is_relevant: bool = Field(
        description="True if the document is relevant to the question"
    )


class BatchRelevanceDecision(BaseModel):
    results: List[DocumentGrade] = Field(
        description="Relevance judgment for each retrieved document"
    )


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
    """Rewritten query for search and retrieval."""
    query: str = Field(
        description="The rewritten standalone search query string"
    )
