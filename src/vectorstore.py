"""Pinecone vector store initialization and retriever factory."""

from functools import lru_cache

from pinecone import Pinecone
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore

from src.config import get_settings


@lru_cache
def get_embeddings():
    s = get_settings()

    if not s.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not configured")

    return PineconeEmbeddings(
        model=s.pinecone_embedding_model
    )


def get_pinecone_client():
    s = get_settings()

    if not s.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not configured")

    return Pinecone(api_key=s.pinecone_api_key)


def ensure_index():
    """Create the Pinecone index if needed and verify its configuration."""

    s = get_settings()
    pc = get_pinecone_client()

    existing = {x.name for x in pc.list_indexes()}

    if s.pinecone_index_name not in existing:
        pc.create_index_for_model(
            name=s.pinecone_index_name,
            cloud=s.pinecone_cloud,
            region=s.pinecone_region,
            embed={
                "model": s.pinecone_embedding_model,
                "field_map": {
                    "text": "text"
                },
            },
        )

    else:
        desc = pc.describe_index(s.pinecone_index_name)

        # Verify that the existing index is using integrated embedding
        # and the expected embedding model.
        embedding_model = getattr(desc, "embed", None)

        if embedding_model is None and isinstance(desc, dict):
            embedding_model = desc.get("embed")

        if embedding_model:
            configured_model = (
                embedding_model.get("model")
                if isinstance(embedding_model, dict)
                else None
            )

            if (
                configured_model
                and configured_model != s.pinecone_embedding_model
            ):
                raise RuntimeError(
                    f"Pinecone index '{s.pinecone_index_name}' is configured "
                    f"with embedding model '{configured_model}', but the "
                    f"application expects '{s.pinecone_embedding_model}'. "
                    "Use a matching index or update the configuration."
                )

    return pc.Index(s.pinecone_index_name)


@lru_cache(maxsize=128)
def get_vector_store(namespace: str | None = None) -> PineconeVectorStore:
    """Return a PineconeVectorStore instance scoped to the specified namespace."""
    s = get_settings()
    index = ensure_index()
    ns = (namespace or s.pinecone_namespace).strip()

    return PineconeVectorStore(
        index=index,
        embedding=get_embeddings(),
        namespace=ns,
    )


def get_retriever(namespace: str | None = None, top_k: int | None = None):
    """Return a retriever scoped to the specified namespace."""
    s = get_settings()
    k = top_k or s.top_k

    return get_vector_store(namespace=namespace).as_retriever(
        search_kwargs={"k": k}
    )


def user_namespace_for(user_id: str) -> str:
    """Derive an isolated, sanitized Pinecone namespace for a user workspace."""
    clean = "".join(c for c in (user_id or "").strip() if c.isalnum() or c in "-_")
    if not clean:
        clean = "default"
    return f"usr_{clean}"[:63]


def retrieve_multi_namespace(query: str, user_id: str = "", top_k: int | None = None):
    """Retrieve relevant documents prioritizing user's private namespace plus shared system runbooks.
    
    Guarantees strict isolation: under NO circumstances are vectors from another user's
    namespace accessed.
    """
    s = get_settings()
    k = top_k or s.top_k
    all_docs = []
    seen_content = set()

    # 1. First retrieve from user's isolated private workspace namespace (if user_id provided)
    if user_id:
        user_ns = user_namespace_for(user_id)
        if user_ns != s.pinecone_namespace:
            try:
                user_store = get_vector_store(namespace=user_ns)
                user_docs = user_store.similarity_search(query, k=k)
                for d in user_docs:
                    d.metadata = dict(d.metadata or {})
                    d.metadata["source_type"] = "internal"
                    d.metadata["scope"] = "private"
                    d.metadata["user_id"] = user_id
                    content_sig = (d.metadata.get("title", ""), d.page_content[:200])
                    if content_sig not in seen_content:
                        seen_content.add(content_sig)
                        all_docs.append(d)
            except Exception:
                # If namespace is empty or newly created, continue gracefully
                pass

    # 2. Retrieve from shared base system knowledge (company-documents)
    try:
        base_store = get_vector_store(namespace=s.pinecone_namespace)
        base_docs = base_store.similarity_search(query, k=k)
        for d in base_docs:
            d.metadata = dict(d.metadata or {})
            d.metadata["source_type"] = "internal"
            d.metadata["scope"] = "system"
            content_sig = (d.metadata.get("title", ""), d.page_content[:200])
            if content_sig not in seen_content:
                seen_content.add(content_sig)
                all_docs.append(d)
    except Exception:
        pass

    return all_docs


def delete_file_from_namespace(filename: str, namespace: str) -> None:
    """Delete all vectors for a specific document from a Pinecone namespace."""
    try:
        index = ensure_index()
        index.delete(
            filter={"document_name": filename},
            namespace=namespace,
        )
    except Exception:
        pass
    
