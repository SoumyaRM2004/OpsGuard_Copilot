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


@lru_cache
def get_vector_store():
    s = get_settings()

    index = ensure_index()

    return PineconeVectorStore(
        index=index,
        embedding=get_embeddings(),
        namespace=s.pinecone_namespace,
    )


def get_retriever():
    s = get_settings()

    return get_vector_store().as_retriever(
        search_kwargs={"k": s.top_k}
    )
    
