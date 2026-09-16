from src.config import get_settings

settings = get_settings()

print("Pinecone API Key:", settings.pinecone_api_key)
# print("Pinecone Index Name:", settings.pinecone_index_name)
# print("Pinecone Namespace:", settings.pinecone_namespace)
# print("Pinecone Cloud:", settings.pinecone_cloud)
# print("Pinecone Region:", settings.pinecone_region)
# print("Pinecone Embedding Model:", settings.pinecone_embedding_model)

