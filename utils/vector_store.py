import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

VECTOR_STORE_PATH = "faiss_index"


def build_vector_store(chunks: list, api_key: str) -> FAISS:
    """Build a FAISS vector store from document chunks."""
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(VECTOR_STORE_PATH)
    return store


def load_vector_store(api_key: str) -> FAISS | None:
    """Load an existing FAISS vector store if it exists.

    ``allow_dangerous_deserialization=True`` is required by LangChain's FAISS
    loader because the index is persisted with pickle.  This is safe here
    because the index is written exclusively by ``build_vector_store`` in the
    same application — no untrusted data ever reaches this path.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    if os.path.exists(VECTOR_STORE_PATH):
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return None


def search_documents(store: FAISS, query: str, k: int = 4) -> list:
    """Search for relevant document chunks."""
    return store.similarity_search(query, k=k)
