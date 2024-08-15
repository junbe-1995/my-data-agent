from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS, Pinecone

from my_data_backend.config import config
from my_data_backend.utils.pinecone import PineconeSingleton


def create_vectorstore(documents, use_pinecone=False):
    embeddings = CohereEmbeddings(
        model="multilingual-22-12", cohere_api_key=config.COHERE_API_KEY
    )

    if use_pinecone:
        # Pinecone 사용 시 초기화
        pinecone_singleton = PineconeSingleton()
        pinecone_singleton.initialize()
        vectorstore = Pinecone.from_documents(
            documents, embeddings, index_name=config.PINECONE_INDEX_NAME
        )
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore


def load_vectorstore(file_path: str, use_pinecone=False):
    embeddings = CohereEmbeddings(
        model="multilingual-22-12", cohere_api_key=config.COHERE_API_KEY
    )

    if use_pinecone:
        vectorstore = Pinecone(
            index_name=config.PINECONE_INDEX_NAME, embedding_function=embeddings
        )
    else:
        vectorstore = FAISS.load_local(
            file_path, embeddings, allow_dangerous_deserialization=True
        )
    return vectorstore
