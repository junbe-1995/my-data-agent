from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS, Pinecone
from transformers import CLIPProcessor, CLIPModel

from my_data_backend.config import config


class CLIPImageEmbeddings(Embeddings):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed(self, texts):
        # 토큰 수를 77로 제한하여 텍스트 자르기
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        features = self.model.get_text_features(**inputs)
        return features.detach().numpy()

    def embed_documents(self, texts):
        return self.embed(texts)

    def embed_query(self, query):
        return self.embed([query])


def create_vectorstore_by_clip(documents, use_pinecone=False):
    embeddings = CLIPImageEmbeddings()

    if use_pinecone:
        from my_data_backend.utils.pinecone import PineconeSingleton

        pinecone_singleton = PineconeSingleton()
        pinecone_singleton.initialize()

        # Pinecone에 저장
        vectorstore = Pinecone.from_documents(
            documents,
            embedding_function=embeddings.embed,
            index_name=config.PINECONE_INDEX_NAME,
        )
    else:
        # FAISS에 저장
        vectorstore = FAISS.from_documents(documents, embedding=embeddings)

    return vectorstore


def load_vectorstore_by_clip(file_path: str, use_pinecone=False):
    embeddings = CLIPImageEmbeddings()

    if use_pinecone:
        vectorstore = Pinecone(
            index_name=config.PINECONE_INDEX_NAME, embedding_function=embeddings.embed
        )
    else:
        vectorstore = FAISS.load_local(
            file_path, embeddings=embeddings, allow_dangerous_deserialization=True
        )
    return vectorstore
