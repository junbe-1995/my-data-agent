import numpy as np
import torch
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS, Pinecone
from transformers import CLIPProcessor, CLIPModel

from my_data_backend.config import config


class CLIPImageEmbeddings(Embeddings):
    def __init__(self, model_name=config.CLIP_MODEL_NAME):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed(self, texts=None, images=None, batch_size=32):
        features = []
        with torch.no_grad():  # 메모리 최적화
            if texts:
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    inputs = self.processor(
                        text=batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=77,
                    )
                    batch_features = self.model.get_text_features(**inputs)
                    # 2차원으로 변환하여 numpy 배열로 변환
                    batch_features = (
                        batch_features.view(batch_features.size(0), -1).detach().numpy()
                    )
                    features.append(batch_features)
            if images:
                for i in range(0, len(images), batch_size):
                    batch_images = images[i : i + batch_size]
                    inputs = self.processor(
                        images=batch_images,
                        return_tensors="pt",
                        padding=True,
                    )
                    batch_features = self.model.get_image_features(**inputs)
                    # 2차원으로 변환하여 numpy 배열로 변환
                    batch_features = (
                        batch_features.view(batch_features.size(0), -1).detach().numpy()
                    )
                    features.append(batch_features)
        result = np.vstack(features)
        if result.shape[0] == 1:
            result = np.squeeze(result, axis=0)
        return result

    def embed_documents(self, texts=None, images=None):
        return self.embed(texts=texts, images=images)

    def embed_query(self, query=None, image=None):
        return self.embed(
            texts=[query] if query else None, images=[image] if image else None
        )


def create_vectorstore_by_clip(documents, use_pinecone=False):
    embeddings = CLIPImageEmbeddings()

    if use_pinecone:
        from my_data_backend.modules.pinecone import PineconeSingleton

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
