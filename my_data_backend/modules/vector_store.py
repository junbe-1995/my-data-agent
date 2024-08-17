import cohere
from typing import List, Optional

from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS, Pinecone

from my_data_backend.config import config
from my_data_backend.utils.pinecone import PineconeSingleton


class CohereEmbeddingsForAsync(CohereEmbeddings):
    async def aembed(
        self,
        texts: List[str],
        *,
        input_type: Optional[cohere.EmbedInputType] = None,
    ) -> List[List[float]]:
        embeddings = None  # 변수를 초기화하여 할당되지 않은 상태로 접근하는 것을 방지

        try:
            # 부모 클래스의 aembed_with_retry 메서드를 사용하여 임베딩 생성
            embeddings = (
                await self.aembed_with_retry(
                    model=self.model,
                    texts=texts,
                    input_type=input_type,
                    truncate=self.truncate,
                    embedding_types=self.embedding_types,
                )
            ).embeddings

            # 반환된 임베딩 데이터를 float로 변환
            # 각 임베딩에서 float_ 속성에 접근하여 데이터를 가져옴
            if isinstance(embeddings, cohere.EmbedByTypeResponseEmbeddings):
                processed_embeddings = [list(map(float, e)) for e in embeddings.float_]
            else:
                # original code
                processed_embeddings = [list(map(float, e)) for e in embeddings]
            return processed_embeddings
        except Exception as e:
            print(f"Error in aembed: {e}")
            if embeddings is not None:
                print(f"Embeddings data before error: {embeddings}")
            else:
                print("Embeddings were not generated before the error.")
            raise


def create_vectorstore(documents, use_pinecone=False):
    embeddings = CohereEmbeddingsForAsync(
        model=config.COHERE_MODEL_NAME, cohere_api_key=config.COHERE_API_KEY
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
    # embeddings = CohereEmbeddings(
    #     model=config.COHERE_MODEL_NAME, cohere_api_key=config.COHERE_API_KEY
    # )
    embeddings = CohereEmbeddingsForAsync(
        model=config.COHERE_MODEL_NAME, cohere_api_key=config.COHERE_API_KEY
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
