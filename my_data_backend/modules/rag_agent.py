from __future__ import annotations

import asyncio
import threading
from typing import Optional

from PIL import Image
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from my_data_backend.modules.history_manager import HistoryManager
from my_data_backend.modules.prompt_template import PromptTemplateSingleton


class RAGAgentSingleton:
    _instance: Optional[RAGAgentSingleton] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(RAGAgentSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = False

    def initialize(
        self,
        vectorstore,
        image_vectorstore,
        model_name: str = "gpt-4o-mini",
        top_k: int = 3,
        max_tokens: int = 1000,
    ):
        with self._lock:
            if not self._initialized:
                # 벡터스토어 초기화
                self.vectorstore = vectorstore
                self.image_vectorstore = image_vectorstore
                # history manager 초기화
                self.history_manager = HistoryManager()
                # ChatOpenAI 모델 초기화
                self.llm = ChatOpenAI(
                    model=model_name, temperature=0.3, max_tokens=max_tokens
                )
                # 프롬프트 템플릿 설정
                prompt_template = PromptTemplateSingleton().get_template()

                # create_stuff_documents_chain을 이용해 CombineDocumentsChain 생성
                combine_documents_chain = create_stuff_documents_chain(
                    llm=self.llm, prompt=prompt_template
                )

                # create_retrieval_chain을 이용해 RAG 체인 생성
                self.rag_chain = create_retrieval_chain(
                    self.vectorstore.as_retriever(
                        search_type="mmr", search_kwargs={"fetch_k": top_k}
                    ),
                    combine_documents_chain,
                )

                self._initialized = True

    async def get_session_history(self, device_id: str):
        memory = await self.history_manager.get_or_create_memory(device_id, self.llm)
        return memory.chat_memory

    async def get_documents_by_image(self, image):
        # 이미지 임베딩 생성
        image_embedding = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.image_vectorstore.embedding_function.embed_query(image=image),
        )

        # FAISS 벡터 스토어에서 유사 문서 검색
        image_documents = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.image_vectorstore.similarity_search_by_vector(
                image_embedding, top_k=2
            ),
        )

        return image_documents

    async def inference(
        self, device_id: str, query: str = None, image: Image.Image = None
    ):
        if not self.rag_chain:
            raise ValueError(
                "RAG Agent has not been initialized with a vector store and LLM."
            )

        # 사용자별 메모리 가져오기
        memory = await self.get_session_history(device_id)

        # 이미지 입력이 존재하는 경우 CLIP 기반 벡터스토어에서 문서 검색 후 추가
        retrieved_documents_by_image = []
        if image:
            image_documents = await self.get_documents_by_image(image)
            retrieved_documents_by_image.extend(image_documents[:2])
        source_documents_by_image = [
            doc.page_content for doc in retrieved_documents_by_image
        ]

        # 리트리브한 문서와 히스토리를 포함하여 체인 실행
        runnable_chain = RunnableWithMessageHistory(
            self.rag_chain,
            get_session_history=lambda: memory,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        input_data = {
            "input": query if query else "Image search",
            "source_documents": source_documents_by_image,
        }

        response = await runnable_chain.ainvoke(input_data)
        if not isinstance(response, dict) or "answer" not in response:
            raise ValueError("Unexpected response format from the chain.")

        return response.get("answer")
