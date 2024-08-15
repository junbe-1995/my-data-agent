from typing import Optional

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages.system import SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from my_data_backend.modules.history_manager import HistoryManager
from my_data_backend.modules.prompt_template import PromptTemplateSingleton


class RAGAgentSingleton:
    _instance: Optional["RAGAgentSingleton"] = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(RAGAgentSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.vectorstore = None
        self.rag_chain = None
        self.history_manager = None
        self.llm = None
        self._initialized = True

    def initialize(
        self,
        vectorstore,
        model_name: str = "gpt-4o-mini",
        top_k: int = 3,
        max_tokens: int = 1000,
    ):
        # 벡터스토어 초기화
        self.vectorstore = vectorstore
        # history manager 초기화
        self.history_manager = HistoryManager()
        # ChatOpenAI 모델 초기화
        self.llm = ChatOpenAI(model=model_name, temperature=0.3, max_tokens=max_tokens)
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

    def get_session_history(self, device_id: str):
        memory = self.history_manager.get_or_create_memory(device_id, self.llm)
        return memory.chat_memory

    def inference(self, device_id: str, query: str):
        if not self.rag_chain:
            raise ValueError(
                "RAG Agent has not been initialized with a vector store and LLM."
            )

        # 사용자별 메모리 가져오기
        memory = self.get_session_history(device_id)

        # 리트리벌된 문서 내용을 히스토리에 SystemMessage로 추가
        retrieved_documents = self.vectorstore.similarity_search(query, top_k=1)
        if retrieved_documents:
            doc = retrieved_documents[0]
            memory.add_message(
                SystemMessage(content=f"Retrieved Document: {doc.page_content}")
            )

        # 사용자의 메시지 히스토리와 함께 체인 생성
        runnable_chain = RunnableWithMessageHistory(
            self.rag_chain,
            get_session_history=lambda: memory,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # AI 응답 생성
        input_data = {"input": query}
        response = runnable_chain.invoke(input_data)
        if not isinstance(response, dict) or "answer" not in response:
            raise ValueError("Unexpected response format from the chain.")

        return response.get("answer")
