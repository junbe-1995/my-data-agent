from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


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
        # ChatOpenAI 모델 초기화
        llm = ChatOpenAI(model=model_name, temperature=0.2, max_tokens=max_tokens)

        # 프롬프트 템플릿 설정
        prompt_template = ChatPromptTemplate.from_template(
            "You are an expert in API specifications. Based on the following retrieved documents from the API specification:\n\n"
            "{context}\n\n"
            "Please provide a clear and concise explanation of what the term '{input}' specifically refers to in the context of the provided API specifications. "
            "Ensure your answer is directly related to the provided documents and gives an accurate definition or explanation. "
            "Avoid unnecessary details, including repeating the query, and respond in Korean."
        )

        # create_stuff_documents_chain을 이용해 CombineDocumentsChain 생성
        combine_documents_chain = create_stuff_documents_chain(
            llm=llm, prompt=prompt_template
        )

        # create_retrieval_chain을 이용해 RAG 체인 생성
        self.rag_chain = create_retrieval_chain(
            self.vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"fetch_k": top_k}
            ),
            combine_documents_chain,
        )

    def inference(self, query: str):
        if not self.rag_chain:
            raise ValueError(
                "RAG Agent has not been initialized with a vector store and LLM."
            )

        # 검색된 문서들을 기반으로 답변을 생성
        response = self.rag_chain.invoke({"input": query})

        answer = response["answer"]
        return answer
