import os

from fastapi import FastAPI
from pydantic import BaseModel

from _version import __version__
from .config import config
from .middlewares.custom_middle_wares import CustomMiddleware
from .modules.pdf_loader import load_pdfs_async
from .modules.rag_agent import RAGAgentSingleton
from .modules.vector_store import create_vectorstore, load_vectorstore

fastapi_init_args = {}
fastapi_init_args["openapi_url"] = "/api/openapi.json"
fastapi_init_args["docs_url"] = "/api/docs"
fastapi_init_args["redoc_url"] = "/api/redoc"

app = FastAPI(
    title=config.APP_NAME,
    version=__version__,
    description="my data agent backend app",
    options={"origins": "*"},
    **fastapi_init_args,
)


# Run at app startup
@app.on_event("startup")
async def startup_event():
    from .routers.router import router

    # 초기화 단계에서 벡터스토어 로드 및 agent 초기화
    vectorstore = await initialize_vectorstore()

    query = "API 스펙 중 aNS는 어떤 것을 뜻하나요?"
    results = vectorstore.similarity_search(query, k=5)
    for i, result in enumerate(results, 1):
        print(f"Result {i}: {result.page_content}")

    rag_agent = RAGAgentSingleton()
    rag_agent.initialize(
        vectorstore=vectorstore, model_name=config.LLM_MODEL_NAME, top_k=3
    )

    app.include_router(router)


# 벡터스토어 파일 경로
vectorstore_path = os.path.join(
    os.path.dirname(__file__), "vector_store/vectorstore.faiss"
)
# PDF 파일 경로
pdf_file_path = os.path.join(os.path.dirname(__file__), "pdf_files")


async def initialize_vectorstore():
    # 벡터스토어가 이미 존재하는지 확인
    if os.path.exists(vectorstore_path):
        # 기존 벡터스토어 로드
        vectorstore = load_vectorstore(vectorstore_path, use_pinecone=False)
    else:
        # PDF 로드 및 벡터스토어 생성
        documents = await load_pdfs_async(pdf_file_path)
        vectorstore = create_vectorstore(documents, use_pinecone=False)  # 로컬로 예시
        # 벡터스토어 저장
        vectorstore.save_local(vectorstore_path)

    return vectorstore


app.add_middleware(
    CustomMiddleware,
)


class AppInfoData(BaseModel):
    app: str
    version: str
    stage: str
    apidocs: str


class AppInfoResponse(BaseModel):
    status: str
    data: AppInfoData


@app.get("/")
async def health_check():
    return AppInfoResponse(
        status="ok",
        data=AppInfoData(
            app=config.APP_NAME,
            version=__version__,
            stage=config.STAGE,
            apidocs="/api/docs",
        ),
    )
