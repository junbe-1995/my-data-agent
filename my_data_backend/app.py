import os

from fastapi import FastAPI
from pydantic import BaseModel

from _version import __version__
from .config import config
from .middlewares.custom_middle_wares import CustomMiddleware
from .modules.history_manager import HistoryManager
from .modules.pdf_loader import load_pdfs_async
from .modules.prompt_template import PromptTemplateSingleton
from .modules.rag_agent import RAGAgentSingleton
from .modules.vector_store_by_clip import (
    load_vectorstore_by_clip,
    create_vectorstore_by_clip,
)

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

    # vectorstore 로드, prompt template / history manager / agent 초기화
    # vectorstore = await initialize_vectorstore()
    vectorstore = await initialize_vectorstore_by_clip()
    PromptTemplateSingleton().initialize()
    HistoryManager()

    rag_agent = RAGAgentSingleton()
    rag_agent.initialize(
        vectorstore=vectorstore,
        model_name=config.OPENAI_LLM_MODEL_NAME,
        top_k=config.VECTOR_SEARCH_TOP_K,
    )

    app.include_router(router)


# 벡터스토어 파일 경로
vectorstore_path = os.path.join(
    os.path.dirname(__file__), "vector_store/vectorstore.faiss"
)
vectorstore_path_by_clip = os.path.join(
    os.path.dirname(__file__), "vector_store/vectorstore_by_clip.faiss"
)
# PDF 파일 경로
pdf_file_path = os.path.join(os.path.dirname(__file__), "pdf_files")


# async def initialize_vectorstore():
#     # 벡터스토어가 이미 존재하는지 확인
#     if os.path.exists(vectorstore_path):
#         # 기존 벡터스토어 로드
#         vectorstore = load_vectorstore(
#             vectorstore_path, use_pinecone=config.USE_PINECONE
#         )
#     else:
#         # PDF 로드 및 벡터스토어 생성
#         documents = await load_pdfs_async(pdf_file_path)
#         vectorstore = create_vectorstore(documents, use_pinecone=config.USE_PINECONE)
#         # 벡터스토어 저장
#         vectorstore.save_local(vectorstore_path)
#
#     return vectorstore


async def initialize_vectorstore_by_clip():
    # 벡터스토어가 이미 존재하는지 확인
    if os.path.exists(vectorstore_path_by_clip):
        # 기존 벡터스토어 로드
        vectorstore = load_vectorstore_by_clip(
            vectorstore_path_by_clip, use_pinecone=config.USE_PINECONE
        )
    else:
        # PDF 로드 및 벡터스토어 생성
        documents = await load_pdfs_async(pdf_file_path)
        vectorstore = create_vectorstore_by_clip(
            documents, use_pinecone=config.USE_PINECONE
        )
        # 벡터스토어 저장
        vectorstore.save_local(vectorstore_path_by_clip)

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
