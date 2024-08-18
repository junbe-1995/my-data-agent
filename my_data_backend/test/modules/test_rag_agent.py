import os

import pytest

from my_data_backend.modules.prompt_template import PromptTemplateSingleton
from my_data_backend.modules.rag_agent import RAGAgentSingleton
from my_data_backend.modules.vector_store import load_vectorstore


@pytest.fixture
def rag_agent():
    return RAGAgentSingleton()


@pytest.fixture
def vectorstore():
    vectorstore_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "vector_store/vectorstore.faiss",
    )
    return load_vectorstore(vectorstore_path, use_pinecone=False)


@pytest.mark.asyncio
async def test_initialize(rag_agent, vectorstore):
    # prompt template initialize
    PromptTemplateSingleton().initialize()

    rag_agent.initialize(vectorstore, None)
    assert rag_agent.vectorstore is not None
    assert rag_agent.llm is not None


@pytest.mark.asyncio
async def test_inference_with_query(rag_agent, vectorstore):
    # prompt template initialize
    PromptTemplateSingleton().initialize()

    rag_agent.initialize(vectorstore, None)
    device_id = "test_device"
    query = "API 스펙 중 aNS는 어떤 것을 뜻하나요?"

    response = await rag_agent.inference(device_id, query=query)
    assert isinstance(response, str)
