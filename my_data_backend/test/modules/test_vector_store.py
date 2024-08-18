import os

import pytest

from my_data_backend.modules.vector_store import load_vectorstore


@pytest.mark.asyncio
def test_load_vectorstore_singleton(documents, tmpdir):
    # FAISS vector store를 로드하고 문서 검색을 수행하는 테스트

    # 1. vector store load
    vectorstore_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "vector_store/vectorstore.faiss",
    )

    vectorstore_loaded = load_vectorstore(vectorstore_path, use_pinecone=False)
    assert vectorstore_loaded is not None

    # 2. 검색 쿼리
    query = "API 스펙 중 aNS는 어떤 것을 뜻하나요?"
    search_results = vectorstore_loaded.similarity_search(query)

    # 3. 검색 결과 확인
    assert len(search_results) > 0  # 결과가 존재하는지 확인
    for result in search_results:
        print(result.page_content)  # 검색된 문서 내용을 출력

    # 테스트 통과 여부 확인
    assert search_results[0].page_content is not None
