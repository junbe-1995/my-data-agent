# My Data Agent

## 프로젝트 구조

- LangChain 기반의 RAG Agent 를 Fast API 기반의 RESTful API 서버로 구현한 프로젝트입니다.
- 전체 구조는 아래 LangChain의 구조를 참고하여 구현되었습니다.
![img.png](img.png)
* 출처: LangChain

---

## API docs
- 실행 환경의 로컬에서 9001번 포트로 서버가 실행됩니다.
- api docs 는 [/api/docs](http://localhost:9001/api/docs) 로 접근 가능하며, 아래와 같이 확인 가능합니다.
- path: /api/agent/query
<img width="1492" alt="image" src="https://github.com/user-attachments/assets/cf050ec2-c7ff-4e58-9f21-4d8e0b32ea30">



---

## 주요 기능 및 구현 상세
- LangChain 활용
```python
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS, Pinecone
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationSummaryMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
```
- 멀티턴 에이전트
  - 히스토리 매니저 싱글톤 모듈:
    - 멀티턴 대화에서의 맥락을 유지하기 위해 히스토리 매니저를 싱글톤 패턴으로 구현했습니다.
    - 이를 통해 동일한 디바이스 아이디로부터 들어온 요청에 대해 대화 히스토리를 관리하고, 이를 바탕으로 답변을 생성합니다.
- 멀티모달 지원
  - 이미지 쿼리 처리:
    - 이미지가 쿼리로 들어올 경우, CLIP 모델을 사용해 이미지를 임베딩하고, 이를 바탕으로 추가적인 문서 리트리벌을 수행한 뒤, 결과를 함께 반환합니다.
    - 단, 현재 이미지 벡터스토어는 pdf파일의 텍스트를 기반으로 생성되어 있어 해당 텍스트의 의미적 특징에 매핑되는 시각적 특징을 지닌 이미지만 유효한 결과를 얻을 수 있습니다.
- 확장 가능한 프로젝트 구조
  - 모듈화 및 싱글톤 패턴 사용:
    - 프로젝트에 필요한 기능들을 모듈화하여, 각 모듈이 독립적으로 관리되고 수정될 수 있도록 설계했습니다.
    - 환경 변수 관리:
      - 로컬 벡터스토어(FAISS)와 클라우드 기반 벡터스토어(Pinecone)를 쉽게 전환할 수 있도록 했습니다.
      - LLM의 맥스 토큰 설정, 벡터스토어에서 검색할 top K 개수 등의 설정을 환경 변수로 관리할 수 있게 했습니다.
    - 의존성 주입:
      - 예를 들어, 임베딩 모델과 벡터스토어를 쉽게 교체하여 사용할 수 있도록 app.py에서 의존성을 주입하는 방식으로 유연하게 구성했습니다.
- 동시성 처리
  - AsyncIO 락 사용:
    - 동일 디바이스 아이디에서 동시에 여러 요청이 들어올 때, 히스토리의 일관성을 유지하기 위해 AsyncIO의 Lock을 활용하여 동시성 문제를 해결했습니다. 
    - 이를 통해 히스토리 순서가 섞이는 문제를 방지합니다.
- 모델 토큰 제한 해결
  - ConversationSummaryMemory 사용:
    - 대화 히스토리를 관리하기 위해 ConversationSummaryMemory를 활용했습니다. 
    - 해당 클래스는 대화의 중요 부분을 요약하여 저장하므로, 모델의 토큰 제한을 완화할 수 있습니다.

---

## project setting 방법
```

1. 가상환경 생성

    $ ./scripts/venv.create.sh

2. interpreter 설정

    path: my-data-agent/.venv/bin/python

3. 필수 환경 변수 파일 (.env) 생성
    
    my_data_backend 디렉토리에 .env 파일 생성 후 아래 필수 필드 추가
    
        COHERE_API_KEY=your_cohere_api_key
        OPENAI_API_KEY=your_openai_api_key
   
4. 간단한 동작 확인 및 디버깅 시 루트 디렉토리의 run_my_data_backend.py 실행

    $ python run_my_data_backend.py

5. 단, 운영 환경 혹은 비슷한 환경에서 테스트가 필요하다면,
   my_data_backend에 있는 gunicorn.conf.py 파일을 통해 멀티 프로세스로 띄우는 것을 권장합니다.
    
    $ gunicorn -c my_data_backend/gunicorn.conf.py
    
    # objc fork safety 관련 에러 발생 시 아래와 같이 실행 합니다.
    $ OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES gunicorn -c my_data_backend/gunicorn.conf.py

```

---

## 참고 사항
```
사용 모델은 아래와 같으며, 환경변수로 관리중입니다.

LLM : gpt-4o-mini
Embedding : embed-multilingual-v3.0 / CLIP (openai/clip-vit-base-patch16)

```
- vector store 는 로컬에 저장되어 있으며
  - mydata_backend/vector_store
- 이 후 Pinecone으로 저장소 이전 시 config 에서 환경 변수를 변경하면 됩니다.
    - USE_PINECONE = True
    - PINECONE_ 관련 변수들 설정 필요
- 현재 vector store는 
  - cohere의 embed-multilingual-v3.0 모델과
  - CLIP (openai/clip-vit-base-patch16) 모델로
  - 이원화하여 관리중입니다.
    - 사용자의 텍스트 쿼리는 cohere의 embed-multilingual-v3.0 모델을 기반으로 document retrieval을 수행하며,
    - 사용자의 이미지 쿼리는 CLIP 모델을 기반으로 document retrieval을 수행합니다.
