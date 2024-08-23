import io
from PIL import Image
from fastapi import Depends, UploadFile, File, Form

from .model import AgentQueryResponse
from .service import AgentService


async def get_agent_query_answer(
    deviceId: str = Form(...),
    query: str = Form(...),
    imageFile: UploadFile = File(None),
    service: AgentService = Depends(AgentService),
) -> AgentQueryResponse:
    """
    MyData RAG Agent Query API.

    입력 쿼리 및 이미지 파일을 기반으로 retrieval 된 document를 바탕으로 생성한 답변을 반환합니다.

    Args:

        deviceId (str):
            사용자를 구분할 수 있는 고유한 필드로, 해당 아이디를 기준으로 히스토리가 생성 및 관리됩니다.

        query (str):
            사용자의 쿼리 텍스트 필드입니다.
            이 필드는 사용자가 알고자 하는 정보나 질문을 포함합니다.

        imageFile (UploadFile, optional):
            멀티 모달 검색에 활용될 이미지 파일 필드입니다.
            이미지 파일이 제공된 경우, 이미지의 시각적 특징을 기반으로 매핑되는 의미적 특징을 지닌 도큐먼트를 추가적으로 추출하여 함께 처리됩니다.
            * pdf 파일의 추출된 텍스트의 의미적 특징에 매핑되는 시각적 특징을 지닌 이미지만 유효하게 사용됩니다.

    Returns:

        AgentQueryResponse: 생성된 답변이 포함된 응답 객체입니다.
    """

    image = None
    if imageFile is not None:
        image_bytes = await imageFile.read()
        image = Image.open(io.BytesIO(image_bytes))

    return await service.get_agent_query_answer(
        device_id=deviceId, query=query, image=image
    )
