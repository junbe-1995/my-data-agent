import io

from PIL import Image
from fastapi import Depends, UploadFile, File, Form

from .model import AgentQueryResponse
from .service import AgentService


async def get_agent_query_answer(
    deviceId: str = Form(...),
    query: str = Form(...),
    file: UploadFile = File(None),
    service: AgentService = Depends(AgentService),
) -> AgentQueryResponse:
    image = None
    if file is not None:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

    return await service.get_agent_query_answer(
        device_id=deviceId, query=query, image=image
    )
