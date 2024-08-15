from fastapi import Depends

from .model import AgentQueryResponse, AgentQueryRequest
from .service import AgentService


async def get_agent_query_answer(
    request: AgentQueryRequest, service: AgentService = Depends(AgentService)
) -> AgentQueryResponse:
    return await service.get_agent_query_answer(request.query)
