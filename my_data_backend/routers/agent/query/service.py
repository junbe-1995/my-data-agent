from PIL import Image
from my_data_backend.modules.rag_agent import RAGAgentSingleton
from my_data_backend.routers.agent.query.model import AgentQueryResponse


class AgentService:
    def __init__(self):
        self.rag_agent = RAGAgentSingleton()

    async def get_agent_query_answer(
        self, device_id: str, query: str = None, image: Image.Image = None
    ) -> AgentQueryResponse:
        # 인퍼런스 메서드에 query와 image를 모두 넘깁니다.
        answer = await self.rag_agent.inference(
            device_id=device_id, query=query, image=image
        )
        return AgentQueryResponse(answer=answer)
