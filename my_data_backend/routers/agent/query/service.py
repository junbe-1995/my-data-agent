from my_data_backend.modules.rag_agent import RAGAgentSingleton
from my_data_backend.routers.agent.query.model import AgentQueryResponse


class AgentService:
    def __init__(self):
        self.rag_agent = RAGAgentSingleton()

    async def get_agent_query_answer(
        self, device_id: str, query: str
    ) -> AgentQueryResponse:
        answer = self.rag_agent.inference(device_id=device_id, query=query)
        return AgentQueryResponse(answer=answer)
