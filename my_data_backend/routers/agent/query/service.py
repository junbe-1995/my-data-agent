from my_data_backend.modules.rag_agent import RAGAgentSingleton
from my_data_backend.routers.agent.query.model import AgentQueryResponse


class AgentService:
    def __init__(self):
        self.rag_agent = RAGAgentSingleton()

    async def get_agent_query_answer(self, query: str) -> AgentQueryResponse:
        # custom_prompt = (
        #     "You are an expert assistant specialized in API documentation and guidelines. "
        #     "The following information has been retrieved from a knowledge base containing API documentation and guidelines:\n\n"
        #     "{context}\n\n"
        #     "Based on the above information, please provide a detailed and comprehensive answer to the following question:\n"
        #     "{question}\n"
        #     "Make sure your answer is accurate and references the relevant API documentation or guidelines where applicable."
        # )
        #
        # self.rag_agent.set_custom_prompt(custom_prompt)
        answer = self.rag_agent.inference(query=query)
        return AgentQueryResponse(answer=answer)
