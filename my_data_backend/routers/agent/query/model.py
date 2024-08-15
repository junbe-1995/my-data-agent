from pydantic import BaseModel


class AgentQueryRequest(BaseModel):
    query: str


class AgentQueryResponse(BaseModel):
    answer: str
