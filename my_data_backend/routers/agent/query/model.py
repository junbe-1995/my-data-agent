from pydantic import BaseModel


class AgentQueryRequest(BaseModel):
    deviceId: str
    query: str


class AgentQueryResponse(BaseModel):
    answer: str
