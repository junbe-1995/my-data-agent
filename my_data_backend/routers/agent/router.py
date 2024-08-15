from fastapi import APIRouter
from .query.controller import get_agent_query_answer

router = APIRouter(prefix="/agent", tags=["My Data Agent"])

router.add_api_route("/query", get_agent_query_answer, methods=["post"])
