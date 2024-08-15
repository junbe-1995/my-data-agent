from fastapi import APIRouter
from .agent.router import router as query_router


# set main route
router = APIRouter(prefix="/api")

router.include_router(query_router)
