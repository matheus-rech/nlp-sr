"""
API v1 router aggregation
"""
from fastapi import APIRouter

from app.api.api_v1.endpoints import (
    auth,
    projects,
    citations,
    screening,
    export,
    health
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(citations.router, prefix="/citations", tags=["citations"])
api_router.include_router(screening.router, prefix="/screening", tags=["screening"])
api_router.include_router(export.router, prefix="/export", tags=["export"])