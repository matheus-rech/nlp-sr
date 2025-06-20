"""
Health check endpoints
"""
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.database import get_db
from app.core.config import settings

router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "otto-sr-backend",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.0.0"
    }


@router.get("/db")
async def database_health(db: Session = Depends(get_db)):
    """Database connectivity check"""
    try:
        # Execute a simple query
        result = db.execute(text("SELECT 1"))
        result.scalar()
        
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/config")
async def config_check():
    """Configuration check (non-sensitive info only)"""
    return {
        "environment": settings.ENVIRONMENT,
        "cors_enabled": bool(settings.BACKEND_CORS_ORIGINS),
        "max_upload_size_mb": settings.MAX_UPLOAD_SIZE / (1024 * 1024),
        "allowed_file_types": settings.ALLOWED_EXTENSIONS,
        "batch_size": settings.BATCH_SIZE,
        "llm_providers_configured": {
            "openai": bool(settings.OPENAI_API_KEY),
            "anthropic": bool(settings.ANTHROPIC_API_KEY)
        },
        "timestamp": datetime.utcnow().isoformat()
    }