"""
Authentication endpoints (placeholder for now)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.schemas import Token, UserCreate, User

router = APIRouter()


@router.post("/register", response_model=User)
async def register(
    user: UserCreate,
    db: Session = Depends(get_db)
):
    """Register a new user"""
    # TODO: Implement user registration
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.post("/login", response_model=Token)
async def login(
    username: str,
    password: str,
    db: Session = Depends(get_db)
):
    """Login and get access token"""
    # TODO: Implement login
    raise HTTPException(status_code=501, detail="Not implemented yet")