"""
Configuration settings for Otto-SR backend
"""
from typing import List, Optional, Union
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, field_validator
import os


class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Otto-SR"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://ottosr_user:secure_password@localhost/ottosr"
    )
    
    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    
    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    
    # Advanced LLM Model Settings
    PRIMARY_MODEL: str = "gpt-4.1"  # Ultra-precise screening
    REASONING_MODEL: str = "o3-pro"  # Deep reasoning for conflicts
    RAPID_MODEL: str = "gemini-2.5-flash-lite"  # Fast initial screening
    
    # Legacy model settings (for backward compatibility)
    CONSERVATIVE_MODEL: str = "gpt-4o"
    LIBERAL_MODEL: str = "gpt-4o-mini"
    DEFAULT_TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 4000
    
    # Advanced screening settings
    USE_TRIPLE_LLM: bool = os.getenv("USE_TRIPLE_LLM", "true").lower() == "true"
    REASONING_EFFORT: str = "high"  # For O3 model
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = [".ris", ".xml", ".enw", ".bib", ".rdf", ".json", ".csv", ".txt"]
    
    # Screening Configuration
    BATCH_SIZE: int = 10  # Number of citations to process in parallel
    SCREENING_TIMEOUT: int = 300  # 5 minutes per batch
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()