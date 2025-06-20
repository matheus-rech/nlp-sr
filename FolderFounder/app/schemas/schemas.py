"""
Pydantic schemas for request/response validation
"""
from datetime import datetime
from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID


# User schemas
class UserBase(BaseModel):
    email: str
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str


class UserUpdate(UserBase):
    password: Optional[str] = None


class UserInDB(UserBase):
    id: UUID
    is_active: bool
    is_superuser: bool
    created_at: datetime


class User(UserInDB):
    model_config = ConfigDict(from_attributes=True)


# Auth schemas
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    user_id: Optional[str] = None


# Project schemas
class ProjectCriteria(BaseModel):
    """PICO-TT criteria for systematic review"""
    population: str = ""
    intervention: str = ""
    comparison: str = ""
    outcome: str = ""
    timeframe: str = ""
    study_types: str = ""
    
    # Inclusion criteria
    inclusion_language: str = ""
    inclusion_publication: str = ""
    inclusion_sample_size: str = ""
    inclusion_data_availability: str = ""
    other_inclusion: str = ""
    
    # Exclusion criteria
    exclusion_study_types: str = ""
    exclusion_populations: str = ""
    exclusion_interventions: str = ""
    exclusion_languages: str = ""
    other_exclusion: str = ""
    
    research_question: str = ""


class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    screening_mode: Literal["single", "dual"] = "dual"


class ProjectCreate(ProjectBase):
    criteria: Optional[ProjectCriteria] = None


class ProjectUpdate(ProjectBase):
    name: Optional[str] = None
    criteria: Optional[ProjectCriteria] = None
    status: Optional[Literal["active", "completed", "archived"]] = None


class ProjectInDB(ProjectBase):
    id: UUID
    owner_id: UUID
    criteria: Optional[Dict[str, Any]] = None
    status: str
    created_at: datetime
    updated_at: datetime
    
    # Statistics
    total_citations: int = 0
    screened_citations: int = 0
    included_citations: int = 0
    excluded_citations: int = 0
    conflict_citations: int = 0


class Project(ProjectInDB):
    model_config = ConfigDict(from_attributes=True)


# Citation schemas
class CitationBase(BaseModel):
    title: str
    authors: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    keywords: Optional[str] = None
    full_text_url: Optional[str] = None


class CitationCreate(CitationBase):
    project_id: UUID
    upload_batch_id: Optional[str] = None


class CitationInDB(CitationBase):
    id: UUID
    project_id: UUID
    relevance_score: float
    created_at: datetime


class Citation(CitationInDB):
    screening_result: Optional['ScreeningResult'] = None
    model_config = ConfigDict(from_attributes=True)


# Screening schemas
class ScreeningDecision(BaseModel):
    """Structured output from LLM screening"""
    decision: Literal["include", "exclude", "uncertain"] = Field(
        description="Final decision: include, exclude, or uncertain"
    )
    confidence: float = Field(
        ge=0, le=100,
        description="Confidence percentage from 0-100"
    )
    reasoning: str = Field(
        description="Detailed explanation of the decision"
    )
    evidence_quotes: List[str] = Field(
        default_factory=list,
        description="Specific quotes from the citation supporting the decision"
    )
    pico_assessment: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Assessment of PICO elements"
    )
    quality_indicators: Dict[str, bool] = Field(
        default_factory=dict,
        description="Quality assessment indicators"
    )


class ScreeningResultBase(BaseModel):
    citation_id: UUID
    project_id: UUID


class ScreeningResultCreate(ScreeningResultBase):
    pass


class ScreeningResultUpdate(BaseModel):
    human_decision: Optional[Literal["include", "exclude"]] = None
    human_notes: Optional[str] = None


class ScreeningResultInDB(ScreeningResultBase):
    id: UUID
    ai1_result: Optional[Dict[str, Any]] = None
    ai2_result: Optional[Dict[str, Any]] = None
    consensus: Optional[str] = None
    final_decision: Optional[str] = None
    confidence_score: Optional[float] = None
    human_decision: Optional[str] = None
    human_notes: Optional[str] = None
    reviewed_by_id: Optional[UUID] = None
    reviewed_at: Optional[datetime] = None
    screening_time_ms: Optional[int] = None
    created_at: datetime
    updated_at: datetime


class ScreeningResult(ScreeningResultInDB):
    model_config = ConfigDict(from_attributes=True)


# File upload schemas
class FileUploadResponse(BaseModel):
    project_id: UUID
    citations_count: int
    upload_batch_id: str
    message: str


# Screening job schemas
class ScreeningJobRequest(BaseModel):
    criteria: ProjectCriteria
    batch_size: int = 10
    use_cache: bool = True


class ScreeningJobResponse(BaseModel):
    job_id: str
    project_id: UUID
    total_citations: int
    message: str


class ScreeningProgress(BaseModel):
    job_id: str
    project_id: UUID
    total: int
    processed: int
    completed: int
    errors: int
    conflicts: int
    status: Literal["running", "completed", "failed"]


# Export schemas
class ExportRequest(BaseModel):
    format: Literal["csv", "json", "ris", "bibtex", "excel"]
    include_only: bool = False
    include_screening_details: bool = True


# Activity log schemas
class ActivityLogEntry(BaseModel):
    action: str
    details: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[UUID] = None
    project_id: UUID