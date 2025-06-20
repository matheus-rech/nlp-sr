"""
Database models for Otto-SR
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, Integer, JSON, Float, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class User(Base):
    """User model for authentication"""
    __tablename__ = "users"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255))
    hashed_password = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    projects = relationship("Project", back_populates="owner")
    activity_logs = relationship("ActivityLog", back_populates="user")


class Project(Base):
    """Systematic review project"""
    __tablename__ = "projects"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    criteria = Column(JSON)  # PICO-TT criteria
    owner_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    screening_mode = Column(String(50), default="dual")  # single, dual
    status = Column(String(50), default="active")  # active, completed, archived
    
    # Relationships
    owner = relationship("User", back_populates="projects")
    citations = relationship("Citation", back_populates="project", cascade="all, delete-orphan")
    screening_results = relationship("ScreeningResult", back_populates="project", cascade="all, delete-orphan")
    activity_logs = relationship("ActivityLog", back_populates="project", cascade="all, delete-orphan")


class Citation(Base):
    """Citation/paper to be screened"""
    __tablename__ = "citations"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(PG_UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)
    title = Column(Text, nullable=False)
    authors = Column(Text)
    journal = Column(Text)
    year = Column(Integer)
    abstract = Column(Text)
    doi = Column(String(255), index=True)
    pmid = Column(String(50))
    keywords = Column(Text)
    full_text_url = Column(Text)
    relevance_score = Column(Float, default=0.5)
    upload_batch_id = Column(String(100), index=True)  # Track upload batches
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="citations")
    screening_result = relationship("ScreeningResult", back_populates="citation", uselist=False)


class ScreeningResult(Base):
    """Results from AI screening"""
    __tablename__ = "screening_results"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    citation_id = Column(PG_UUID(as_uuid=True), ForeignKey("citations.id"), nullable=False, index=True)
    project_id = Column(PG_UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)
    
    # AI Results
    ai1_result = Column(JSON)  # Conservative reviewer result
    ai2_result = Column(JSON)  # Liberal reviewer result
    
    # Consensus
    consensus = Column(String(50))  # agree_include, agree_exclude, dispute
    final_decision = Column(String(50))  # include, exclude, uncertain
    confidence_score = Column(Float)
    
    # Human Review
    human_decision = Column(String(50))  # include, exclude
    human_notes = Column(Text)
    reviewed_by_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"))
    reviewed_at = Column(DateTime)
    
    # Metadata
    screening_time_ms = Column(Integer)  # Time taken for AI screening
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    citation = relationship("Citation", back_populates="screening_result")
    project = relationship("Project", back_populates="screening_results")
    reviewer = relationship("User", foreign_keys=[reviewed_by_id])


class ActivityLog(Base):
    """Audit trail for all actions"""
    __tablename__ = "activity_logs"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(PG_UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"))
    action = Column(String(255), nullable=False)  # upload, screen, review, export, etc.
    details = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    project = relationship("Project", back_populates="activity_logs")
    user = relationship("User", back_populates="activity_logs")