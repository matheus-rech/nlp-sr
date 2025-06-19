# ==============================================================================
#
# Otto-SR: Production LLM Screening Tool v3.0
#
# Complete single-file application with advanced features:
# - Multiple LLM provider support (OpenAI, Claude, Ollama, LM Studio)
# - Real-time collaborative screening
# - Advanced PICO-TT criteria configuration
# - Batch processing and AI-assisted modes
# - Progress tracking and activity logging
#
# ==============================================================================

import os
import json
import asyncio
import uuid
import xml.etree.ElementTree as ET
import re
from datetime import datetime
from typing import List, Dict, Optional, Any, Literal, Union

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, JSON, Float
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel, Field

# --- 1. Configuration & Initialization ---

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/ottosr")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI(
    title="Otto-SR Production Tool v3.0",
    description="Advanced systematic review screening with multiple LLM providers and real-time collaboration",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

# --- 2. Enhanced Database Models ---

class Project(Base):
    __tablename__ = "projects"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    criteria = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    screening_mode = Column(String(50), default="single")

class CitationRecord(Base):
    __tablename__ = "citations"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(PG_UUID(as_uuid=True), nullable=False, index=True)
    title = Column(Text, nullable=False)
    authors = Column(Text)
    journal = Column(Text)
    year = Column(Integer)
    abstract = Column(Text)
    file_content = Column(Text, nullable=True)
    doi = Column(Text, nullable=True)
    keywords = Column(Text, nullable=True)
    relevance_score = Column(Float, default=0.5)

class ScreeningResult(Base):
    __tablename__ = "screening_results"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    citation_id = Column(PG_UUID(as_uuid=True), nullable=False, index=True)
    project_id = Column(PG_UUID(as_uuid=True), nullable=False, index=True)
    job_id = Column(String, index=True)
    status = Column(String, default="pending")
    ai1_result = Column(JSON)  # Conservative AI result
    ai2_result = Column(JSON)  # Pragmatic AI result
    final_decision = Column(String)
    confidence_score = Column(Float)
    human_decision = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

class ActivityLog(Base):
    __tablename__ = "activity_logs"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(PG_UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(String(100), default="system")
    action = Column(String(255))
    details = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 3. Enhanced Pydantic Models ---

class AdvancedScreeningCriteria(BaseModel):
    population: str = ""
    intervention: str = ""
    comparison: str = ""
    outcome: str = ""
    timeframe: str = ""
    studyTypes: str = ""
    inclusionLanguage: str = ""
    inclusionPublication: str = ""
    inclusionSampleSize: str = ""
    inclusionDataAvailability: str = ""
    otherInclusion: str = ""
    exclusionStudyTypes: str = ""
    exclusionPopulations: str = ""
    exclusionInterventions: str = ""
    exclusionLanguages: str = ""
    otherExclusion: str = ""
    researchQuestion: str = ""

# --- Enhanced LLM Provider Support ---

class ProviderConfig(BaseModel):
    name: str
    display_name: str
    models: List[str]
    default_model: str
    requires_api_key: bool
    default_endpoint: Optional[str] = None
    supports_streaming: bool = True

# LLM Provider Configurations - Trusted and Tested Providers
LLM_PROVIDERS = {
    # Tier 1: Most Reliable for Production Screening
    "openai": ProviderConfig(
        name="openai",
        display_name="OpenAI (Recommended)",
        models=["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        default_model="gpt-4o",
        requires_api_key=True,
        supports_streaming=True
    ),
    "anthropic": ProviderConfig(
        name="anthropic",
        display_name="Anthropic Claude (Recommended)",
        models=["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        default_model="claude-3-5-sonnet-20241022",
        requires_api_key=True,
        supports_streaming=True
    ),
    
    # Tier 2: Local/Self-Hosted Options
    "ollama": ProviderConfig(
        name="ollama",
        display_name="Ollama (Local)",
        models=["llama3.1:8b", "llama3.1:70b", "llama3:70b", "mistral:7b", "phi3:medium", "qwen2.5:7b"],
        default_model="llama3.1:8b",
        requires_api_key=False,
        default_endpoint="http://localhost:11434",
        supports_streaming=True
    ),
    "openai_compatible": ProviderConfig(
        name="openai_compatible",
        display_name="OpenAI-Compatible (Local/Custom)",
        models=["llama-3.1-8b-instruct", "llama-3.1-70b-instruct", "mistral-7b-instruct", "qwen2.5-7b-instruct"],
        default_model="llama-3.1-8b-instruct",
        requires_api_key=False,
        default_endpoint="http://localhost:1234/v1",
        supports_streaming=True
    ),
    
    # Tier 3: Alternative Commercial Providers
    "cohere": ProviderConfig(
        name="cohere",
        display_name="Cohere",
        models=["command-r-plus", "command-r", "command-nightly"],
        default_model="command-r-plus",
        requires_api_key=True,
        supports_streaming=True
    ),
    "groq": ProviderConfig(
        name="groq",
        display_name="Groq (Fast Inference)",
        models=["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        default_model="llama-3.1-70b-versatile",
        requires_api_key=True,
        default_endpoint="https://api.groq.com/openai/v1",
        supports_streaming=True
    ),
    "together": ProviderConfig(
        name="together",
        display_name="Together AI",
        models=["meta-llama/Llama-3-8b-chat-hf", "meta-llama/Llama-3-70b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
        default_model="meta-llama/Llama-3-70b-chat-hf",
        requires_api_key=True,
        default_endpoint="https://api.together.xyz/v1",
        supports_streaming=True
    ),
    
    # Provider Agnostic Option
    "custom": ProviderConfig(
        name="custom",
        display_name="Custom Provider (Advanced)",
        models=["custom-model-1", "custom-model-2", "custom-model-3"],
        default_model="custom-model-1",
        requires_api_key=True,
        default_endpoint="https://your-api-endpoint.com/v1",
        supports_streaming=True
    )
}

class LLMConfig(BaseModel):
    provider: str
    model: str
    endpoint: str
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2000

# Structured Output Schema with Pydantic
class ScreeningDecision(LangChainBaseModel):
    """Structured screening decision output"""
    decision: Literal["include", "exclude"] = Field(
        description="Final decision: 'include' to include the study, 'exclude' to exclude it"
    )
    confidence: float = Field(
        ge=0, le=100, 
        description="Confidence percentage from 0-100"
    )
    reasoning: str = Field(
        description="Detailed explanation of the decision with specific criteria references"
    )
    pico_scores: Dict[str, float] = Field(
        description="PICO component scores from 0.0-1.0",
        default_factory=lambda: {
            "population": 0.5,
            "intervention": 0.5, 
            "comparison": 0.5,
            "outcome": 0.5
        }
    )
    study_design: str = Field(
        description="Identified study design (e.g., RCT, cohort study, systematic review)"
    )
    quality_assessment: Literal["high", "medium", "low", "unclear"] = Field(
        description="Overall study quality assessment"
    )
    key_findings: List[str] = Field(
        description="Key findings or limitations identified",
        default_factory=list
    )

class CitationUploadResponse(BaseModel):
    project_id: str
    citations_count: int
    message: str

# --- 4. Advanced File Parsing ---

def parse_ris_file(content: str) -> List[Dict[str, Any]]:
    """Enhanced RIS parser with better field extraction"""
    citations = []
    current_citation: Dict[str, Any] = {}
    
    lines = content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('TY  -'):
            if current_citation:
                citations.append(current_citation)
            current_citation = {'relevance_score': 0.5}
        elif line.startswith('TI  -'):
            current_citation['title'] = line[6:].strip()
        elif line.startswith('AU  -'):
            if 'authors' not in current_citation:
                current_citation['authors'] = []
            current_citation['authors'].append(line[6:].strip())
        elif line.startswith('JO  -') or line.startswith('JF  -'):
            current_citation['journal'] = line[6:].strip()
        elif line.startswith('PY  -'):
            try:
                current_citation['year'] = int(line[6:].strip()[:4])
            except:
                pass
        elif line.startswith('AB  -'):
            current_citation['abstract'] = line[6:].strip()
        elif line.startswith('DO  -'):
            current_citation['doi'] = line[6:].strip()
        elif line.startswith('KW  -'):
            if 'keywords' not in current_citation:
                current_citation['keywords'] = []
            current_citation['keywords'].append(line[6:].strip())
        elif line.startswith('ER  -'):
            if current_citation:
                citations.append(current_citation)
                current_citation = {}
    
    if current_citation:
        citations.append(current_citation)
    
    # Process and clean citations
    for citation in citations:
        if 'authors' in citation and isinstance(citation['authors'], list):
            citation['authors'] = '; '.join(citation['authors'])
        if 'keywords' in citation and isinstance(citation['keywords'], list):
            citation['keywords'] = '; '.join(citation['keywords'])
        
        # Calculate basic relevance score based on content completeness
        score = 0.3  # Base score
        if citation.get('title'): score += 0.2
        if citation.get('abstract'): score += 0.3
        if citation.get('authors'): score += 0.1
        if citation.get('year'): score += 0.1
        citation['relevance_score'] = min(score, 1.0)
    
    return citations

def parse_xml_file(content: str) -> List[Dict[str, Any]]:
    """Enhanced XML parser supporting multiple formats"""
    citations = []
    try:
        root = ET.fromstring(content)
        
        # Try different XML structures
        records = (root.findall('.//record') or 
                  root.findall('.//citation') or 
                  root.findall('.//item') or
                  root.findall('.//reference'))
        
        for record in records:
            citation: Dict[str, Any] = {'relevance_score': 0.5}
            
            # Extract title
            title_elem = (record.find('.//title') or 
                         record.find('.//article-title') or
                         record.find('.//primary-title'))
            if title_elem is not None and title_elem.text:
                citation['title'] = title_elem.text
            
            # Extract authors
            authors = []
            for author in (record.findall('.//author') or 
                          record.findall('.//name') or
                          record.findall('.//contributor')):
                author_text = author.text if author.text else ''
                if not author_text:
                    given = author.find('.//given-names')
                    surname = author.find('.//surname')
                    if given is not None and surname is not None and given.text and surname.text:
                        author_text = f"{given.text} {surname.text}"
                if author_text:
                    authors.append(author_text)
            
            if authors:
                citation['authors'] = '; '.join(authors)
            
            # Extract other fields
            journal_elem = (record.find('.//journal') or 
                           record.find('.//source') or
                           record.find('.//secondary-title'))
            if journal_elem is not None and journal_elem.text:
                citation['journal'] = journal_elem.text
            
            year_elem = (record.find('.//year') or 
                        record.find('.//pub-date') or
                        record.find('.//date'))
            if year_elem is not None and year_elem.text:
                try:
                    year_match = re.search(r'\d{4}', year_elem.text)
                    if year_match:
                        citation['year'] = int(year_match.group())
                except:
                    pass
            
            abstract_elem = record.find('.//abstract')
            if abstract_elem is not None and abstract_elem.text:
                citation['abstract'] = abstract_elem.text
            
            doi_elem = record.find('.//doi')
            if doi_elem is not None and doi_elem.text:
                citation['doi'] = doi_elem.text
            
            # Calculate relevance score
            score = 0.3
            if citation.get('title'): score += 0.2
            if citation.get('abstract'): score += 0.3
            if citation.get('authors'): score += 0.1
            if citation.get('year'): score += 0.1
            citation['relevance_score'] = min(score, 1.0)
            
            if citation.get('title'):
                citations.append(citation)
                
    except ET.ParseError as e:
        raise HTTPException(status_code=400, detail=f"Invalid XML format: {str(e)}")
    
    return citations

# --- 5. Dynamic LLM Provider Factory ---

class LLMProviderFactory:
    """Factory for creating LLM instances across different providers"""
    
    @staticmethod
    def create_llm(config: LLMConfig):
        """Create LLM instance based on provider configuration"""
        provider = config.provider.lower()
        
        if provider == "openai":
            return ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=config.api_key or os.getenv("OPENAI_API_KEY")
            )
        elif provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY")
                )
            except ImportError:
                raise ImportError("langchain_anthropic not installed. Install with: pip install langchain-anthropic")
        elif provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
                return ChatOllama(
                    model=config.model,
                    temperature=config.temperature,
                    base_url=config.endpoint or "http://localhost:11434"
                )
            except ImportError:
                raise ImportError("langchain_ollama not installed. Install with: pip install langchain-ollama")
        elif provider == "cohere":
            try:
                from langchain_cohere import ChatCohere
                return ChatCohere(
                    model=config.model,
                    temperature=config.temperature,
                    cohere_api_key=config.api_key or os.getenv("COHERE_API_KEY")
                )
            except ImportError:
                raise ImportError("langchain_cohere not installed. Install with: pip install langchain-cohere")
        elif provider == "groq":
            # Groq uses OpenAI-compatible API
            return ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                base_url=config.endpoint or "https://api.groq.com/openai/v1",
                api_key=config.api_key or os.getenv("GROQ_API_KEY")
            )
        elif provider == "together":
            # Together AI uses OpenAI-compatible API
            return ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                base_url=config.endpoint or "https://api.together.xyz/v1",
                api_key=config.api_key or os.getenv("TOGETHER_API_KEY")
            )
        elif provider in ["openai_compatible", "custom"]:
            # Generic OpenAI-compatible endpoint for local models and custom providers
            return ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                base_url=config.endpoint,
                api_key=config.api_key or "dummy-key"
            )
        else:
            # Fallback to OpenAI-compatible endpoint
            return ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                base_url=config.endpoint,
                api_key=config.api_key or "dummy-key"
            )

# --- 6. Advanced Prompt Engineering with LangChain ---

class PromptTemplateFactory:
    """Factory for creating dynamic prompt templates"""
    
    @staticmethod
    def create_screening_prompt(strategy: Literal["conservative", "pragmatic"]) -> ChatPromptTemplate:
        """Create LangChain prompt template for screening"""
        
        if strategy == "conservative":
            persona = "Dr. Sarah Chen, a meticulous Cochrane systematic reviewer"
            approach = "conservative, rigorous approach that minimizes false positives"
            decision_rule = "When in doubt, EXCLUDE the study"
            temperature_note = "Be stringent in your evaluation"
        else:
            persona = "Dr. Michael Rodriguez, a pragmatic evidence synthesizer" 
            approach = "inclusive, pragmatic approach that captures potentially relevant evidence"
            decision_rule = "When uncertain, lean toward INCLUSION for human review"
            temperature_note = "Consider broader relevance and transferability"

        system_template = f"""You are {persona} conducting systematic review screening with a {approach}.

EVALUATION APPROACH: {temperature_note}
DECISION RULE: {decision_rule}

Your task is to evaluate research studies against specific inclusion/exclusion criteria and provide structured output.

RESEARCH QUESTION: {{research_question}}

PICO-TT CRITERIA:
{{pico_criteria}}

INCLUSION CRITERIA:
{{inclusion_criteria}}

EXCLUSION CRITERIA:
{{exclusion_criteria}}

Evaluate the study carefully and provide your assessment in the exact JSON format specified in the output schema.
{decision_rule}"""

        human_template = """STUDY TO EVALUATE:
Title: {title}
Authors: {authors}
Journal: {journal} ({year})
Abstract: {abstract}

Please provide your structured evaluation of this study."""

        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])

def create_advanced_prompt(criteria: AdvancedScreeningCriteria, strategy: Literal["conservative", "pragmatic"]) -> ChatPromptTemplate:
    """Create sophisticated prompts based on screening strategy"""
    
    if strategy == "conservative":
        persona = "Dr. Sarah Chen, a meticulous Cochrane systematic reviewer"
        approach = "conservative, rigorous approach that minimizes false positives"
        decision_rule = "When in doubt, EXCLUDE the study"
        temperature_note = "Be stringent in your evaluation"
    else:
        persona = "Dr. Michael Rodriguez, a pragmatic evidence synthesizer"
        approach = "inclusive, pragmatic approach that captures potentially relevant evidence"
        decision_rule = "When uncertain, lean toward INCLUSION for human review"
        temperature_note = "Consider broader relevance and transferability"
    
    # Build comprehensive criteria text
    criteria_sections = []
    if criteria.population:
        criteria_sections.append(f"Population: {criteria.population}")
    if criteria.intervention:
        criteria_sections.append(f"Intervention: {criteria.intervention}")
    if criteria.comparison:
        criteria_sections.append(f"Comparison: {criteria.comparison}")
    if criteria.outcome:
        criteria_sections.append(f"Outcomes: {criteria.outcome}")
    if criteria.timeframe:
        criteria_sections.append(f"Timeframe: {criteria.timeframe}")
    if criteria.studyTypes:
        criteria_sections.append(f"Study Types: {criteria.studyTypes}")
    
    inclusion_criteria = []
    if criteria.inclusionLanguage:
        inclusion_criteria.append(f"Language: {criteria.inclusionLanguage}")
    if criteria.inclusionPublication:
        inclusion_criteria.append(f"Publication: {criteria.inclusionPublication}")
    if criteria.inclusionSampleSize:
        inclusion_criteria.append(f"Sample Size: {criteria.inclusionSampleSize}")
    if criteria.inclusionDataAvailability:
        inclusion_criteria.append(f"Data: {criteria.inclusionDataAvailability}")
    if criteria.otherInclusion:
        inclusion_criteria.append(f"Other: {criteria.otherInclusion}")
    
    exclusion_criteria = []
    if criteria.exclusionStudyTypes:
        exclusion_criteria.append(f"Study Types: {criteria.exclusionStudyTypes}")
    if criteria.exclusionPopulations:
        exclusion_criteria.append(f"Populations: {criteria.exclusionPopulations}")
    if criteria.exclusionInterventions:
        exclusion_criteria.append(f"Interventions: {criteria.exclusionInterventions}")
    if criteria.exclusionLanguages:
        exclusion_criteria.append(f"Languages: {criteria.exclusionLanguages}")
    if criteria.otherExclusion:
        exclusion_criteria.append(f"Other: {criteria.otherExclusion}")
    
    prompt = f"""You are {persona} conducting systematic review screening with a {approach}.

RESEARCH QUESTION: {criteria.researchQuestion}

PICO-TT CRITERIA:
{chr(10).join(criteria_sections) if criteria_sections else "No specific criteria provided"}

INCLUSION CRITERIA:
{chr(10).join(inclusion_criteria) if inclusion_criteria else "Standard inclusion criteria apply"}

EXCLUSION CRITERIA:
{chr(10).join(exclusion_criteria) if exclusion_criteria else "Standard exclusion criteria apply"}

EVALUATION APPROACH: {temperature_note}
DECISION RULE: {decision_rule}

STUDY TO EVALUATE:
Title: {{title}}
Authors: {{authors}}
Journal: {{journal}} ({{year}})
Abstract: {{abstract}}

Provide your evaluation in JSON format:
{{
  "decision": "include" or "exclude",
  "confidence": 0-100,
  "reasoning": "detailed explanation of your decision",
  "pico": {{
    "population_score": 0.0-1.0,
    "intervention_score": 0.0-1.0,
    "comparison_score": 0.0-1.0,
    "outcome_score": 0.0-1.0
  }},
  "study_design": "identified study type",
  "quality_assessment": "High/Medium/Low quality rating"
}}

{decision_rule}"""
    
    return prompt

# --- 6. Enhanced Background Processing ---

async def advanced_screening_task(result_id: str, job_id: str, llm_configs: Dict[str, Any]):
    """Advanced screening with multiple LLM provider support"""
    db = SessionLocal()
    try:
        result = db.query(ScreeningResult).filter(ScreeningResult.id == result_id).first()
        if not result or result.status != 'pending':
            return
        
        result.status = "processing"
        db.commit()
        
        citation = db.query(CitationRecord).filter(CitationRecord.id == result.citation_id).first()
        project = db.query(Project).filter(Project.id == result.project_id).first()
        
        if not citation or not project or not project.criteria:
            raise ValueError("Citation, Project, or Criteria not found")
        
        criteria = AdvancedScreeningCriteria(**project.criteria)
        
        # Create prompt templates for both strategies
        conservative_template = PromptTemplateFactory.create_screening_prompt("conservative")
        pragmatic_template = PromptTemplateFactory.create_screening_prompt("pragmatic")
        
        # Build criteria strings for prompts
        pico_criteria = []
        if criteria.population:
            pico_criteria.append(f"Population: {criteria.population}")
        if criteria.intervention:
            pico_criteria.append(f"Intervention: {criteria.intervention}")
        if criteria.comparison:
            pico_criteria.append(f"Comparison: {criteria.comparison}")
        if criteria.outcome:
            pico_criteria.append(f"Outcomes: {criteria.outcome}")
        if criteria.timeframe:
            pico_criteria.append(f"Timeframe: {criteria.timeframe}")
        if criteria.studyTypes:
            pico_criteria.append(f"Study Types: {criteria.studyTypes}")
        
        inclusion_criteria = []
        if criteria.inclusionLanguage:
            inclusion_criteria.append(f"Language: {criteria.inclusionLanguage}")
        if criteria.inclusionPublication:
            inclusion_criteria.append(f"Publication: {criteria.inclusionPublication}")
        if criteria.inclusionSampleSize:
            inclusion_criteria.append(f"Sample Size: {criteria.inclusionSampleSize}")
        if criteria.inclusionDataAvailability:
            inclusion_criteria.append(f"Data: {criteria.inclusionDataAvailability}")
        if criteria.otherInclusion:
            inclusion_criteria.append(f"Other: {criteria.otherInclusion}")
        
        exclusion_criteria = []
        if criteria.exclusionStudyTypes:
            exclusion_criteria.append(f"Study Types: {criteria.exclusionStudyTypes}")
        if criteria.exclusionPopulations:
            exclusion_criteria.append(f"Populations: {criteria.exclusionPopulations}")
        if criteria.exclusionInterventions:
            exclusion_criteria.append(f"Interventions: {criteria.exclusionInterventions}")
        if criteria.exclusionLanguages:
            exclusion_criteria.append(f"Languages: {criteria.exclusionLanguages}")
        if criteria.otherExclusion:
            exclusion_criteria.append(f"Other: {criteria.otherExclusion}")
        
        # Prepare prompt variables
        prompt_variables = {
            "research_question": criteria.researchQuestion or "Systematic review research question not specified",
            "pico_criteria": "\n".join(pico_criteria) if pico_criteria else "No specific PICO criteria provided",
            "inclusion_criteria": "\n".join(inclusion_criteria) if inclusion_criteria else "Standard inclusion criteria apply",
            "exclusion_criteria": "\n".join(exclusion_criteria) if exclusion_criteria else "Standard exclusion criteria apply",
            "title": citation.title or "",
            "authors": citation.authors or "",
            "journal": citation.journal or "",
            "year": str(citation.year) if citation.year else "",
            "abstract": citation.abstract or ""
        }
        
        # Call AI models with structured prompts
        ai1_config = llm_configs.get('ai1', {})
        ai2_config = llm_configs.get('ai2', {})
        
        ai1_result = await call_llm_api(ai1_config, conservative_template, prompt_variables)
        ai2_result = await call_llm_api(ai2_config, pragmatic_template, prompt_variables)
        
        # Process results
        result.ai1_result = ai1_result
        result.ai2_result = ai2_result
        
        # Determine final decision
        ai1_decision = ai1_result.get('decision', 'exclude')
        ai2_decision = ai2_result.get('decision', 'exclude')
        ai1_confidence = ai1_result.get('confidence', 50)
        ai2_confidence = ai2_result.get('confidence', 50)
        
        if ai1_decision == ai2_decision:
            result.final_decision = ai1_decision
            result.confidence_score = (ai1_confidence + ai2_confidence) / 2
        else:
            result.final_decision = "conflict"
            result.confidence_score = abs(ai1_confidence - ai2_confidence)
        
        result.status = "completed"
        result.processed_at = datetime.utcnow()
        
        # Log activity
        activity = ActivityLog(
            project_id=result.project_id,
            action="citation_screened",
            details={
                "citation_id": str(result.citation_id),
                "decision": result.final_decision,
                "confidence": result.confidence_score
            }
        )
        db.add(activity)
        
    except Exception as e:
        if result:
            result.status = "error"
            result.ai1_result = {"error": str(e)}
            result.ai2_result = {"error": str(e)}
            
            # Log error
            activity = ActivityLog(
                project_id=result.project_id,
                action="screening_error",
                details={"error": str(e), "citation_id": str(result.citation_id)}
            )
            db.add(activity)
        
    finally:
        db.commit()
        db.close()

async def call_llm_api(config: Dict[str, Any], prompt_template: ChatPromptTemplate, prompt_variables: Dict[str, str]) -> Dict[str, Any]:
    """Call LLM APIs with structured output using LangChain and Pydantic"""
    if not config:
        raise ValueError("LLM configuration not provided")
    
    try:
        # Create LLM configuration object
        llm_config = LLMConfig(**config)
        
        # Create LLM instance using factory
        llm = LLMProviderFactory.create_llm(llm_config)
        
        # Create output parser for structured responses
        output_parser = PydanticOutputParser(pydantic_object=ScreeningDecision)
        
        # Create the chain with prompt template, LLM, and output parser
        chain = prompt_template | llm | output_parser
        
        # Add format instructions to prompt variables
        prompt_variables["format_instructions"] = output_parser.get_format_instructions()
        
        # Execute the chain
        try:
            structured_result = await chain.ainvoke(prompt_variables)
            
            # Convert Pydantic model to dict for database storage
            return {
                "decision": structured_result.decision,
                "confidence": float(structured_result.confidence),
                "reasoning": structured_result.reasoning,
                "pico": structured_result.pico_scores,
                "study_design": structured_result.study_design,
                "quality_assessment": structured_result.quality_assessment,
                "key_findings": structured_result.key_findings
            }
            
        except Exception as parse_error:
            # Fallback to raw response parsing if structured parsing fails
            raw_response = await llm.ainvoke(prompt_template.format(**prompt_variables))
            return parse_llm_response(raw_response.content if hasattr(raw_response, 'content') else str(raw_response))
            
    except Exception as e:
        return {
            "decision": "exclude",
            "confidence": 0,
            "reasoning": f"API Error: {str(e)}",
            "pico": {"population": 0, "intervention": 0, "comparison": 0, "outcome": 0},
            "study_design": "Unknown",
            "quality_assessment": "unclear",
            "key_findings": [],
            "error": True
        }

def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Parse LLM response with fallback handling"""
    try:
        # Try to extract JSON from response
        if isinstance(response_text, str):
            # Look for JSON in markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            # Try to parse as JSON
            parsed = json.loads(response_text)
        else:
            parsed = response_text
        
        # Normalize and validate response
        return {
            "decision": parsed.get('decision', 'exclude').lower(),
            "confidence": float(parsed.get('confidence', 50)),
            "reasoning": str(parsed.get('reasoning', 'No reasoning provided')),
            "pico": parsed.get('pico', {
                "population_score": 0.5,
                "intervention_score": 0.5,
                "comparison_score": 0.5,
                "outcome_score": 0.5
            }),
            "study_design": str(parsed.get('study_design', 'Unknown')),
            "quality_assessment": str(parsed.get('quality_assessment', 'Unknown'))
        }
        
    except Exception as e:
        # Fallback parsing for non-JSON responses
        decision = 'exclude'
        if 'include' in response_text.lower():
            decision = 'include'
        
        return {
            "decision": decision,
            "confidence": 50,
            "reasoning": response_text[:200] + "..." if len(response_text) > 200 else response_text,
            "pico": {
                "population_score": 0.5,
                "intervention_score": 0.5,
                "comparison_score": 0.5,
                "outcome_score": 0.5
            },
            "study_design": "Unknown",
            "quality_assessment": "Unknown",
            "parse_error": str(e)
        }

# --- 7. Enhanced Frontend ---

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Otto-SR: Production LLM Screening Tool v3.0</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Progress Bar */
        .progress-bar {
            width: 100%;
            height: 4px;
            background-color: #e9ecef;
            position: relative;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3498db 0%, #2ecc71 100%);
            transition: width 0.3s ease;
            position: relative;
        }

        .progress-text {
            position: absolute;
            top: -20px;
            right: 0;
            font-size: 0.8rem;
            color: #6c757d;
        }

        .header {
            background-color: #2c3e50;
            color: white;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: relative;
        }

        .header h1 {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }

        .header .subtitle {
            font-size: 0.9rem;
            color: #bdc3c7;
            margin-bottom: 1rem;
        }

        /* Screening Mode Selector */
        .screening-mode-selector {
            background-color: #34495e;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }

        .screening-mode-selector h3 {
            color: #ecf0f1;
            margin-bottom: 0.75rem;
            font-size: 1rem;
        }

        .mode-buttons {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .mode-button {
            background-color: #2c3e50;
            color: #bdc3c7;
            border: 2px solid transparent;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9rem;
        }

        .mode-button:hover {
            background-color: #3a526b;
            border-color: #3498db;
        }

        .mode-button.active {
            background-color: #3498db;
            color: white;
            border-color: #3498db;
        }

        /* Criteria Configuration */
        .criteria-section {
            background-color: #2c3e50;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }

        .pico-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .pico-item {
            background-color: #34495e;
            padding: 1rem;
            border-radius: 6px;
        }

        .pico-item label {
            display: block;
            font-size: 0.9rem;
            color: #ecf0f1;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }

        .pico-item textarea {
            width: 100%;
            padding: 0.5rem;
            border-radius: 4px;
            border: 1px solid #bdc3c7;
            font-size: 0.85rem;
            background-color: white;
            min-height: 60px;
            resize: vertical;
        }

        .criteria-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 1.5rem;
        }

        .criteria-column {
            background-color: #34495e;
            padding: 1rem;
            border-radius: 6px;
        }

        .criteria-column h4 {
            color: #ecf0f1;
            margin-bottom: 1rem;
            font-size: 1rem;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }

        .criteria-item {
            margin-bottom: 1rem;
        }

        .criteria-item label {
            display: block;
            font-size: 0.9rem;
            color: #bdc3c7;
            font-weight: bold;
            margin-bottom: 0.25rem;
        }

        .criteria-item input, .criteria-item textarea {
            width: 100%;
            padding: 0.5rem;
            border-radius: 4px;
            border: 1px solid #bdc3c7;
            font-size: 0.85rem;
            background-color: white;
            min-height: 35px;
        }

        .research-question {
            background-color: #34495e;
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 1.5rem;
        }

        .research-question h4 {
            color: #e74c3c;
            margin-bottom: 0.75rem;
            font-size: 1rem;
        }

        .research-question textarea {
            width: 100%;
            padding: 0.75rem;
            border-radius: 4px;
            border: 2px solid #e74c3c;
            font-size: 0.9rem;
            background-color: white;
            min-height: 80px;
            font-weight: 500;
        }

        /* LLM Configuration */
        .llm-config {
            background-color: #34495e;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }

        .llm-config h3 {
            margin-bottom: 0.5rem;
            color: #ecf0f1;
        }

        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .config-section {
            background-color: #2c3e50;
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid #34495e;
        }

        .config-section h4 {
            color: #3498db;
            margin-bottom: 0.75rem;
            font-size: 1rem;
        }

        .config-row {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }

        .config-row label {
            font-size: 0.9rem;
            color: #bdc3c7;
            font-weight: bold;
        }

        select, input[type="text"], input[type="password"], input[type="number"] {
            padding: 0.5rem;
            border-radius: 4px;
            border: 1px solid #bdc3c7;
            font-size: 0.9rem;
            background-color: white;
        }

        .api-status {
            padding: 0.5rem;
            border-radius: 4px;
            margin-top: 0.5rem;
            font-size: 0.85rem;
            text-align: center;
        }

        .api-status.connected {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }

        .api-status.disconnected {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }

        .api-status.testing {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }

        /* Controls */
        .controls {
            display: flex;
            gap: 1rem;
            align-items: center;
            flex-wrap: wrap;
            margin-top: 1rem;
        }

        .upload-area {
            border: 2px dashed #bdc3c7;
            border-radius: 8px;
            padding: 1rem;
            background-color: white;
            cursor: pointer;
            transition: all 0.3s ease;
            min-width: 200px;
            text-align: center;
        }

        .upload-area:hover {
            border-color: #3498db;
            background-color: #ecf0f1;
        }

        .upload-area.dragover {
            border-color: #2ecc71;
            background-color: #e8f8f5;
        }

        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background-color 0.3s ease;
        }

        button:hover {
            background-color: #2980b9;
        }

        button:disabled {
            background-color: #bdc3c7;
            cursor: not-allowed;
        }

        .ai-button {
            background-color: #9b59b6;
        }

        .ai-button:hover {
            background-color: #8e44ad;
        }

        /* Main Content Layout */
        .main-content {
            display: flex;
            flex: 1;
            overflow: hidden;
        }

        .sidebar {
            width: 320px;
            background-color: white;
            border-right: 1px solid #ddd;
            padding: 1rem;
            overflow-y: auto;
        }

        .reference-list {
            flex: 1;
            padding: 1rem;
            overflow-y: auto;
            background-color: white;
        }

        /* Abstract Navigator */
        .abstract-navigator {
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        .navigator-header {
            background-color: #f8f9fa;
            padding: 1rem;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .navigator-controls {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }

        .nav-button {
            background-color: #6c757d;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background-color 0.3s ease;
        }

        .nav-button:hover {
            background-color: #5a6268;
        }

        .nav-button:disabled {
            background-color: #dee2e6;
            color: #6c757d;
            cursor: not-allowed;
        }

        .abstract-counter {
            font-weight: bold;
            color: #495057;
            font-size: 0.9rem;
        }

        .filter-tabs {
            display: flex;
            gap: 0.25rem;
            margin-left: 1rem;
        }

        .filter-tab {
            background-color: #e9ecef;
            color: #495057;
            border: none;
            padding: 0.4rem 0.8rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.3s ease;
        }

        .filter-tab.active {
            background-color: #007bff;
            color: white;
        }

        .filter-tab:hover {
            background-color: #0056b3;
            color: white;
        }

        .abstract-viewer {
            flex: 1;
            padding: 1.5rem;
            overflow-y: auto;
        }

        .abstract-card {
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            margin-bottom: 1rem;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .abstract-header {
            background-color: #f8f9fa;
            padding: 1rem;
            border-bottom: 1px solid #dee2e6;
            position: relative;
        }

        .abstract-title {
            font-size: 1.1rem;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 0.5rem;
            line-height: 1.4;
        }

        .abstract-meta {
            color: #6c757d;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }

        .abstract-status {
            position: absolute;
            top: 1rem;
            right: 1rem;
            display: flex;
            gap: 0.5rem;
        }

        .status-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: bold;
            text-transform: uppercase;
        }

        .status-badge.include {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }

        .status-badge.exclude {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }

        .status-badge.conflict {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }

        .status-badge.pending {
            background-color: #e2e3e5;
            color: #383d41;
            border: 1px solid #d1ecf1;
        }

        .confidence-score {
            background-color: #e9ecef;
            color: #495057;
            padding: 0.25rem 0.5rem;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: bold;
        }

        .abstract-content {
            padding: 1.5rem;
        }

        .abstract-text {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 6px;
            line-height: 1.6;
            color: #495057;
            margin-bottom: 1.5rem;
            border-left: 4px solid #007bff;
        }

        .ai-evaluation {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-top: 1rem;
        }

        .ai-panel {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 1rem;
        }

        .ai-panel h4 {
            color: #495057;
            margin-bottom: 0.75rem;
            font-size: 0.95rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .ai-decision {
            font-weight: bold;
            padding: 0.5rem;
            border-radius: 4px;
            text-align: center;
            margin-bottom: 0.75rem;
            font-size: 0.9rem;
        }

        .ai-decision.include {
            background-color: #d4edda;
            color: #155724;
        }

        .ai-decision.exclude {
            background-color: #f8d7da;
            color: #721c24;
        }

        .ai-reasoning {
            font-size: 0.85rem;
            color: #6c757d;
            line-height: 1.4;
            background-color: white;
            padding: 0.75rem;
            border-radius: 4px;
            border: 1px solid #e9ecef;
        }

        .pico-scores {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.5rem;
            margin-top: 0.75rem;
        }

        .pico-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.25rem 0.5rem;
            background-color: white;
            border-radius: 4px;
            font-size: 0.8rem;
        }

        .pico-score {
            font-weight: bold;
            color: #495057;
        }

        .study-info {
            background-color: #e9ecef;
            padding: 0.75rem;
            border-radius: 4px;
            margin-top: 0.75rem;
            font-size: 0.85rem;
        }

        .study-info strong {
            color: #495057;
        }

        .no-evaluation {
            text-align: center;
            color: #6c757d;
            padding: 2rem;
            font-style: italic;
        }

        /* Metrics Panel Enhancement */
        .metrics-panel {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 0.75rem;
        }

        .metric-item {
            text-align: center;
            padding: 0.75rem;
            background-color: white;
            border-radius: 4px;
            border: 1px solid #dee2e6;
        }

        .metric-value {
            font-size: 1.25rem;
            font-weight: bold;
            color: #2c3e50;
        }

        .metric-label {
            font-size: 0.75rem;
            color: #6c757d;
            margin-top: 0.25rem;
        }

        .progress-metrics {
            margin-top: 1rem;
            padding: 0.75rem;
            background-color: white;
            border-radius: 4px;
            border: 1px solid #dee2e6;
        }

        .progress-bar-container {
            margin: 0.5rem 0;
        }

        .progress-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: #6c757d;
            margin-bottom: 0.25rem;
        }

        .progress-bar-bg {
            height: 8px;
            background-color: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            transition: width 0.3s ease;
            border-radius: 4px;
        }

        .progress-bar-fill.include {
            background-color: #28a745;
        }

        .progress-bar-fill.exclude {
            background-color: #dc3545;
        }

        .progress-bar-fill.conflict {
            background-color: #ffc107;
        }

        .progress-bar-fill.processed {
            background-color: #007bff;
        }

        /* Advanced Search Interface */
        .search-panel {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .search-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .search-toggle {
            background-color: #6c757d;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background-color 0.3s ease;
        }

        .search-toggle:hover {
            background-color: #5a6268;
        }

        .search-toggle.active {
            background-color: #007bff;
        }

        .search-form {
            display: none;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .search-form.active {
            display: grid;
        }

        .search-group {
            display: flex;
            flex-direction: column;
        }

        .search-group label {
            font-weight: 500;
            color: #495057;
            margin-bottom: 0.25rem;
            font-size: 0.9rem;
        }

        .search-input {
            padding: 0.5rem;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 0.9rem;
            transition: border-color 0.3s ease;
        }

        .search-input:focus {
            border-color: #007bff;
            outline: none;
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
        }

        .search-controls {
            grid-column: span 2;
            display: flex;
            gap: 0.5rem;
            justify-content: flex-end;
            margin-top: 0.5rem;
        }

        .search-btn {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background-color 0.3s ease;
        }

        .search-btn:hover {
            background-color: #0056b3;
        }

        .search-btn.clear {
            background-color: #6c757d;
        }

        .search-btn.clear:hover {
            background-color: #5a6268;
        }

        .search-results-info {
            background-color: #e7f3ff;
            border: 1px solid #b8daff;
            border-radius: 4px;
            padding: 0.75rem;
            margin-bottom: 1rem;
            font-size: 0.9rem;
            color: #0c5460;
        }

        .search-results-info.no-results {
            background-color: #fff3cd;
            border-color: #ffeaa7;
            color: #856404;
        }

        .search-highlight {
            background-color: #fff3cd;
            padding: 0.1rem 0.2rem;
            border-radius: 2px;
            font-weight: 500;
        }

        .filter-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .filter-chip {
            background-color: #e9ecef;
            color: #495057;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }

        .filter-chip.active {
            background-color: #007bff;
            color: white;
        }

        .filter-chip .remove {
            background: none;
            border: none;
            color: inherit;
            cursor: pointer;
            font-size: 1rem;
            line-height: 1;
            padding: 0;
            margin-left: 0.25rem;
        }

        .sorting-controls {
            display: flex;
            gap: 1rem;
            align-items: center;
            margin-bottom: 1rem;
        }

        .sort-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .sort-select {
            padding: 0.4rem;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 0.9rem;
        }

        .sort-direction {
            background-color: #6c757d;
            color: white;
            border: none;
            padding: 0.4rem 0.6rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
        }

        .sort-direction:hover {
            background-color: #5a6268;
        }

        .sort-direction.desc {
            background-color: #007bff;
        }

        /* Mobile Responsive */
        @media (max-width: 768px) {
            .ai-evaluation {
                grid-template-columns: 1fr;
            }
            
            .navigator-header {
                flex-direction: column;
                gap: 0.5rem;
                align-items: stretch;
            }
            
            .navigator-controls {
                justify-content: space-between;
            }
            
            .filter-tabs {
                margin-left: 0;
                justify-content: center;
            }
            
            .pico-scores {
                grid-template-columns: 1fr;
            }

            .search-form {
                grid-template-columns: 1fr;
            }

            .search-controls {
                grid-column: span 1;
            }

            .sorting-controls {
                flex-direction: column;
                align-items: stretch;
                gap: 0.5rem;
            }
        }

        /* Reference Display */
        .reference {
            border: 1px solid #ddd;
            margin-bottom: 1rem;
            border-radius: 8px;
            overflow: hidden;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
        }

        .reference:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .reference.selected {
            border-color: #3498db;
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
        }

        .relevance-indicator {
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            transition: all 0.3s ease;
        }

        .relevance-indicator.high {
            background: linear-gradient(180deg, #28a745 0%, #20c997 100%);
        }

        .relevance-indicator.medium {
            background: linear-gradient(180deg, #ffc107 0%, #fd7e14 100%);
        }

        .relevance-indicator.low {
            background: linear-gradient(180deg, #dc3545 0%, #c82333 100%);
        }

        .reference-header {
            padding: 1rem;
            padding-left: 2rem;
            background-color: #f8f9fa;
            border-bottom: 1px solid #ddd;
            position: relative;
        }

        .reference-title {
            font-weight: bold;
            font-size: 1rem;
            margin-bottom: 0.5rem;
            color: #2c3e50;
        }

        .reference-authors {
            color: #6c757d;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }

        .reference-journal {
            color: #6c757d;
            font-size: 0.85rem;
        }

        .llm-status {
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            display: flex;
            gap: 0.5rem;
        }

        .llm-badge {
            background-color: rgba(155, 89, 182, 0.1);
            color: #8e44ad;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: bold;
        }

        .llm-badge.processing {
            background-color: rgba(241, 196, 15, 0.1);
            color: #f39c12;
            animation: pulse 1.5s ease-in-out infinite;
        }

        .llm-badge.include {
            background-color: rgba(46, 204, 113, 0.1);
            color: #27ae60;
        }

        .llm-badge.exclude {
            background-color: rgba(231, 76, 60, 0.1);
            color: #e74c3c;
        }

        .llm-badge.conflict {
            background-color: rgba(243, 156, 18, 0.1);
            color: #f39c12;
        }

        .llm-analysis {
            background-color: #f8f9ff;
            border-top: 1px solid #ddd;
            padding: 1rem;
            display: none;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .llm-result {
            background-color: white;
            padding: 0.75rem;
            border-radius: 6px;
            border: 1px solid #e9ecef;
        }

        .llm-result h5 {
            color: #495057;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }

        .decision-summary {
            font-weight: bold;
            padding: 0.5rem;
            border-radius: 4px;
            text-align: center;
            margin-bottom: 0.5rem;
        }

        .decision-summary.include {
            background-color: #d4edda;
            color: #155724;
        }

        .decision-summary.exclude {
            background-color: #f8d7da;
            color: #721c24;
        }

        .reasoning-text {
            font-size: 0.85rem;
            color: #6c757d;
            line-height: 1.4;
        }

        /* Stats and Processing */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .stat-item {
            text-align: center;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 6px;
        }

        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
        }

        .stat-label {
            font-size: 0.8rem;
            color: #6c757d;
            margin-top: 0.25rem;
        }

        .processing-queue {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .processing-queue h4 {
            color: #495057;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }

        .queue-item {
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 0.5rem;
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
        }

        .no-references {
            text-align: center;
            color: #6c757d;
            padding: 2rem;
        }

        @keyframes pulse {
            0%, 60%, 100% { opacity: 0.7; }
            30% { opacity: 1; }
        }

        #fileInput { display: none; }

        /* Mobile Responsive */
        @media (max-width: 768px) {
            .sidebar { width: 100%; }
            .main-content { flex-direction: column; }
            .config-grid { grid-template-columns: 1fr; }
            .pico-grid { grid-template-columns: 1fr; }
            .criteria-details { grid-template-columns: 1fr; gap: 1rem; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="progress-bar">
            <div class="progress-fill" id="progressFill" style="width: 0%;">
                <span class="progress-text">0%</span>
            </div>
        </div>
        
        <h1>Otto-SR: Production LLM Screening Tool v3.0</h1>
        <div class="subtitle">Advanced systematic review screening with multiple LLM providers</div>
        
        <!-- Screening Mode Selector -->
        <div class="screening-mode-selector">
            <h3>Screening Mode</h3>
            <div class="mode-buttons">
                <button class="mode-button active" onclick="setScreeningMode('single')" id="singleModeBtn">
                    Single Citation
                </button>
                <button class="mode-button" onclick="setScreeningMode('batch')" id="batchModeBtn">
                    Batch Screening
                </button>
                <button class="mode-button" onclick="setScreeningMode('ai-assisted')" id="aiModeBtn">
                    AI-Assisted
                </button>
            </div>
        </div>
        
        <div class="criteria-section">
            <h3>Systematic Review Criteria (PICO-TT)</h3>
            
            <div class="pico-grid">
                <div class="pico-item">
                    <label>Population (P):</label>
                    <textarea id="population" placeholder="e.g., Adults aged 18+ with Type 2 diabetes mellitus, BMI >25 kg/m²">Adults aged 18+ with Type 2 diabetes mellitus, diagnosed within the last 5 years</textarea>
                </div>
                <div class="pico-item">
                    <label>Intervention (I):</label>
                    <textarea id="intervention" placeholder="e.g., Metformin therapy, lifestyle interventions">Metformin therapy, lifestyle interventions (diet and exercise programs), diabetes education programs</textarea>
                </div>
                <div class="pico-item">
                    <label>Comparison (C):</label>
                    <textarea id="comparison" placeholder="e.g., Placebo, standard care">Placebo, standard diabetes care, other oral antidiabetic medications, control groups</textarea>
                </div>
                <div class="pico-item">
                    <label>Outcomes (O):</label>
                    <textarea id="outcome" placeholder="e.g., HbA1c reduction, quality of life">Primary: HbA1c levels, blood glucose control. Secondary: cardiovascular outcomes, quality of life, medication adherence</textarea>
                </div>
                <div class="pico-item">
                    <label>Time Frame (T1):</label>
                    <textarea id="timeframe" placeholder="e.g., Follow-up ≥3 months">Follow-up duration ≥3 months, intervention duration ≥8 weeks, studies published 2015-2024</textarea>
                </div>
                <div class="pico-item">
                    <label>Study Types (T2):</label>
                    <textarea id="studyTypes" placeholder="e.g., RCTs, systematic reviews">Randomized controlled trials (RCTs), systematic reviews, meta-analyses, controlled clinical trials</textarea>
                </div>
            </div>
            
            <div class="criteria-details">
                <div class="criteria-column">
                    <h4>✅ Inclusion Criteria</h4>
                    <div class="criteria-item">
                        <label>Language:</label>
                        <input type="text" id="inclusionLanguage" value="English" placeholder="e.g., English, Spanish">
                    </div>
                    <div class="criteria-item">
                        <label>Publication Status:</label>
                        <input type="text" id="inclusionPublication" value="Peer-reviewed journals" placeholder="e.g., Peer-reviewed studies">
                    </div>
                    <div class="criteria-item">
                        <label>Sample Size:</label>
                        <input type="text" id="inclusionSampleSize" value="≥20 participants per group" placeholder="e.g., ≥50 participants">
                    </div>
                    <div class="criteria-item">
                        <label>Data Availability:</label>
                        <input type="text" id="inclusionDataAvailability" value="Sufficient data for analysis" placeholder="e.g., Extractable outcome data">
                    </div>
                    <div class="criteria-item">
                        <label>Other Inclusion:</label>
                        <textarea id="otherInclusion" placeholder="Additional requirements...">Studies with adequate randomization and blinding procedures. Clear definition of intervention protocols.</textarea>
                    </div>
                </div>
                
                <div class="criteria-column">
                    <h4>❌ Exclusion Criteria</h4>
                    <div class="criteria-item">
                        <label>Study Types to Exclude:</label>
                        <input type="text" id="exclusionStudyTypes" value="Case reports, editorials, conference abstracts" placeholder="e.g., Case studies, reviews">
                    </div>
                    <div class="criteria-item">
                        <label>Populations to Exclude:</label>
                        <input type="text" id="exclusionPopulations" value="Pediatric populations (<18 years), pregnancy" placeholder="e.g., Children, pregnant women">
                    </div>
                    <div class="criteria-item">
                        <label>Interventions to Exclude:</label>
                        <input type="text" id="exclusionInterventions" value="Surgical interventions, insulin therapy" placeholder="e.g., Invasive procedures">
                    </div>
                    <div class="criteria-item">
                        <label>Languages to Exclude:</label>
                        <input type="text" id="exclusionLanguages" value="Non-English languages" placeholder="e.g., Other languages">
                    </div>
                    <div class="criteria-item">
                        <label>Other Exclusion:</label>
                        <textarea id="otherExclusion" placeholder="Additional exclusions...">Animal studies, in vitro studies, studies with significant methodological flaws, duplicate publications</textarea>
                    </div>
                </div>
            </div>
            
            <div class="research-question">
                <h4>Primary Research Question</h4>
                <textarea id="researchQuestion" placeholder="State your main research question clearly...">In adults with Type 2 diabetes, how effective are metformin and lifestyle interventions compared to standard care in improving glycemic control and cardiovascular outcomes?</textarea>
            </div>
        </div>
        
        <div class="llm-config">
            <h3>LLM Configuration</h3>
            <div class="config-grid">
                <div class="config-section">
                    <h4>AI Model 1 (Conservative)</h4>
                    <div class="config-row">
                        <label>Provider:</label>
                        <select id="ai1Provider" onchange="updateProviderModels('ai1')">
                            <optgroup label="🥇 Recommended for Production">
                                <option value="openai">OpenAI (Recommended)</option>
                                <option value="anthropic">Anthropic Claude (Recommended)</option>
                            </optgroup>
                            <optgroup label="🏠 Local/Self-Hosted">
                                <option value="ollama">Ollama (Local)</option>
                                <option value="openai_compatible">OpenAI-Compatible (Local/Custom)</option>
                            </optgroup>
                            <optgroup label="🔥 Fast Inference">
                                <option value="groq">Groq (Fast Inference)</option>
                                <option value="together">Together AI</option>
                                <option value="cohere">Cohere</option>
                            </optgroup>
                            <optgroup label="⚙️ Advanced">
                                <option value="custom">Custom Provider (Advanced)</option>
                            </optgroup>
                        </select>
                    </div>
                    <div class="config-row">
                        <label>API Key:</label>
                        <input type="password" id="ai1ApiKey" placeholder="Enter API key">
                    </div>
                    <div class="config-row">
                        <label>Model:</label>
                        <input type="text" id="ai1Model" value="gpt-4o" placeholder="Model name">
                    </div>
                    <div class="config-row">
                        <label>Endpoint:</label>
                        <input type="text" id="ai1Endpoint" value="https://api.openai.com/v1/chat/completions" placeholder="API endpoint">
                    </div>
                    <div class="api-status disconnected" id="ai1Status">Not Connected</div>
                    <button onclick="testConnection('ai1')" id="testAI1">Test Connection</button>
                </div>

                <div class="config-section">
                    <h4>AI Model 2 (Pragmatic)</h4>
                    <div class="config-row">
                        <label>Provider:</label>
                        <select id="ai2Provider" onchange="updateProviderModels('ai2')">
                            <optgroup label="🥇 Recommended for Production">
                                <option value="openai">OpenAI (Recommended)</option>
                                <option value="anthropic">Anthropic Claude (Recommended)</option>
                            </optgroup>
                            <optgroup label="🏠 Local/Self-Hosted">
                                <option value="ollama">Ollama (Local)</option>
                                <option value="openai_compatible">OpenAI-Compatible (Local/Custom)</option>
                            </optgroup>
                            <optgroup label="🔥 Fast Inference">
                                <option value="groq">Groq (Fast Inference)</option>
                                <option value="together">Together AI</option>
                                <option value="cohere">Cohere</option>
                            </optgroup>
                            <optgroup label="⚙️ Advanced">
                                <option value="custom">Custom Provider (Advanced)</option>
                            </optgroup>
                        </select>
                    </div>
                    <div class="config-row">
                        <label>API Key:</label>
                        <input type="password" id="ai2ApiKey" placeholder="Enter API key">
                    </div>
                    <div class="config-row">
                        <label>Model:</label>
                        <input type="text" id="ai2Model" value="gpt-4o" placeholder="Model name">
                    </div>
                    <div class="config-row">
                        <label>Endpoint:</label>
                        <input type="text" id="ai2Endpoint" value="https://api.openai.com/v1/chat/completions" placeholder="API endpoint">
                    </div>
                    <div class="api-status disconnected" id="ai2Status">Not Connected</div>
                    <button onclick="testConnection('ai2')" id="testAI2">Test Connection</button>
                </div>
            </div>
        </div>

        <div class="controls">
            <div class="upload-area" id="uploadArea">
                <div>Drop XML/RIS file here or click to browse</div>
                <input type="file" id="fileInput" accept=".xml,.ris" multiple />
            </div>
            <button onclick="startScreening()" id="startBtn" class="ai-button" disabled>Start Screening</button>
            <button onclick="pauseScreening()" id="pauseBtn" disabled>Pause</button>
            <button onclick="exportResults()" id="exportBtn">Export Results</button>
        </div>
    </div>

    <div class="main-content">
        <div class="sidebar">
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value" id="totalCount">0</div>
                    <div class="stat-label">Total</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="processedCount">0</div>
                    <div class="stat-label">Processed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="conflictCount">0</div>
                    <div class="stat-label">Conflicts</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="includeCount">0</div>
                    <div class="stat-label">Included</div>
                </div>
            </div>
            
            <div class="processing-queue">
                <h4>Processing Queue</h4>
                <div id="queueList">
                    <div class="queue-item">No items in queue</div>
                </div>
            </div>

            <!-- Advanced Search Panel -->
            <div class="search-panel" id="searchPanel" style="display: none;">
                <div class="search-header">
                    <h4 style="color: #2c3e50; margin: 0;">Advanced Search & Filtering</h4>
                    <button class="search-toggle" id="searchToggle" onclick="toggleSearch()">🔍 Advanced Search</button>
                </div>
                
                <div class="search-form" id="searchForm">
                    <div class="search-group">
                        <label for="searchTitle">Title:</label>
                        <input type="text" id="searchTitle" class="search-input" placeholder="Search by title...">
                    </div>
                    
                    <div class="search-group">
                        <label for="searchAuthors">Authors:</label>
                        <input type="text" id="searchAuthors" class="search-input" placeholder="Search by authors...">
                    </div>
                    
                    <div class="search-group">
                        <label for="searchJournal">Journal:</label>
                        <input type="text" id="searchJournal" class="search-input" placeholder="Search by journal...">
                    </div>
                    
                    <div class="search-group">
                        <label for="searchYear">Publication Year:</label>
                        <input type="text" id="searchYear" class="search-input" placeholder="e.g. 2020-2024 or 2023">
                    </div>
                    
                    <div class="search-group">
                        <label for="searchKeywords">Keywords:</label>
                        <input type="text" id="searchKeywords" class="search-input" placeholder="Search by keywords...">
                    </div>
                    
                    <div class="search-group">
                        <label for="searchAbstract">Abstract Content:</label>
                        <input type="text" id="searchAbstract" class="search-input" placeholder="Search within abstracts...">
                    </div>
                    
                    <div class="search-controls">
                        <button class="search-btn clear" onclick="clearSearch()">Clear All</button>
                        <button class="search-btn" onclick="performSearch()">Search</button>
                    </div>
                </div>

                <!-- Active Filter Chips -->
                <div class="filter-chips" id="filterChips" style="display: none;"></div>

                <!-- Search Results Info -->
                <div class="search-results-info" id="searchResultsInfo" style="display: none;"></div>

                <!-- Sorting Controls -->
                <div class="sorting-controls" id="sortingControls" style="display: none;">
                    <div class="sort-group">
                        <label for="sortBy">Sort by:</label>
                        <select id="sortBy" class="sort-select" onchange="applySorting()">
                            <option value="relevance">Relevance</option>
                            <option value="title">Title</option>
                            <option value="authors">Authors</option>
                            <option value="year">Year</option>
                            <option value="journal">Journal</option>
                            <option value="decision">AI Decision</option>
                            <option value="confidence">Confidence Score</option>
                        </select>
                    </div>
                    
                    <button class="sort-direction" id="sortDirection" onclick="toggleSortDirection()">▲ Ascending</button>
                </div>
            </div>

            <!-- Enhanced Metrics Panel -->
            <div class="metrics-panel" id="metricsPanel" style="display: none;">
                <h4 style="margin-bottom: 1rem; color: #2c3e50;">Detailed Screening Metrics</h4>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-value" id="totalAbstracts">0</div>
                        <div class="metric-label">Total Abstracts</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="processedAbstracts">0</div>
                        <div class="metric-label">Processed</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="includedAbstracts">0</div>
                        <div class="metric-label">Included</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="excludedAbstracts">0</div>
                        <div class="metric-label">Excluded</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="conflictAbstracts">0</div>
                        <div class="metric-label">Conflicts</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="inclusionRate">0%</div>
                        <div class="metric-label">Inclusion Rate</div>
                    </div>
                </div>
                
                <div class="progress-metrics">
                    <div class="progress-bar-container">
                        <div class="progress-label">
                            <span>Processing Progress</span>
                            <span id="processingPercentage">0%</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill processed" id="processingProgressBar" style="width: 0%;"></div>
                        </div>
                    </div>
                    
                    <div class="progress-bar-container">
                        <div class="progress-label">
                            <span>Inclusion Distribution</span>
                            <span id="inclusionPercentage">0%</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill include" id="inclusionProgressBar" style="width: 0%;"></div>
                        </div>
                    </div>
                    
                    <div class="progress-bar-container">
                        <div class="progress-label">
                            <span>Exclusion Distribution</span>
                            <span id="exclusionPercentage">0%</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill exclude" id="exclusionProgressBar" style="width: 0%;"></div>
                        </div>
                    </div>
                    
                    <div class="progress-bar-container">
                        <div class="progress-label">
                            <span>Conflicts</span>
                            <span id="conflictPercentage">0%</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill conflict" id="conflictProgressBar" style="width: 0%;"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Abstract Navigator -->
            <div class="abstract-navigator" id="abstractNavigator" style="display: none;">
                <div class="navigator-header">
                    <div class="navigator-controls">
                        <button class="nav-button" id="prevButton" onclick="navigateAbstract(-1)">◀ Previous</button>
                        <span class="abstract-counter" id="abstractCounter">0 / 0</span>
                        <button class="nav-button" id="nextButton" onclick="navigateAbstract(1)">Next ▶</button>
                    </div>
                    
                    <div class="filter-tabs">
                        <button class="filter-tab active" data-filter="all" onclick="setFilter('all')">All</button>
                        <button class="filter-tab" data-filter="include" onclick="setFilter('include')">Included</button>
                        <button class="filter-tab" data-filter="exclude" onclick="setFilter('exclude')">Excluded</button>
                        <button class="filter-tab" data-filter="conflict" onclick="setFilter('conflict')">Conflicts</button>
                        <button class="filter-tab" data-filter="pending" onclick="setFilter('pending')">Pending</button>
                    </div>
                </div>
                
                <div class="abstract-viewer" id="abstractViewer">
                    <div class="no-evaluation">
                        <p>No abstracts available for review.</p>
                        <p>Upload citations and complete screening to see results here.</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="reference-list" id="referenceList">
            <div class="no-references">Configure LLM connections and upload files to begin screening</div>
        </div>
    </div>

    <script>
        // --- Global Variables ---
        let currentProject = null;
        let references = [];
        let screeningMode = 'single';
        let isProcessing = false;
        let llmConfigs = {
            ai1: { connected: false },
            ai2: { connected: false }
        };
        let providerConfigs = {};
        
        // Abstract Navigator Variables
        let currentAbstractIndex = 0;
        let filteredReferences = [];
        let currentFilter = 'all';
        
        // Search and Filtering Variables
        let searchCriteria = {};
        let activeFilters = [];
        let sortBy = 'relevance';
        let sortDirection = 'asc';
        let searchResults = [];
        let originalReferences = [];

        // --- Event Listeners ---
        document.addEventListener('DOMContentLoaded', () => {
            document.getElementById('fileInput').addEventListener('change', handleFileUpload);
            setupDragAndDrop();
            loadProviderConfigurations();
        });

        // --- Provider Management ---
        async function loadProviderConfigurations() {
            try {
                const response = await fetch('/api/providers');
                const data = await response.json();
                providerConfigs = data.providers;
                
                // Initialize provider dropdowns and default values
                updateProviderModels('ai1');
                updateProviderModels('ai2');
            } catch (error) {
                console.error('Error loading provider configurations:', error);
            }
        }

        function updateProviderModels(aiModel) {
            const providerSelect = document.getElementById(`${aiModel}Provider`);
            const modelSelect = document.getElementById(`${aiModel}Model`);
            const endpointInput = document.getElementById(`${aiModel}Endpoint`);
            const apiKeyInput = document.getElementById(`${aiModel}ApiKey`);
            
            const selectedProvider = providerSelect.value;
            const providerConfig = providerConfigs[selectedProvider];
            
            if (!providerConfig) return;
            
            // Update model options
            modelSelect.innerHTML = '';
            providerConfig.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                if (model === providerConfig.default_model) {
                    option.selected = true;
                }
                modelSelect.appendChild(option);
            });
            
            // Update endpoint if provider has default
            if (providerConfig.default_endpoint) {
                endpointInput.value = providerConfig.default_endpoint;
            } else {
                // Set default endpoints for known providers
                const defaultEndpoints = {
                    'openai': 'https://api.openai.com/v1/chat/completions',
                    'anthropic': 'https://api.anthropic.com/v1/messages',
                    'cohere': 'https://api.cohere.ai/v1/chat',
                    'groq': 'https://api.groq.com/openai/v1',
                    'together': 'https://api.together.xyz/v1',
                    'custom': 'https://your-api-endpoint.com/v1'
                };
                endpointInput.value = defaultEndpoints[selectedProvider] || '';
            }
            
            // Add provider-specific hints
            const configSection = endpointInput.closest('.config-section');
            let hintElement = configSection.querySelector('.provider-hint');
            if (!hintElement) {
                hintElement = document.createElement('div');
                hintElement.className = 'provider-hint';
                hintElement.style.cssText = 'font-size: 0.8rem; color: #6c757d; margin-top: 0.5rem; padding: 0.5rem; background: #f8f9fa; border-radius: 4px;';
                configSection.appendChild(hintElement);
            }
            
            const providerHints = {
                'openai': '🥇 Most reliable for production. Excellent instruction following and consistent outputs.',
                'anthropic': '🥇 Highly reliable with strong reasoning. Great for complex screening decisions.',
                'ollama': '🏠 Run models locally. Requires Ollama to be installed and running on your system.',
                'openai_compatible': '🏠 Works with LM Studio, Oobabooga, or any OpenAI-compatible local server.',
                'groq': '🔥 Ultra-fast inference with good quality. Great for high-volume screening.',
                'together': '🔥 Fast and cost-effective. Good balance of speed and quality.',
                'cohere': '⚡ Strong performance with good multilingual support.',
                'custom': '⚙️ Configure your own endpoint. Supports any OpenAI-compatible API.'
            };
            
            hintElement.textContent = providerHints[selectedProvider] || 'Configure the endpoint and model for this provider.';
            
            // Show/hide API key field based on provider requirements
            const apiKeyRow = apiKeyInput.closest('.config-row');
            if (providerConfig.requires_api_key) {
                apiKeyRow.style.display = 'flex';
                apiKeyInput.placeholder = `Enter ${providerConfig.display_name} API key`;
            } else {
                apiKeyRow.style.display = 'none';
                apiKeyInput.value = '';
            }
            
            // Reset connection status
            const statusElement = document.getElementById(`${aiModel}Status`);
            statusElement.textContent = 'Not Connected';
            statusElement.className = 'api-status disconnected';
            llmConfigs[aiModel].connected = false;
            
            updateStartButton();
        }

        function setupDragAndDrop() {
            const uploadArea = document.getElementById('uploadArea');
            
            uploadArea.addEventListener('click', () => document.getElementById('fileInput').click());
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                handleFileUpload({ target: { files: e.dataTransfer.files } });
            });
        }

        // --- Mode Management ---
        function setScreeningMode(mode) {
            screeningMode = mode;
            
            // Update button states
            document.querySelectorAll('.mode-button').forEach(btn => btn.classList.remove('active'));
            document.getElementById(mode + 'ModeBtn').classList.add('active');
            
            updateStartButton();
        }

        // --- File Upload ---
        async function handleFileUpload(event) {
            const files = event.target.files;
            if (!files.length) return;

            const formData = new FormData();
            for (const file of files) {
                formData.append('files', file);
            }

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error('Upload failed');

                const result = await response.json();
                currentProject = result.project_id;
                
                // Fetch the uploaded citations
                await loadReferences();
                updateDisplay();
                updateStartButton();
                
            } catch (error) {
                alert('Error uploading files: ' + error.message);
            }
        }

        async function loadReferences() {
            if (!currentProject) return;
            
            try {
                const response = await fetch(`/api/projects/${currentProject}/citations`);
                if (response.ok) {
                    references = await response.json();
                }
            } catch (error) {
                console.error('Error loading references:', error);
            }
        }

        // --- LLM Configuration ---
        async function testConnection(aiModel) {
            const statusElement = document.getElementById(`${aiModel}Status`);
            statusElement.textContent = 'Testing...';
            statusElement.className = 'api-status testing';

            const config = {
                provider: document.getElementById(`${aiModel}Provider`).value,
                model: document.getElementById(`${aiModel}Model`).value,
                endpoint: document.getElementById(`${aiModel}Endpoint`).value,
                api_key: document.getElementById(`${aiModel}ApiKey`).value
            };

            try {
                const response = await fetch('/api/test-llm', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ai_model: aiModel, config: config })
                });

                const result = await response.json();
                
                if (result.success) {
                    statusElement.textContent = 'Connected ✓';
                    statusElement.className = 'api-status connected';
                    llmConfigs[aiModel] = { ...config, connected: true };
                } else {
                    statusElement.textContent = `Error: ${result.error}`;
                    statusElement.className = 'api-status disconnected';
                    llmConfigs[aiModel].connected = false;
                }
            } catch (error) {
                statusElement.textContent = `Failed: ${error.message}`;
                statusElement.className = 'api-status disconnected';
                llmConfigs[aiModel].connected = false;
            }

            updateStartButton();
        }

        // --- Screening Process ---
        async function startScreening() {
            if (!currentProject || isProcessing) return;

            const criteria = collectCriteria();
            
            try {
                isProcessing = true;
                updateStartButton();

                const response = await fetch(`/api/projects/${currentProject}/screen`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        criteria: criteria,
                        llm_configs: llmConfigs,
                        mode: screeningMode
                    })
                });

                if (!response.ok) throw new Error('Failed to start screening');

                const result = await response.json();
                
                // Start monitoring progress
                monitorProgress(result.job_id);
                
            } catch (error) {
                alert('Error starting screening: ' + error.message);
                isProcessing = false;
                updateStartButton();
            }
        }

        function pauseScreening() {
            isProcessing = false;
            updateStartButton();
        }

        async function monitorProgress(jobId) {
            const eventSource = new EventSource(`/api/stream/${jobId}`);
            
            eventSource.addEventListener('progress', (event) => {
                const data = JSON.parse(event.data);
                updateReferenceStatus(data);
                updateDisplay();
            });
            
            eventSource.addEventListener('complete', () => {
                isProcessing = false;
                updateStartButton();
                eventSource.close();
            });
            
            eventSource.addEventListener('error', () => {
                isProcessing = false;
                updateStartButton();
                eventSource.close();
            });
        }

        // --- UI Updates ---
        function updateDisplay() {
            updateStats();
            updateReferenceList();
            updateProgressBar();
        }

        function updateStats() {
            const total = references.length;
            const processed = references.filter(r => r.status === 'completed' || r.status === 'error').length;
            const conflicts = references.filter(r => r.final_decision === 'conflict').length;
            const includes = references.filter(r => r.final_decision === 'include').length;

            document.getElementById('totalCount').textContent = total;
            document.getElementById('processedCount').textContent = processed;
            document.getElementById('conflictCount').textContent = conflicts;
            document.getElementById('includeCount').textContent = includes;
        }

        function updateProgressBar() {
            const total = references.length;
            const processed = references.filter(r => r.status === 'completed' || r.status === 'error').length;
            const percentage = total > 0 ? Math.round((processed / total) * 100) : 0;

            const progressFill = document.getElementById('progressFill');
            progressFill.style.width = `${percentage}%`;
            progressFill.querySelector('.progress-text').textContent = `${percentage}%`;
        }

        function updateReferenceList() {
            const listElement = document.getElementById('referenceList');
            
            if (!references.length) {
                listElement.innerHTML = '<div class="no-references">Configure LLM connections and upload files to begin screening</div>';
                // Show search panel and navigator when we have references
                updateSearchInterface();
                updateAbstractNavigator();
                updateMetricsPanel();
                return;
            }
            
            // Store original references for search
            if (originalReferences.length === 0) {
                originalReferences = [...references];
            }

            const referencesHtml = references.map(ref => {
                const relevanceClass = ref.relevance_score > 0.7 ? 'high' : 
                                     ref.relevance_score > 0.4 ? 'medium' : 'low';
                
                let statusBadges = '';
                if (ref.status === 'processing') {
                    statusBadges = '<div class="llm-badge processing">Processing...</div>';
                } else if (ref.status === 'completed') {
                    const decision = ref.final_decision || 'pending';
                    statusBadges = `<div class="llm-badge ${decision}">${decision.toUpperCase()}</div>`;
                }

                return `
                    <div class="reference" onclick="toggleAnalysis('${ref.id}')">
                        <div class="relevance-indicator ${relevanceClass}"></div>
                        <div class="reference-header">
                            <div class="llm-status">${statusBadges}</div>
                            <div class="reference-title">${ref.title}</div>
                            <div class="reference-authors">${ref.authors || 'Unknown authors'}</div>
                            <div class="reference-journal">${ref.journal || ''} ${ref.year || ''}</div>
                        </div>
                        <div class="llm-analysis" id="analysis-${ref.id}">
                            ${ref.ai1_result ? renderAnalysisResults(ref) : ''}
                        </div>
                    </div>
                `;
            }).join('');

            listElement.innerHTML = referencesHtml;
            
            // Update search interface, abstract navigator and metrics when references change
            updateSearchInterface();
            updateAbstractNavigator();
            updateMetricsPanel();
        }

        function renderAnalysisResults(ref) {
            if (!ref.ai1_result || !ref.ai2_result) return '';
            
            return `
                <div class="llm-result">
                    <h5>Conservative AI</h5>
                    <div class="decision-summary ${ref.ai1_result.decision}">
                        ${ref.ai1_result.decision.toUpperCase()} (${ref.ai1_result.confidence}%)
                    </div>
                    <div class="reasoning-text">${ref.ai1_result.reasoning}</div>
                </div>
                <div class="llm-result">
                    <h5>Pragmatic AI</h5>
                    <div class="decision-summary ${ref.ai2_result.decision}">
                        ${ref.ai2_result.decision.toUpperCase()} (${ref.ai2_result.confidence}%)
                    </div>
                    <div class="reasoning-text">${ref.ai2_result.reasoning}</div>
                </div>
            `;
        }

        function toggleAnalysis(refId) {
            const analysisEl = document.getElementById(`analysis-${refId}`);
            if (analysisEl) {
                analysisEl.style.display = analysisEl.style.display === 'grid' ? 'none' : 'grid';
            }
        }

        function updateReferenceStatus(data) {
            const ref = references.find(r => r.id === data.citation_id);
            if (ref) {
                Object.assign(ref, data);
            }
        }

        function updateStartButton() {
            const startBtn = document.getElementById('startBtn');
            const ai1Connected = llmConfigs.ai1.connected;
            const ai2Connected = llmConfigs.ai2.connected;
            const hasReferences = references.length > 0;
            
            if (isProcessing) {
                startBtn.disabled = true;
                startBtn.textContent = 'Processing...';
            } else if (!ai1Connected || !ai2Connected) {
                startBtn.disabled = true;
                startBtn.textContent = 'Connect LLMs First';
            } else if (!hasReferences) {
                startBtn.disabled = true;
                startBtn.textContent = 'Upload Studies First';
            } else {
                startBtn.disabled = false;
                startBtn.textContent = `Start ${screeningMode} Screening`;
            }
        }

        // --- Utility Functions ---
        function collectCriteria() {
            return {
                population: document.getElementById('population').value,
                intervention: document.getElementById('intervention').value,
                comparison: document.getElementById('comparison').value,
                outcome: document.getElementById('outcome').value,
                timeframe: document.getElementById('timeframe').value,
                studyTypes: document.getElementById('studyTypes').value,
                inclusionLanguage: document.getElementById('inclusionLanguage').value,
                inclusionPublication: document.getElementById('inclusionPublication').value,
                inclusionSampleSize: document.getElementById('inclusionSampleSize').value,
                inclusionDataAvailability: document.getElementById('inclusionDataAvailability').value,
                otherInclusion: document.getElementById('otherInclusion').value,
                exclusionStudyTypes: document.getElementById('exclusionStudyTypes').value,
                exclusionPopulations: document.getElementById('exclusionPopulations').value,
                exclusionInterventions: document.getElementById('exclusionInterventions').value,
                exclusionLanguages: document.getElementById('exclusionLanguages').value,
                otherExclusion: document.getElementById('otherExclusion').value,
                researchQuestion: document.getElementById('researchQuestion').value
            };
        }

        async function exportResults() {
            if (!currentProject) return;
            
            try {
                const response = await fetch(`/api/projects/${currentProject}/export`);
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `screening-results-${new Date().toISOString().split('T')[0]}.json`;
                    a.click();
                    window.URL.revokeObjectURL(url);
                }
            } catch (error) {
                alert('Error exporting results: ' + error.message);
            }
        }

        // --- Abstract Navigator Functions ---
        function updateAbstractNavigator() {
            const navigator = document.getElementById('abstractNavigator');
            const metricsPanel = document.getElementById('metricsPanel');
            
            // Show navigator if we have any references
            if (references.length > 0) {
                navigator.style.display = 'flex';
                metricsPanel.style.display = 'block';
                
                // Apply current filter
                applyFilter();
                
                // Update navigation controls
                updateNavigationControls();
                
                // Display current abstract
                displayCurrentAbstract();
            } else {
                navigator.style.display = 'none';
                metricsPanel.style.display = 'none';
            }
        }

        function applyFilter() {
            switch (currentFilter) {
                case 'include':
                    filteredReferences = references.filter(ref => ref.final_decision === 'include');
                    break;
                case 'exclude':
                    filteredReferences = references.filter(ref => ref.final_decision === 'exclude');
                    break;
                case 'conflict':
                    filteredReferences = references.filter(ref => ref.final_decision === 'conflict');
                    break;
                case 'pending':
                    filteredReferences = references.filter(ref => !ref.final_decision || ref.status !== 'completed');
                    break;
                default:
                    filteredReferences = [...references];
            }
            
            // Reset index if it's out of bounds
            if (currentAbstractIndex >= filteredReferences.length) {
                currentAbstractIndex = 0;
            }
        }

        function setFilter(filter) {
            currentFilter = filter;
            currentAbstractIndex = 0;
            
            // Update filter tab UI
            document.querySelectorAll('.filter-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelector(`[data-filter="${filter}"]`).classList.add('active');
            
            // Apply filter and update display
            applyFilter();
            updateNavigationControls();
            displayCurrentAbstract();
        }

        function navigateAbstract(direction) {
            if (filteredReferences.length === 0) return;
            
            currentAbstractIndex += direction;
            
            // Handle wraparound
            if (currentAbstractIndex < 0) {
                currentAbstractIndex = filteredReferences.length - 1;
            } else if (currentAbstractIndex >= filteredReferences.length) {
                currentAbstractIndex = 0;
            }
            
            updateNavigationControls();
            displayCurrentAbstract();
        }

        function updateNavigationControls() {
            const prevButton = document.getElementById('prevButton');
            const nextButton = document.getElementById('nextButton');
            const counter = document.getElementById('abstractCounter');
            
            const total = filteredReferences.length;
            const current = total > 0 ? currentAbstractIndex + 1 : 0;
            
            counter.textContent = `${current} / ${total}`;
            
            // Enable/disable navigation buttons
            prevButton.disabled = total === 0;
            nextButton.disabled = total === 0;
        }

        function displayCurrentAbstract() {
            const viewer = document.getElementById('abstractViewer');
            
            if (filteredReferences.length === 0) {
                viewer.innerHTML = `
                    <div class="no-evaluation">
                        <p>No abstracts match the current filter: "${currentFilter}"</p>
                        <p>Try selecting a different filter or complete more screening.</p>
                    </div>
                `;
                return;
            }
            
            const ref = filteredReferences[currentAbstractIndex];
            viewer.innerHTML = renderAbstractCard(ref);
        }

        function renderAbstractCard(ref) {
            const statusClass = getStatusClass(ref);
            const confidenceScore = getConfidenceScore(ref);
            
            return `
                <div class="abstract-card">
                    <div class="abstract-header">
                        <div class="abstract-title">${ref.title || 'Untitled'}</div>
                        <div class="abstract-meta">
                            <strong>Authors:</strong> ${ref.authors || 'Unknown authors'}<br>
                            <strong>Journal:</strong> ${ref.journal || 'Unknown journal'} ${ref.year ? `(${ref.year})` : ''}<br>
                            ${ref.doi ? `<strong>DOI:</strong> ${ref.doi}<br>` : ''}
                            ${ref.keywords ? `<strong>Keywords:</strong> ${ref.keywords}<br>` : ''}
                        </div>
                        <div class="abstract-status">
                            <div class="status-badge ${statusClass}">${getStatusText(ref)}</div>
                            ${confidenceScore ? `<div class="confidence-score">${confidenceScore}% confidence</div>` : ''}
                        </div>
                    </div>
                    
                    <div class="abstract-content">
                        <div class="abstract-text">
                            ${ref.abstract || 'No abstract available'}
                        </div>
                        
                        ${renderAIEvaluation(ref)}
                    </div>
                </div>
            `;
        }

        function renderAIEvaluation(ref) {
            if (!ref.ai1_result && !ref.ai2_result) {
                return `
                    <div class="no-evaluation">
                        <p>No AI evaluation available yet.</p>
                        <p>This abstract is pending screening.</p>
                    </div>
                `;
            }
            
            return `
                <div class="ai-evaluation">
                    ${ref.ai1_result ? renderAIPanel('Conservative AI', ref.ai1_result) : ''}
                    ${ref.ai2_result ? renderAIPanel('Pragmatic AI', ref.ai2_result) : ''}
                </div>
            `;
        }

        function renderAIPanel(title, aiResult) {
            const decision = aiResult.decision || 'unknown';
            const confidence = aiResult.confidence || 0;
            const reasoning = aiResult.reasoning || 'No reasoning provided';
            const picoScores = aiResult.pico_scores || {};
            const studyDesign = aiResult.study_design || 'Unknown';
            const qualityAssessment = aiResult.quality_assessment || 'Unknown';
            const keyFindings = aiResult.key_findings || [];
            
            return `
                <div class="ai-panel">
                    <h4>${title} <span class="confidence-score">${confidence}%</span></h4>
                    
                    <div class="ai-decision ${decision}">
                        ${decision.toUpperCase()}
                    </div>
                    
                    <div class="ai-reasoning">
                        ${reasoning}
                    </div>
                    
                    <div class="pico-scores">
                        <div class="pico-item">
                            <span>Population:</span>
                            <span class="pico-score">${(picoScores.population * 100 || 0).toFixed(0)}%</span>
                        </div>
                        <div class="pico-item">
                            <span>Intervention:</span>
                            <span class="pico-score">${(picoScores.intervention * 100 || 0).toFixed(0)}%</span>
                        </div>
                        <div class="pico-item">
                            <span>Comparison:</span>
                            <span class="pico-score">${(picoScores.comparison * 100 || 0).toFixed(0)}%</span>
                        </div>
                        <div class="pico-item">
                            <span>Outcome:</span>
                            <span class="pico-score">${(picoScores.outcome * 100 || 0).toFixed(0)}%</span>
                        </div>
                    </div>
                    
                    <div class="study-info">
                        <strong>Study Design:</strong> ${studyDesign}<br>
                        <strong>Quality Assessment:</strong> ${qualityAssessment}
                        ${keyFindings.length > 0 ? `<br><strong>Key Findings:</strong> ${keyFindings.join(', ')}` : ''}
                    </div>
                </div>
            `;
        }

        function getStatusClass(ref) {
            if (ref.status === 'processing') return 'pending';
            if (ref.final_decision === 'conflict') return 'conflict';
            if (ref.final_decision === 'include') return 'include';
            if (ref.final_decision === 'exclude') return 'exclude';
            return 'pending';
        }

        function getStatusText(ref) {
            if (ref.status === 'processing') return 'Processing';
            if (ref.final_decision === 'conflict') return 'Conflict';
            if (ref.final_decision === 'include') return 'Include';
            if (ref.final_decision === 'exclude') return 'Exclude';
            return 'Pending';
        }

        function getConfidenceScore(ref) {
            if (ref.ai1_result && ref.ai2_result) {
                return Math.round((ref.ai1_result.confidence + ref.ai2_result.confidence) / 2);
            } else if (ref.ai1_result) {
                return ref.ai1_result.confidence;
            } else if (ref.ai2_result) {
                return ref.ai2_result.confidence;
            }
            return null;
        }

        function updateMetricsPanel() {
            if (references.length === 0) {
                document.getElementById('metricsPanel').style.display = 'none';
                return;
            }
            
            const total = references.length;
            const processed = references.filter(r => r.status === 'completed').length;
            const included = references.filter(r => r.final_decision === 'include').length;
            const excluded = references.filter(r => r.final_decision === 'exclude').length;
            const conflicts = references.filter(r => r.final_decision === 'conflict').length;
            
            const inclusionRate = processed > 0 ? Math.round((included / processed) * 100) : 0;
            const processingPercentage = Math.round((processed / total) * 100);
            const inclusionPercentage = total > 0 ? Math.round((included / total) * 100) : 0;
            const exclusionPercentage = total > 0 ? Math.round((excluded / total) * 100) : 0;
            const conflictPercentage = total > 0 ? Math.round((conflicts / total) * 100) : 0;
            
            // Update metric values
            document.getElementById('totalAbstracts').textContent = total;
            document.getElementById('processedAbstracts').textContent = processed;
            document.getElementById('includedAbstracts').textContent = included;
            document.getElementById('excludedAbstracts').textContent = excluded;
            document.getElementById('conflictAbstracts').textContent = conflicts;
            document.getElementById('inclusionRate').textContent = `${inclusionRate}%`;
            
            // Update progress bars
            document.getElementById('processingPercentage').textContent = `${processingPercentage}%`;
            document.getElementById('processingProgressBar').style.width = `${processingPercentage}%`;
            
            document.getElementById('inclusionPercentage').textContent = `${inclusionPercentage}%`;
            document.getElementById('inclusionProgressBar').style.width = `${inclusionPercentage}%`;
            
            document.getElementById('exclusionPercentage').textContent = `${exclusionPercentage}%`;
            document.getElementById('exclusionProgressBar').style.width = `${exclusionPercentage}%`;
            
            document.getElementById('conflictPercentage').textContent = `${conflictPercentage}%`;
            document.getElementById('conflictProgressBar').style.width = `${conflictPercentage}%`;
        }

        // --- Advanced Search and Filtering Functions ---
        function updateSearchInterface() {
            const searchPanel = document.getElementById('searchPanel');
            const sortingControls = document.getElementById('sortingControls');
            
            if (references.length > 0) {
                searchPanel.style.display = 'block';
                sortingControls.style.display = 'flex';
            } else {
                searchPanel.style.display = 'none';
                sortingControls.style.display = 'none';
            }
        }

        function toggleSearch() {
            const searchForm = document.getElementById('searchForm');
            const searchToggle = document.getElementById('searchToggle');
            
            if (searchForm.classList.contains('active')) {
                searchForm.classList.remove('active');
                searchToggle.classList.remove('active');
                searchToggle.textContent = '🔍 Advanced Search';
            } else {
                searchForm.classList.add('active');
                searchToggle.classList.add('active');
                searchToggle.textContent = '✕ Hide Search';
            }
        }

        function performSearch() {
            const criteria = collectSearchCriteria();
            
            // Clear previous search if no criteria
            if (Object.keys(criteria).length === 0) {
                clearSearch();
                return;
            }
            
            searchCriteria = criteria;
            searchResults = filterReferences(originalReferences, criteria);
            
            // Update references with search results
            references = [...searchResults];
            
            // Apply sorting
            applySorting();
            
            // Update UI components
            updateFilterChips();
            updateSearchResultsInfo();
            updateReferenceList();
            updateAbstractNavigator();
            updateMetricsPanel();
        }

        function collectSearchCriteria() {
            const criteria = {};
            
            const title = document.getElementById('searchTitle').value.trim();
            const authors = document.getElementById('searchAuthors').value.trim();
            const journal = document.getElementById('searchJournal').value.trim();
            const year = document.getElementById('searchYear').value.trim();
            const keywords = document.getElementById('searchKeywords').value.trim();
            const abstract = document.getElementById('searchAbstract').value.trim();
            
            if (title) criteria.title = title;
            if (authors) criteria.authors = authors;
            if (journal) criteria.journal = journal;
            if (year) criteria.year = year;
            if (keywords) criteria.keywords = keywords;
            if (abstract) criteria.abstract = abstract;
            
            return criteria;
        }

        function filterReferences(refs, criteria) {
            return refs.filter(ref => {
                // Title search
                if (criteria.title && !searchInText(ref.title, criteria.title)) {
                    return false;
                }
                
                // Authors search
                if (criteria.authors && !searchInText(ref.authors, criteria.authors)) {
                    return false;
                }
                
                // Journal search
                if (criteria.journal && !searchInText(ref.journal, criteria.journal)) {
                    return false;
                }
                
                // Year search (supports ranges like "2020-2024" or single year "2023")
                if (criteria.year && !matchYear(ref.year, criteria.year)) {
                    return false;
                }
                
                // Keywords search
                if (criteria.keywords && !searchInText(ref.keywords, criteria.keywords)) {
                    return false;
                }
                
                // Abstract content search
                if (criteria.abstract && !searchInText(ref.abstract, criteria.abstract)) {
                    return false;
                }
                
                return true;
            });
        }

        function searchInText(text, searchTerm) {
            if (!text || !searchTerm) return false;
            return text.toLowerCase().includes(searchTerm.toLowerCase());
        }

        function matchYear(refYear, searchYear) {
            if (!refYear || !searchYear) return false;
            
            const yearStr = refYear.toString();
            
            // Check for range (e.g., "2020-2024")
            if (searchYear.includes('-')) {
                const [startYear, endYear] = searchYear.split('-').map(y => parseInt(y.trim()));
                const year = parseInt(yearStr);
                return year >= startYear && year <= endYear;
            }
            
            // Single year or partial match
            return yearStr.includes(searchYear.trim());
        }

        function highlightSearchTerms(text, searchTerm) {
            if (!text || !searchTerm) return text;
            
            const regex = new RegExp(`(${searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
            return text.replace(regex, '<span class="search-highlight">$1</span>');
        }

        function updateFilterChips() {
            const chipsContainer = document.getElementById('filterChips');
            
            if (Object.keys(searchCriteria).length === 0) {
                chipsContainer.style.display = 'none';
                return;
            }
            
            chipsContainer.style.display = 'flex';
            
            const chips = Object.entries(searchCriteria).map(([key, value]) => {
                const displayName = {
                    title: 'Title',
                    authors: 'Authors',
                    journal: 'Journal',
                    year: 'Year',
                    keywords: 'Keywords',
                    abstract: 'Abstract'
                }[key] || key;
                
                return `
                    <div class="filter-chip active">
                        ${displayName}: ${value}
                        <button class="remove" onclick="removeFilter('${key}')">&times;</button>
                    </div>
                `;
            }).join('');
            
            chipsContainer.innerHTML = chips;
        }

        function removeFilter(filterKey) {
            delete searchCriteria[filterKey];
            document.getElementById(`search${filterKey.charAt(0).toUpperCase() + filterKey.slice(1)}`).value = '';
            
            if (Object.keys(searchCriteria).length === 0) {
                clearSearch();
            } else {
                performSearch();
            }
        }

        function updateSearchResultsInfo() {
            const infoElement = document.getElementById('searchResultsInfo');
            
            if (Object.keys(searchCriteria).length === 0) {
                infoElement.style.display = 'none';
                return;
            }
            
            infoElement.style.display = 'block';
            
            const totalRefs = originalReferences.length;
            const foundRefs = searchResults.length;
            
            if (foundRefs === 0) {
                infoElement.className = 'search-results-info no-results';
                infoElement.innerHTML = `
                    No abstracts found matching your search criteria. 
                    Try adjusting your search terms or clearing some filters.
                `;
            } else {
                infoElement.className = 'search-results-info';
                infoElement.innerHTML = `
                    Found ${foundRefs} of ${totalRefs} abstracts matching your search criteria.
                    ${foundRefs < totalRefs ? 'Use filters or modify your search to refine results.' : ''}
                `;
            }
        }

        function clearSearch() {
            // Clear search form
            document.getElementById('searchTitle').value = '';
            document.getElementById('searchAuthors').value = '';
            document.getElementById('searchJournal').value = '';
            document.getElementById('searchYear').value = '';
            document.getElementById('searchKeywords').value = '';
            document.getElementById('searchAbstract').value = '';
            
            // Reset variables
            searchCriteria = {};
            searchResults = [];
            references = [...originalReferences];
            
            // Update UI
            updateFilterChips();
            updateSearchResultsInfo();
            updateReferenceList();
            updateAbstractNavigator();
            updateMetricsPanel();
        }

        function applySorting() {
            const sortField = document.getElementById('sortBy').value;
            const isDescending = document.getElementById('sortDirection').classList.contains('desc');
            
            references.sort((a, b) => {
                let valueA, valueB;
                
                switch (sortField) {
                    case 'title':
                        valueA = (a.title || '').toLowerCase();
                        valueB = (b.title || '').toLowerCase();
                        break;
                    case 'authors':
                        valueA = (a.authors || '').toLowerCase();
                        valueB = (b.authors || '').toLowerCase();
                        break;
                    case 'year':
                        valueA = parseInt(a.year) || 0;
                        valueB = parseInt(b.year) || 0;
                        break;
                    case 'journal':
                        valueA = (a.journal || '').toLowerCase();
                        valueB = (b.journal || '').toLowerCase();
                        break;
                    case 'decision':
                        valueA = a.final_decision || 'pending';
                        valueB = b.final_decision || 'pending';
                        break;
                    case 'confidence':
                        valueA = getConfidenceScore(a) || 0;
                        valueB = getConfidenceScore(b) || 0;
                        break;
                    case 'relevance':
                    default:
                        valueA = a.relevance_score || 0;
                        valueB = b.relevance_score || 0;
                        break;
                }
                
                if (typeof valueA === 'string') {
                    const comparison = valueA.localeCompare(valueB);
                    return isDescending ? -comparison : comparison;
                } else {
                    const comparison = valueA - valueB;
                    return isDescending ? -comparison : comparison;
                }
            });
            
            // Update the reference list and navigator
            updateReferenceList();
            updateAbstractNavigator();
        }

        function toggleSortDirection() {
            const sortButton = document.getElementById('sortDirection');
            
            if (sortButton.classList.contains('desc')) {
                sortButton.classList.remove('desc');
                sortButton.textContent = '▲ Ascending';
            } else {
                sortButton.classList.add('desc');
                sortButton.textContent = '▼ Descending';
            }
            
            applySorting();
        }

        // Add search functionality to existing filter functions
        function applyFilter() {
            let baseReferences = searchResults.length > 0 ? searchResults : originalReferences;
            
            switch (currentFilter) {
                case 'include':
                    filteredReferences = baseReferences.filter(ref => ref.final_decision === 'include');
                    break;
                case 'exclude':
                    filteredReferences = baseReferences.filter(ref => ref.final_decision === 'exclude');
                    break;
                case 'conflict':
                    filteredReferences = baseReferences.filter(ref => ref.final_decision === 'conflict');
                    break;
                case 'pending':
                    filteredReferences = baseReferences.filter(ref => !ref.final_decision || ref.status !== 'completed');
                    break;
                default:
                    filteredReferences = [...baseReferences];
            }
            
            // Reset index if it's out of bounds
            if (currentAbstractIndex >= filteredReferences.length) {
                currentAbstractIndex = 0;
            }
        }

        // Enhanced renderAbstractCard to include search highlighting
        function renderAbstractCardWithHighlighting(ref) {
            const statusClass = getStatusClass(ref);
            const confidenceScore = getConfidenceScore(ref);
            
            // Apply search highlighting if there are active search criteria
            let title = ref.title || 'Untitled';
            let authors = ref.authors || 'Unknown authors';
            let journal = ref.journal || 'Unknown journal';
            let abstract = ref.abstract || 'No abstract available';
            
            if (Object.keys(searchCriteria).length > 0) {
                if (searchCriteria.title) title = highlightSearchTerms(title, searchCriteria.title);
                if (searchCriteria.authors) authors = highlightSearchTerms(authors, searchCriteria.authors);
                if (searchCriteria.journal) journal = highlightSearchTerms(journal, searchCriteria.journal);
                if (searchCriteria.abstract) abstract = highlightSearchTerms(abstract, searchCriteria.abstract);
            }
            
            return `
                <div class="abstract-card">
                    <div class="abstract-header">
                        <div class="abstract-title">${title}</div>
                        <div class="abstract-meta">
                            <strong>Authors:</strong> ${authors}<br>
                            <strong>Journal:</strong> ${journal} ${ref.year ? `(${ref.year})` : ''}<br>
                            ${ref.doi ? `<strong>DOI:</strong> ${ref.doi}<br>` : ''}
                            ${ref.keywords ? `<strong>Keywords:</strong> ${ref.keywords}<br>` : ''}
                        </div>
                        <div class="abstract-status">
                            <div class="status-badge ${statusClass}">${getStatusText(ref)}</div>
                            ${confidenceScore ? `<div class="confidence-score">${confidenceScore}% confidence</div>` : ''}
                        </div>
                    </div>
                    
                    <div class="abstract-content">
                        <div class="abstract-text">
                            ${abstract}
                        </div>
                        
                        ${renderAIEvaluation(ref)}
                    </div>
                </div>
            `;
        }

        // Update the displayCurrentAbstract function to use highlighting
        function displayCurrentAbstract() {
            const viewer = document.getElementById('abstractViewer');
            
            if (filteredReferences.length === 0) {
                viewer.innerHTML = `
                    <div class="no-evaluation">
                        <p>No abstracts match the current filter: "${currentFilter}"</p>
                        <p>Try selecting a different filter or complete more screening.</p>
                    </div>
                `;
                return;
            }
            
            const ref = filteredReferences[currentAbstractIndex];
            viewer.innerHTML = renderAbstractCardWithHighlighting(ref);
        }

        // Initialize
        updateStartButton();
    </script>
</body>
</html>
"""

# --- 8. Enhanced API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the enhanced frontend"""
    return HTML_CONTENT

@app.get("/api/providers")
async def get_providers():
    """Get available LLM providers and their configurations"""
    return {
        "providers": {
            provider_id: {
                "name": config.name,
                "display_name": config.display_name,
                "models": config.models,
                "default_model": config.default_model,
                "requires_api_key": config.requires_api_key,
                "default_endpoint": config.default_endpoint,
                "supports_streaming": config.supports_streaming
            }
            for provider_id, config in LLM_PROVIDERS.items()
        }
    }

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    """Enhanced file upload with better parsing"""
    project = Project(
        name=f"Otto-SR Project - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        screening_mode="single"
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    
    total_citations = 0
    
    for file in files:
        try:
            content = (await file.read()).decode('utf-8', errors='ignore')
            
            if file.filename and file.filename.endswith('.ris'):
                citations_data = parse_ris_file(content)
            elif file.filename and file.filename.endswith('.xml'):
                citations_data = parse_xml_file(content)
            else:
                continue
            
            for citation_data in citations_data:
                citation = CitationRecord(
                    project_id=project.id,
                    title=citation_data.get('title', 'No title'),
                    authors=citation_data.get('authors'),
                    journal=citation_data.get('journal'),
                    year=citation_data.get('year'),
                    abstract=citation_data.get('abstract'),
                    doi=citation_data.get('doi'),
                    keywords=citation_data.get('keywords'),
                    relevance_score=citation_data.get('relevance_score', 0.5)
                )
                db.add(citation)
                total_citations += 1
        
        except Exception as e:
            continue  # Skip problematic files
    
    db.commit()
    
    # Log activity
    activity = ActivityLog(
        project_id=project.id,
        action="files_uploaded",
        details={"file_count": len(files), "citations_count": total_citations}
    )
    db.add(activity)
    db.commit()
    
    return CitationUploadResponse(
        project_id=str(project.id),
        citations_count=total_citations,
        message=f"Successfully uploaded {total_citations} citations"
    )

@app.get("/api/projects/{project_id}/citations")
async def get_citations(project_id: str, db: Session = Depends(get_db)):
    """Get citations for a project"""
    citations = db.query(CitationRecord).filter(CitationRecord.project_id == project_id).all()
    
    results = []
    for citation in citations:
        # Get screening result if exists
        screening_result = db.query(ScreeningResult).filter(ScreeningResult.citation_id == citation.id).first()
        
        citation_data = {
            "id": str(citation.id),
            "title": citation.title,
            "authors": citation.authors,
            "journal": citation.journal,
            "year": citation.year,
            "abstract": citation.abstract,
            "relevance_score": citation.relevance_score,
            "status": screening_result.status if screening_result else "pending",
            "final_decision": screening_result.final_decision if screening_result else None,
            "ai1_result": screening_result.ai1_result if screening_result else None,
            "ai2_result": screening_result.ai2_result if screening_result else None
        }
        results.append(citation_data)
    
    return results

@app.post("/api/test-llm")
async def test_llm_connection(request: dict):
    """Test LLM connection"""
    config = request.get('config', {})
    
    try:
        # Simple test prompt
        test_prompt = "Respond with a valid JSON object containing 'status': 'success'"
        result = await call_llm_api(config, test_prompt)
        
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/projects/{project_id}/screen")
async def start_screening_job(
    project_id: str,
    request: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start enhanced screening job"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(404, "Project not found")
    
    # Update project with criteria and LLM configs
    project.criteria = request.get('criteria', {})
    project.screening_mode = request.get('mode', 'single')
    db.commit()
    
    # Get citations to screen
    citations = db.query(CitationRecord).filter(CitationRecord.project_id == project.id).all()
    if not citations:
        raise HTTPException(400, "No citations to screen")
    
    job_id = str(uuid.uuid4())
    llm_configs = request.get('llm_configs', {})
    
    # Create screening results and queue background tasks
    for citation in citations:
        result = ScreeningResult(
            project_id=project.id,
            citation_id=citation.id,
            job_id=job_id,
            status="pending"
        )
        db.add(result)
        db.commit()
        
        # Queue background screening task
        background_tasks.add_task(
            advanced_screening_task,
            str(result.id),
            job_id,
            llm_configs
        )
    
    # Log activity
    activity = ActivityLog(
        project_id=project.id,
        action="screening_started",
        details={
            "job_id": job_id,
            "mode": project.screening_mode,
            "citations_count": len(citations)
        }
    )
    db.add(activity)
    db.commit()
    
    return {"job_id": job_id, "message": "Screening started", "citations_count": len(citations)}

@app.get("/api/stream/{job_id}")
async def stream_progress(job_id: str):
    """Stream screening progress"""
    async def event_generator():
        processed_ids = set()
        
        while True:
            db_session = SessionLocal()
            try:
                # Get completed results
                results = db_session.query(ScreeningResult).filter(
                    ScreeningResult.job_id == job_id,
                    ~ScreeningResult.id.in_(processed_ids),
                    ScreeningResult.status.in_(['completed', 'error'])
                ).all()
                
                for result in results:
                    citation = db_session.query(CitationRecord).filter(CitationRecord.id == result.citation_id).first()
                    
                    progress_data = {
                        "citation_id": str(result.citation_id),
                        "status": result.status,
                        "final_decision": result.final_decision,
                        "confidence_score": result.confidence_score,
                        "ai1_result": result.ai1_result,
                        "ai2_result": result.ai2_result,
                        "title": citation.title if citation else "Unknown"
                    }
                    
                    yield f"event: progress\ndata: {json.dumps(progress_data)}\n\n"
                    processed_ids.add(result.id)
                
                # Check if job is complete
                total_results = db_session.query(ScreeningResult).filter(ScreeningResult.job_id == job_id).count()
                if total_results > 0 and len(processed_ids) >= total_results:
                    yield f"event: complete\ndata: {{\"message\": \"Screening completed\"}}\n\n"
                    break
                    
            except Exception as e:
                yield f"event: error\ndata: {{\"error\": \"{str(e)}\"}}\n\n"
                break
            finally:
                db_session.close()
            
            await asyncio.sleep(2)  # Check every 2 seconds
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/projects/{project_id}/export")
async def export_results(project_id: str, db: Session = Depends(get_db)):
    """Export screening results"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(404, "Project not found")
    
    citations = db.query(CitationRecord).filter(CitationRecord.project_id == project_id).all()
    results = []
    
    for citation in citations:
        screening_result = db.query(ScreeningResult).filter(ScreeningResult.citation_id == citation.id).first()
        
        result_data = {
            "citation": {
                "id": str(citation.id),
                "title": citation.title,
                "authors": citation.authors,
                "journal": citation.journal,
                "year": citation.year,
                "abstract": citation.abstract,
                "doi": citation.doi,
                "keywords": citation.keywords
            },
            "screening": {
                "status": screening_result.status if screening_result else "pending",
                "final_decision": screening_result.final_decision if screening_result else None,
                "confidence_score": screening_result.confidence_score if screening_result else None,
                "ai1_result": screening_result.ai1_result if screening_result else None,
                "ai2_result": screening_result.ai2_result if screening_result else None,
                "human_decision": screening_result.human_decision if screening_result else None,
                "notes": screening_result.notes if screening_result else None,
                "processed_at": screening_result.processed_at.isoformat() if screening_result and screening_result.processed_at else None
            }
        }
        results.append(result_data)
    
    export_data = {
        "project": {
            "id": str(project.id),
            "name": project.name,
            "criteria": project.criteria,
            "created_at": project.created_at.isoformat(),
            "screening_mode": project.screening_mode
        },
        "results": results,
        "summary": {
            "total_citations": len(results),
            "processed": len([r for r in results if r["screening"]["status"] in ["completed", "error"]]),
            "included": len([r for r in results if r["screening"]["final_decision"] == "include"]),
            "excluded": len([r for r in results if r["screening"]["final_decision"] == "exclude"]),
            "conflicts": len([r for r in results if r["screening"]["final_decision"] == "conflict"])
        },
        "exported_at": datetime.utcnow().isoformat()
    }
    
    return JSONResponse(content=export_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)