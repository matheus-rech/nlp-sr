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

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File
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
from pydantic import BaseModel as PydanticBaseModel

# --- Configuration & Initialization ---

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/ottosr")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI(
    title="Otto-SR Production Tool v3.0",
    description="Advanced systematic review screening with multiple LLM providers",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Models ---

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
    ai1_result = Column(JSON)
    ai2_result = Column(JSON)
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

# Create tables
Base.metadata.create_all(bind=engine)

# --- Database Dependency ---

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Pydantic Models ---

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

class ProviderConfig(BaseModel):
    name: str
    display_name: str
    models: List[str]
    default_model: str
    requires_api_key: bool
    default_endpoint: Optional[str] = None
    supports_streaming: bool = True

class LLMConfig(BaseModel):
    provider: str
    model: str
    endpoint: str
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2000
    cost_per_token: float = 0.0
    quality_score: float = 0.85
    response_time_avg: float = 0.0
    success_rate: float = 1.0
    specialization: List[str] = []

class ScreeningDecision(PydanticBaseModel):
    """Enhanced structured screening decision output"""
    decision: Literal["include", "exclude", "uncertain"] = Field(
        description="Final decision: 'include' to include the study, 'exclude' to exclude it, 'uncertain' for review"
    )
    confidence: float = Field(
        ge=0, le=100, 
        description="Confidence percentage from 0-100"
    )
    reasoning: str = Field(
        description="Detailed explanation of the decision with specific criteria references"
    )
    evidence_quotes: List[str] = Field(
        description="Specific quotes from the citation supporting the decision",
        default_factory=list
    )
    criteria_assessment: Dict[str, str] = Field(
        description="Assessment of each criterion (meets/does_not_meet/unclear)",
        default_factory=lambda: {
            "population": "unclear",
            "intervention": "unclear",
            "comparison": "unclear",
            "outcome": "unclear",
            "study_design": "unclear"
        }
    )
    quality_indicators: Dict[str, bool] = Field(
        description="Quality assessment indicators",
        default_factory=lambda: {
            "sample_size_adequate": False,
            "methodology_clear": False,
            "outcomes_relevant": False
        }
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
    processing_metadata: Dict[str, Any] = Field(
        description="Processing metadata including timing and model info",
        default_factory=dict
    )

class CitationUploadResponse(BaseModel):
    project_id: str
    citations_count: int
    message: str
    citations: Optional[List[Dict[str, Any]]] = None
    status: str = "success"

# --- Utility Functions ---

def parse_ris_file(content: str) -> List[Dict[str, Any]]:
    """Robust RIS parser with comprehensive field extraction"""
    citations = []
    current_citation = {}
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('ER  -'):
            if current_citation:
                # Calculate relevance score based on completeness
                score = 0.3  # base score
                if current_citation.get('title'): score += 0.3
                if current_citation.get('abstract'): score += 0.2
                if current_citation.get('authors'): score += 0.1
                if current_citation.get('year'): score += 0.1
                current_citation['relevance_score'] = min(score, 1.0)
                
                citations.append(current_citation)
                current_citation = {}
            continue
            
        if '  - ' in line:
            field, value = line.split('  - ', 1)
            
            if field == 'TI':
                current_citation['title'] = value
            elif field == 'AU':
                if 'authors' not in current_citation:
                    current_citation['authors'] = value
                else:
                    current_citation['authors'] += '; ' + value
            elif field == 'JO' or field == 'T2':
                current_citation['journal'] = value
            elif field == 'PY':
                try:
                    current_citation['year'] = int(value[:4])
                except:
                    current_citation['year'] = None
            elif field == 'AB':
                current_citation['abstract'] = value
            elif field == 'DO':
                current_citation['doi'] = value
            elif field == 'KW':
                if 'keywords' not in current_citation:
                    current_citation['keywords'] = value
                else:
                    current_citation['keywords'] += '; ' + value
    
    return citations

class PerformanceTracker:
    """Track LLM performance metrics for optimization"""
    
    def __init__(self):
        self.metrics = {
            'responses': [],
            'providers': {},
            'session_start': datetime.utcnow()
        }
    
    def record_response(self, provider: str, model: str, response_time: float, 
                       tokens: int, success: bool, confidence: float = 0.0):
        """Record LLM response metrics"""
        record = {
            'provider': provider,
            'model': model,
            'response_time': response_time,
            'tokens': tokens,
            'success': success,
            'confidence': confidence,
            'timestamp': datetime.utcnow()
        }
        
        self.metrics['responses'].append(record)
        
        if provider not in self.metrics['providers']:
            self.metrics['providers'][provider] = {
                'total_calls': 0,
                'successful_calls': 0,
                'total_time': 0.0,
                'total_tokens': 0,
                'avg_confidence': 0.0
            }
        
        provider_metrics = self.metrics['providers'][provider]
        provider_metrics['total_calls'] += 1
        provider_metrics['total_time'] += response_time
        provider_metrics['total_tokens'] += tokens
        
        if success:
            provider_metrics['successful_calls'] += 1
            
        # Update rolling average confidence
        if confidence > 0:
            current_avg = provider_metrics['avg_confidence']
            total_successful = provider_metrics['successful_calls']
            provider_metrics['avg_confidence'] = (
                (current_avg * (total_successful - 1) + confidence) / total_successful
                if total_successful > 0 else confidence
            )
    
    def get_analytics(self) -> Dict[str, Any]:
        """Generate comprehensive performance analytics"""
        total_responses = len(self.metrics['responses'])
        successful_responses = sum(1 for r in self.metrics['responses'] if r['success'])
        
        analytics = {
            'session_duration': (datetime.utcnow() - self.metrics['session_start']).total_seconds(),
            'total_responses': total_responses,
            'success_rate': successful_responses / total_responses if total_responses > 0 else 0,
            'avg_response_time': sum(r['response_time'] for r in self.metrics['responses']) / total_responses if total_responses > 0 else 0,
            'providers': {}
        }
        
        for provider, metrics in self.metrics['providers'].items():
            analytics['providers'][provider] = {
                'success_rate': metrics['successful_calls'] / metrics['total_calls'] if metrics['total_calls'] > 0 else 0,
                'avg_response_time': metrics['total_time'] / metrics['total_calls'] if metrics['total_calls'] > 0 else 0,
                'avg_confidence': metrics['avg_confidence'],
                'total_calls': metrics['total_calls'],
                'total_tokens': metrics['total_tokens']
            }
        
        return analytics

class ConflictResolver:
    """Advanced conflict resolution for dual LLM disagreements"""
    
    @staticmethod
    def calculate_agreement_score(result1: Dict, result2: Dict) -> float:
        """Calculate sophisticated agreement score between two LLM results"""
        agreement_factors = []
        
        # Decision agreement (most important)
        if result1.get('decision') == result2.get('decision'):
            agreement_factors.append(0.4)  # 40% weight
        
        # Confidence alignment
        conf1 = result1.get('confidence', 0)
        conf2 = result2.get('confidence', 0)
        conf_diff = abs(conf1 - conf2) / 100.0
        agreement_factors.append((1 - conf_diff) * 0.2)  # 20% weight
        
        # PICO scores alignment
        pico1 = result1.get('pico_scores', {})
        pico2 = result2.get('pico_scores', {})
        pico_agreement = 0
        pico_count = 0
        
        for component in ['population', 'intervention', 'comparison', 'outcome']:
            if component in pico1 and component in pico2:
                pico_agreement += 1 - abs(pico1[component] - pico2[component])
                pico_count += 1
        
        if pico_count > 0:
            agreement_factors.append((pico_agreement / pico_count) * 0.2)  # 20% weight
        
        # Quality assessment agreement
        if result1.get('quality_assessment') == result2.get('quality_assessment'):
            agreement_factors.append(0.2)  # 20% weight
        
        return sum(agreement_factors)
    
    @staticmethod
    def resolve_conflict(ai1_result: Dict, ai2_result: Dict, citation: Dict) -> Dict[str, Any]:
        """Intelligent conflict resolution"""
        agreement_score = ConflictResolver.calculate_agreement_score(ai1_result, ai2_result)
        
        resolution = {
            'agreement_score': agreement_score,
            'resolution_method': '',
            'final_decision': '',
            'confidence': 0,
            'reasoning': '',
            'metadata': {
                'ai1_confidence': ai1_result.get('confidence', 0),
                'ai2_confidence': ai2_result.get('confidence', 0),
                'conflict_type': 'decision_mismatch'
            }
        }
        
        # High agreement (>80%) - take consensus
        if agreement_score > 0.8:
            resolution['resolution_method'] = 'consensus'
            resolution['final_decision'] = ai1_result.get('decision', 'uncertain')
            resolution['confidence'] = (ai1_result.get('confidence', 0) + ai2_result.get('confidence', 0)) / 2
            resolution['reasoning'] = f"High agreement ({agreement_score:.2f}) between AI models. Consensus decision: {resolution['final_decision']}"
        
        # Moderate agreement (60-80%) - higher confidence wins
        elif agreement_score > 0.6:
            resolution['resolution_method'] = 'higher_confidence'
            if ai1_result.get('confidence', 0) > ai2_result.get('confidence', 0):
                resolution['final_decision'] = ai1_result.get('decision', 'uncertain')
                resolution['confidence'] = ai1_result.get('confidence', 0)
                resolution['reasoning'] = f"Moderate agreement ({agreement_score:.2f}). AI1 has higher confidence ({ai1_result.get('confidence', 0)}%)"
            else:
                resolution['final_decision'] = ai2_result.get('decision', 'uncertain')
                resolution['confidence'] = ai2_result.get('confidence', 0)
                resolution['reasoning'] = f"Moderate agreement ({agreement_score:.2f}). AI2 has higher confidence ({ai2_result.get('confidence', 0)}%)"
        
        # Low agreement (<60%) - flag for human review
        else:
            resolution['resolution_method'] = 'conflict_detected'
            resolution['final_decision'] = 'conflict'
            resolution['confidence'] = 0
            resolution['reasoning'] = f"Low agreement ({agreement_score:.2f}) between AI models. Human review required."
            resolution['metadata']['conflict_type'] = 'major_disagreement'
        
        return resolution

# --- File Parsing Functions ---

def parse_xml_file(content: str) -> List[Dict[str, Any]]:
    """Enhanced XML parser supporting multiple formats"""
    citations = []
    
    try:
        root = ET.fromstring(content)
        
        # Handle different XML namespaces and structures
        for record in root.findall(".//record") + root.findall(".//citation") + root.findall(".//ref"):
            citation = {}
            
            # Try multiple field mappings
            title_elem = (record.find(".//title") or 
                         record.find(".//article-title") or 
                         record.find(".//atitle"))
            if title_elem is not None:
                citation['title'] = title_elem.text
            
            # Authors
            authors = []
            for author in record.findall(".//author") + record.findall(".//name"):
                if author.text:
                    authors.append(author.text)
            if authors:
                citation['authors'] = '; '.join(authors)
            
            # Journal
            journal_elem = (record.find(".//journal") or 
                          record.find(".//source") or 
                          record.find(".//jtitle"))
            if journal_elem is not None:
                citation['journal'] = journal_elem.text
            
            # Year
            year_elem = (record.find(".//year") or 
                        record.find(".//date") or 
                        record.find(".//pub-date"))
            if year_elem is not None:
                try:
                    citation['year'] = int(re.search(r'\d{4}', year_elem.text).group())
                except:
                    pass
            
            # Abstract
            abstract_elem = record.find(".//abstract")
            if abstract_elem is not None:
                citation['abstract'] = abstract_elem.text
            
            # DOI
            doi_elem = record.find(".//doi")
            if doi_elem is not None:
                citation['doi'] = doi_elem.text
            
            if citation:
                # Calculate relevance score
                score = 0.3
                if citation.get('title'): score += 0.3
                if citation.get('abstract'): score += 0.2
                if citation.get('authors'): score += 0.1
                if citation.get('year'): score += 0.1
                citation['relevance_score'] = min(score, 1.0)
                citations.append(citation)
    
    except ET.ParseError:
        pass
    
    return citations

def parse_endnote_file(content: str) -> List[Dict[str, Any]]:
    """Parse EndNote tagged format (.enw) or XML format"""
    if content.strip().startswith('<'):
        return parse_xml_file(content)
    
    citations = []
    current_citation = {}
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('%0'):
            if current_citation:
                citations.append(current_citation)
                current_citation = {}
        elif line.startswith('%T'):
            current_citation['title'] = line[3:]
        elif line.startswith('%A'):
            if 'authors' not in current_citation:
                current_citation['authors'] = line[3:]
            else:
                current_citation['authors'] += '; ' + line[3:]
        elif line.startswith('%J'):
            current_citation['journal'] = line[3:]
        elif line.startswith('%D'):
            try:
                current_citation['year'] = int(line[3:])
            except:
                pass
        elif line.startswith('%X'):
            current_citation['abstract'] = line[3:]
        elif line.startswith('%R'):
            current_citation['doi'] = line[3:]
        elif line.startswith('%K'):
            if 'keywords' not in current_citation:
                current_citation['keywords'] = line[3:]
            else:
                current_citation['keywords'] += '; ' + line[3:]
    
    if current_citation:
        citations.append(current_citation)
    
    # Add relevance scores
    for citation in citations:
        score = 0.3
        if citation.get('title'): score += 0.3
        if citation.get('abstract'): score += 0.2
        if citation.get('authors'): score += 0.1
        if citation.get('year'): score += 0.1
        citation['relevance_score'] = min(score, 1.0)
    
    return citations

def parse_mendeley_file(content: str) -> List[Dict[str, Any]]:
    """Parse Mendeley BibTeX format (.bib)"""
    citations = []
    
    # Simple BibTeX parser
    entries = re.findall(r'@\w+\{[^}]+,([^}]+)\}', content, re.DOTALL)
    
    for entry in entries:
        citation = {}
        
        # Extract fields
        title_match = re.search(r'title\s*=\s*["{]([^"}]+)["}]', entry, re.IGNORECASE)
        if title_match:
            citation['title'] = title_match.group(1)
        
        author_match = re.search(r'author\s*=\s*["{]([^"}]+)["}]', entry, re.IGNORECASE)
        if author_match:
            citation['authors'] = author_match.group(1)
        
        journal_match = re.search(r'journal\s*=\s*["{]([^"}]+)["}]', entry, re.IGNORECASE)
        if journal_match:
            citation['journal'] = journal_match.group(1)
        
        year_match = re.search(r'year\s*=\s*["{]?(\d{4})["}]?', entry, re.IGNORECASE)
        if year_match:
            citation['year'] = int(year_match.group(1))
        
        abstract_match = re.search(r'abstract\s*=\s*["{]([^"}]+)["}]', entry, re.IGNORECASE)
        if abstract_match:
            citation['abstract'] = abstract_match.group(1)
        
        doi_match = re.search(r'doi\s*=\s*["{]([^"}]+)["}]', entry, re.IGNORECASE)
        if doi_match:
            citation['doi'] = doi_match.group(1)
        
        keywords_match = re.search(r'keywords\s*=\s*["{]([^"}]+)["}]', entry, re.IGNORECASE)
        if keywords_match:
            citation['keywords'] = keywords_match.group(1)
        
        if citation:
            # Calculate relevance score
            score = 0.3
            if citation.get('title'): score += 0.3
            if citation.get('abstract'): score += 0.2
            if citation.get('authors'): score += 0.1
            if citation.get('year'): score += 0.1
            citation['relevance_score'] = min(score, 1.0)
            citations.append(citation)
    
    return citations

def parse_zotero_file(content: str) -> List[Dict[str, Any]]:
    """Parse Zotero RDF format (.rdf) or CSL JSON format"""
    citations = []
    
    if content.strip().startswith('{') or content.strip().startswith('['):
        # JSON format
        try:
            data = json.loads(content)
            if isinstance(data, list):
                items = data
            else:
                items = data.get('items', [])
            
            for item in items:
                citation = {
                    'title': item.get('title', ''),
                    'journal': item.get('container-title', ''),
                    'year': item.get('issued', {}).get('date-parts', [[None]])[0][0] if item.get('issued') else None,
                    'abstract': item.get('abstract', ''),
                    'doi': item.get('DOI', ''),
                }
                
                # Authors
                authors = item.get('author', [])
                if authors:
                    author_names = []
                    for author in authors:
                        if 'family' in author and 'given' in author:
                            author_names.append(f"{author['given']} {author['family']}")
                        elif 'literal' in author:
                            author_names.append(author['literal'])
                    citation['authors'] = '; '.join(author_names)
                
                if citation.get('title'):
                    # Calculate relevance score
                    score = 0.3
                    if citation.get('title'): score += 0.3
                    if citation.get('abstract'): score += 0.2
                    if citation.get('authors'): score += 0.1
                    if citation.get('year'): score += 0.1
                    citation['relevance_score'] = min(score, 1.0)
                    citations.append(citation)
        except:
            pass
    else:
        # RDF format - parse as XML
        citations = parse_xml_file(content)
    
    return citations

# --- LLM Provider Factory ---

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
                api_key=config.api_key or os.getenv("OPENAI_API_KEY")
            )
        elif provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(
                    model=config.model,
                    temperature=config.temperature,
                    api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY")
                )
            except ImportError:
                # Fallback to OpenAI-compatible for Anthropic
                return ChatOpenAI(
                    model=config.model,
                    temperature=config.temperature,
                    base_url="https://api.anthropic.com/v1",
                    api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY")
                )
        elif provider == "ollama":
            return ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                base_url=config.endpoint or "http://localhost:11434/v1",
                api_key="ollama"
            )
        elif provider == "groq":
            return ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                base_url=config.endpoint or "https://api.groq.com/openai/v1",
                api_key=config.api_key or os.getenv("GROQ_API_KEY")
            )
        elif provider == "together":
            return ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                base_url=config.endpoint or "https://api.together.xyz/v1",
                api_key=config.api_key or os.getenv("TOGETHER_API_KEY")
            )
        elif provider in ["openai_compatible", "custom", "cohere"]:
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

# --- Prompt Template Factory ---

class PromptTemplateFactory:
    """Factory for creating dynamic prompt templates"""
    
    @staticmethod
    def create_screening_prompt(strategy: Literal["conservative", "pragmatic"]) -> ChatPromptTemplate:
        """Create LangChain prompt template for screening"""
        if strategy == "conservative":
            system_message = """You are a conservative systematic review screening AI. Your role is to be highly selective and only include studies that clearly and unambiguously meet all inclusion criteria. When in doubt, exclude the study.

Screening Guidelines:
- Apply strict interpretation of inclusion/exclusion criteria
- Require clear evidence for all PICO components
- Exclude studies with any methodological concerns
- Prefer high-quality study designs (RCTs, systematic reviews)
- Be cautious about studies with limited sample sizes or unclear methodology"""
        else:
            system_message = """You are a pragmatic systematic review screening AI. Your role is to be inclusive and capture potentially relevant studies that could contribute to the review. When criteria are ambiguous, err on the side of inclusion.

Screening Guidelines:
- Apply flexible interpretation of inclusion/exclusion criteria
- Include studies that partially meet PICO components if potentially relevant
- Consider various study designs and methodologies
- Include studies that could provide valuable insights even with limitations
- Focus on potential contribution to the research question"""
        
        human_message = """Please screen this citation for systematic review inclusion based on the provided criteria.

**Research Question**: {research_question}

**PICO Criteria**:
- Population: {population}
- Intervention: {intervention}
- Comparison: {comparison}
- Outcome: {outcome}

**Additional Criteria**:
- Study Types: {study_types}
- Timeframe: {timeframe}
- Language: {inclusion_language}
- Other Inclusion: {other_inclusion}
- Other Exclusion: {other_exclusion}

**Citation to Screen**:
Title: {title}
Authors: {authors}
Journal: {journal}
Year: {year}
Abstract: {abstract}
Keywords: {keywords}

{format_instructions}

Provide your screening decision with detailed reasoning."""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])

def create_advanced_prompt(criteria: AdvancedScreeningCriteria, strategy: Literal["conservative", "pragmatic"]) -> ChatPromptTemplate:
    """Create sophisticated prompts based on screening strategy"""
    return PromptTemplateFactory.create_screening_prompt(strategy)

# --- Advanced Screening with Multiple Providers ---

performance_tracker = PerformanceTracker()

async def advanced_screening_task(result_id: str, job_id: str, llm_configs: Dict[str, Any]):
    """Advanced screening with multiple LLM provider support"""
    db = SessionLocal()
    try:
        # Get the screening result record
        result = db.query(ScreeningResult).filter(ScreeningResult.id == result_id).first()
        if not result:
            return
        
        # Get the citation
        citation = db.query(CitationRecord).filter(CitationRecord.id == result.citation_id).first()
        if not citation:
            return
        
        # Get project criteria
        project = db.query(Project).filter(Project.id == result.project_id).first()
        if not project:
            return
        
        # Update status to processing
        result.status = "processing"
        db.commit()
        
        # Extract criteria
        criteria_dict = project.criteria or {}
        criteria = AdvancedScreeningCriteria(**criteria_dict)
        
        # Create prompt templates
        conservative_template = create_advanced_prompt(criteria, "conservative")
        pragmatic_template = create_advanced_prompt(criteria, "pragmatic")
        
        # Prepare prompt variables
        prompt_variables = {
            "research_question": criteria.researchQuestion,
            "population": criteria.population,
            "intervention": criteria.intervention,
            "comparison": criteria.comparison,
            "outcome": criteria.outcome,
            "study_types": criteria.studyTypes,
            "timeframe": criteria.timeframe,
            "inclusion_language": criteria.inclusionLanguage,
            "other_inclusion": criteria.otherInclusion,
            "other_exclusion": criteria.otherExclusion,
            "title": citation.title or "",
            "authors": citation.authors or "",
            "journal": citation.journal or "",
            "year": str(citation.year) if citation.year else "",
            "abstract": citation.abstract or "",
            "keywords": citation.keywords or ""
        }
        
        # Get LLM configurations
        ai1_config = llm_configs.get('ai1', {})
        ai2_config = llm_configs.get('ai2', {})
        
        # Call both LLMs
        ai1_result = await call_llm_api(ai1_config, conservative_template, prompt_variables)
        ai2_result = await call_llm_api(ai2_config, pragmatic_template, prompt_variables)
        
        # Store results
        result.ai1_result = ai1_result
        result.ai2_result = ai2_result
        
        # Resolve conflicts
        if ai1_result.get('decision') == ai2_result.get('decision'):
            # Agreement - use consensus
            result.final_decision = ai1_result.get('decision')
            result.confidence_score = (ai1_result.get('confidence', 0) + ai2_result.get('confidence', 0)) / 200.0  # Convert to 0-1 scale
        else:
            # Conflict - use resolver
            resolution = ConflictResolver.resolve_conflict(ai1_result, ai2_result, {
                'title': citation.title,
                'abstract': citation.abstract
            })
            result.final_decision = resolution['final_decision']
            result.confidence_score = resolution['confidence'] / 100.0  # Convert to 0-1 scale
        
        result.status = "completed"
        result.processed_at = datetime.utcnow()
        
        db.commit()
        
    except Exception as e:
        # Handle errors
        if 'result' in locals():
            result.status = "error"
            result.ai1_result = {"error": str(e)}
            result.ai2_result = {"error": str(e)}
            db.commit()
    finally:
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
                "evidence_quotes": structured_result.evidence_quotes,
                "criteria_assessment": structured_result.criteria_assessment,
                "quality_indicators": structured_result.quality_indicators,
                "pico_scores": structured_result.pico_scores,
                "study_design": structured_result.study_design,
                "quality_assessment": structured_result.quality_assessment,
                "key_findings": structured_result.key_findings,
                "processing_metadata": structured_result.processing_metadata
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
            "evidence_quotes": [],
            "criteria_assessment": {},
            "quality_indicators": {},
            "pico_scores": {"population": 0, "intervention": 0, "comparison": 0, "outcome": 0},
            "study_design": "Unknown",
            "quality_assessment": "unclear",
            "key_findings": [],
            "processing_metadata": {},
            "error": True
        }

def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Parse LLM response with fallback handling"""
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed
    except:
        pass
    
    # Fallback parsing
    decision = "uncertain"
    confidence = 50
    reasoning = response_text[:500] if response_text else "No response received"
    
    # Try to extract decision
    if any(word in response_text.lower() for word in ["include", "accept", "yes"]):
        decision = "include"
        confidence = 70
    elif any(word in response_text.lower() for word in ["exclude", "reject", "no"]):
        decision = "exclude"
        confidence = 70
    
    return {
        "decision": decision,
        "confidence": confidence,
        "reasoning": reasoning,
        "evidence_quotes": [],
        "criteria_assessment": {},
        "quality_indicators": {},
        "pico_scores": {"population": 0.5, "intervention": 0.5, "comparison": 0.5, "outcome": 0.5},
        "study_design": "Unknown",
        "quality_assessment": "unclear",
        "key_findings": [],
        "processing_metadata": {},
        "fallback_parsing": True
    }

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the enhanced frontend"""
    return HTMLResponse("""
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

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .header h1 {
            font-size: 1.8rem;
            font-weight: 300;
        }

        .header p {
            opacity: 0.9;
            margin-top: 0.5rem;
        }

        .container {
            display: flex;
            flex: 1;
            overflow: hidden;
            min-height: 0;
        }

        .left-panel {
            width: 350px;
            background: white;
            border-right: 1px solid #e0e0e0;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            height: calc(100vh - 120px);
        }

        .criteria-section {
            padding: 1.5rem;
            border-bottom: 1px solid #e0e0e0;
            overflow-y: auto;
            max-height: 40vh;
        }

        .criteria-section h3 {
            margin-bottom: 1rem;
            color: #333;
            font-size: 1.1rem;
        }

        .form-group {
            margin-bottom: 1rem;
        }

        .form-group label {
            display: block;
            margin-bottom: 0.3rem;
            font-weight: 500;
            color: #555;
            font-size: 0.9rem;
        }

        .form-group input,
        .form-group textarea,
        .form-group select {
            width: 100%;
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 0.9rem;
        }

        .form-group textarea {
            height: 60px;
            resize: vertical;
        }

        .llm-config {
            padding: 1.5rem;
            border-bottom: 1px solid #e0e0e0;
            overflow-y: auto;
            max-height: 30vh;
        }

        .llm-config h4 {
            color: #333;
            margin-bottom: 1rem;
            font-size: 1rem;
        }

        .config-section {
            margin-bottom: 1.5rem;
            padding: 1rem;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            background: #fafafa;
        }

        .config-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }

        .config-row label {
            font-weight: 500;
            color: #555;
            font-size: 0.85rem;
        }

        .config-row select,
        .config-row input {
            width: 180px;
            padding: 0.4rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 0.85rem;
        }

        .test-button {
            padding: 0.5rem 1rem;
            background: #17a2b8;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: background 0.2s;
        }

        .test-button:hover {
            background: #138496;
        }

        .api-status {
            font-size: 0.8rem;
            padding: 0.2rem 0.5rem;
            border-radius: 3px;
            font-weight: 500;
        }

        .connected {
            background: #d4edda;
            color: #155724;
        }

        .disconnected {
            background: #f8d7da;
            color: #721c24;
        }

        .testing {
            background: #fff3cd;
            color: #856404;
        }

        .controls {
            padding: 1.5rem;
            background: #f8f9fa;
            border-top: 1px solid #e0e0e0;
            min-height: 200px;
        }

        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 1rem;
            cursor: pointer;
            transition: border-color 0.3s;
        }

        .upload-area:hover {
            border-color: #667eea;
        }

        .upload-area.dragover {
            border-color: #667eea;
            background-color: #f0f4ff;
        }

        .upload-area input {
            display: none;
        }

        .mode-selection {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .mode-button {
            flex: 1;
            padding: 0.5rem;
            border: 1px solid #ddd;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85rem;
            text-align: center;
            transition: all 0.2s;
        }

        .mode-button.active {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }

        .ai-button {
            width: 100%;
            padding: 0.75rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            transition: transform 0.2s;
        }

        .ai-button:hover:not(:disabled) {
            transform: translateY(-1px);
        }

        .ai-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .button-row {
            display: flex;
            gap: 0.5rem;
        }

        .button-row button {
            flex: 1;
            padding: 0.5rem;
            border: 1px solid #ddd;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85rem;
        }

        .button-row button:hover:not(:disabled) {
            background: #f8f9fa;
        }

        .button-row button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        /* Citations Section */
        .citations-section {
            margin: 1rem;
            padding: 1.5rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #fefefe;
        }

        .citations-controls {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
            align-items: center;
            flex-wrap: wrap;
        }

        .search-box {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }

        .search-box input {
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            width: 300px;
        }

        .sort-controls {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }

        .citations-carousel {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1rem;
            max-height: 600px;
            overflow-y: auto;
            border: 1px solid #eee;
            border-radius: 6px;
            padding: 1rem;
        }

        .citation-card {
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 1rem;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .citation-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }

        .citation-title {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 0.5rem;
            font-size: 1rem;
            line-height: 1.3;
        }

        .citation-meta {
            color: #7f8c8d;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }

        .citation-abstract {
            color: #34495e;
            font-size: 0.85rem;
            line-height: 1.4;
            margin-bottom: 0.5rem;
            max-height: 80px;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 4;
            -webkit-box-orient: vertical;
        }

        .citation-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 0.5rem;
            padding-top: 0.5rem;
            border-top: 1px solid #eee;
        }

        .relevance-score {
            background-color: #3498db;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
        }

        .citation-status {
            padding: 0.2rem 0.5rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
        }

        .status-ready {
            background-color: #2ecc71;
            color: white;
        }

        .status-processing {
            background-color: #f39c12;
            color: white;
        }

        .status-completed {
            background-color: #27ae60;
            color: white;
        }

        /* Metrics Panel */
        .metrics-panel {
            margin: 1rem;
            padding: 1.5rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #f8f9fa;
        }

        .export-btn {
            width: 100%;
            padding: 0.75rem;
            background: #28a745;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }

        .export-btn:hover {
            background: #218838;
        }

        /* Modal Styles */
        .modal {
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .modal-content {
            background-color: white;
            margin: auto;
            padding: 0;
            border-radius: 8px;
            width: 90%;
            max-width: 1000px;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }

        .modal-header {
            padding: 1.5rem;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #f8f9fa;
            border-radius: 8px 8px 0 0;
        }

        .modal-header h3 {
            margin: 0;
            color: #333;
        }

        .close {
            color: #aaa;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            line-height: 1;
        }

        .close:hover {
            color: #333;
        }

        .modal-body {
            padding: 1.5rem;
        }

        .citation-detail {
            margin-bottom: 2rem;
        }

        .citation-detail h4 {
            color: #333;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #007bff;
        }

        .citation-meta-detail {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 1.5rem;
        }

        .ai-analysis {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-top: 1.5rem;
        }

        .ai-result {
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 1rem;
        }

        .ai-result h5 {
            margin-top: 0;
            color: #333;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e0e0e0;
        }

        .decision-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .decision-include {
            background: #d4edda;
            color: #155724;
        }

        .decision-exclude {
            background: #f8d7da;
            color: #721c24;
        }

        .decision-uncertain {
            background: #fff3cd;
            color: #856404;
        }

        .confidence-score {
            background: #e9ecef;
            padding: 0.5rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }

        .reasoning-text {
            background: #f8f9fa;
            padding: 1rem;
            border-left: 4px solid #007bff;
            margin: 1rem 0;
            font-style: italic;
        }

        .evidence-quotes {
            background: #fff3cd;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }

        .evidence-quotes ul {
            margin: 0.5rem 0 0 0;
            padding-left: 1.5rem;
        }

        .pico-scores {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 0.5rem;
            margin: 1rem 0;
        }

        .pico-score {
            text-align: center;
            padding: 0.5rem;
            background: #f8f9fa;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
        }

        .pico-score-value {
            font-weight: bold;
            font-size: 1.1rem;
            color: #007bff;
        }

        .pico-score-label {
            font-size: 0.8rem;
            color: #666;
            text-transform: uppercase;
        }

        .view-details-btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.8rem;
            cursor: pointer;
            margin-left: 0.5rem;
        }

        .view-details-btn:hover {
            background: #0056b3;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }

        .metric-card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        .metric-card h4 {
            margin: 0 0 1rem 0;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }

        .stat-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            padding: 0.3rem 0;
        }

        .stat-label {
            color: #7f8c8d;
            font-weight: 500;
        }

        .stat-value {
            font-weight: bold;
            color: #2c3e50;
        }

        .export-controls {
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
        }

        .export-controls .config-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 0;
        }

        .export-controls label {
            font-weight: 500;
            color: #2c3e50;
        }

        .export-controls select,
        .export-controls input[type="checkbox"] {
            margin-left: 0.5rem;
        }

        .main-content {
            flex: 1;
            padding: 1rem;
            overflow-y: auto;
            background: #f5f5f5;
        }

        .sidebar {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border: 1px solid #e0e0e0;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .stat-item {
            text-align: center;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }

        .stat-number {
            font-size: 1.5rem;
            font-weight: bold;
            color: #667eea;
            display: block;
        }

        .stat-label {
            font-size: 0.85rem;
            color: #666;
            margin-top: 0.25rem;
        }

        .progress-container {
            margin-bottom: 1rem;
        }

        .progress-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            color: #555;
        }

        .progress-bar {
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        .reference-list {
            background: white;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }

        .no-references {
            text-align: center;
            color: #999;
            padding: 2rem;
            font-style: italic;
        }

        .reference {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            margin-bottom: 1rem;
            overflow: hidden;
            transition: box-shadow 0.2s;
        }

        .reference:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .reference-header {
            padding: 1rem;
            cursor: pointer;
        }

        .reference-title {
            font-weight: 600;
            color: #333;
            margin-bottom: 0.5rem;
            line-height: 1.4;
        }

        .reference-authors {
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }

        .reference-journal {
            color: #888;
            font-size: 0.85rem;
        }

        .llm-status {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .llm-badge {
            padding: 0.2rem 0.5rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 500;
        }

        .llm-badge.processing {
            background: #fff3cd;
            color: #856404;
        }

        .llm-badge.include {
            background: #d4edda;
            color: #155724;
        }

        .llm-badge.exclude {
            background: #f8d7da;
            color: #721c24;
        }

        .llm-badge.conflict {
            background: #ffeaa7;
            color: #6c5ce7;
        }

        .llm-analysis {
            display: none;
            padding: 1rem;
            background: #f8f9fa;
            border-top: 1px solid #e0e0e0;
        }

        .llm-analysis.expanded {
            display: block;
        }

        .provider-hint {
            font-size: 0.8rem;
            color: #6c757d;
            margin-top: 0.5rem;
            padding: 0.5rem;
            background: #f8f9fa;
            border-radius: 4px;
        }

        .sort-group {
            display: flex;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Otto-SR: Production LLM Screening Tool v3.0</h1>
        <p>Advanced systematic review screening with multiple LLM providers and real-time collaboration</p>
    </div>

    <div class="container">
        <div class="left-panel">
            <!-- PICO-TT Criteria Configuration -->
            <div class="criteria-section">
                <h3>📋 PICO-TT Criteria Configuration</h3>
                
                <div class="form-group">
                    <label for="researchQuestion">Research Question</label>
                    <textarea id="researchQuestion" placeholder="What is your primary research question?"></textarea>
                </div>

                <div class="form-group">
                    <label for="population">Population (P)</label>
                    <input type="text" id="population" placeholder="Target population or participants">
                </div>

                <div class="form-group">
                    <label for="intervention">Intervention (I)</label>
                    <input type="text" id="intervention" placeholder="Intervention or exposure">
                </div>

                <div class="form-group">
                    <label for="comparison">Comparison (C)</label>
                    <input type="text" id="comparison" placeholder="Comparison or control group">
                </div>

                <div class="form-group">
                    <label for="outcome">Outcome (O)</label>
                    <input type="text" id="outcome" placeholder="Primary outcomes of interest">
                </div>

                <div class="form-group">
                    <label for="timeframe">Timeframe (T)</label>
                    <input type="text" id="timeframe" placeholder="Time period or follow-up duration">
                </div>

                <div class="form-group">
                    <label for="studyTypes">Study Types (T)</label>
                    <select id="studyTypes">
                        <option value="">All study types</option>
                        <option value="randomized controlled trials">Randomized Controlled Trials</option>
                        <option value="systematic reviews">Systematic Reviews</option>
                        <option value="cohort studies">Cohort Studies</option>
                        <option value="case-control studies">Case-Control Studies</option>
                        <option value="cross-sectional studies">Cross-Sectional Studies</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="inclusionLanguage">Language Requirements</label>
                    <input type="text" id="inclusionLanguage" placeholder="e.g., English, multilingual">
                </div>

                <div class="form-group">
                    <label for="inclusionPublication">Publication Types</label>
                    <input type="text" id="inclusionPublication" placeholder="e.g., peer-reviewed, conference abstracts">
                </div>

                <div class="form-group">
                    <label for="otherInclusion">Other Inclusion Criteria</label>
                    <textarea id="otherInclusion" placeholder="Additional inclusion requirements"></textarea>
                </div>

                <div class="form-group">
                    <label for="otherExclusion">Other Exclusion Criteria</label>
                    <textarea id="otherExclusion" placeholder="Additional exclusion requirements"></textarea>
                </div>
            </div>

            <!-- LLM Configuration -->
            <div class="llm-config">
                <h4>🤖 Dual LLM Configuration</h4>
                
                <!-- AI Model 1 (Conservative) -->
                <div class="config-section">
                    <h5 style="margin-bottom: 0.75rem; color: #d63384;">AI Model 1 - Conservative Screening</h5>
                    <div class="config-row">
                        <label>Provider:</label>
                        <select id="ai1Provider" onchange="updateProviderConfig('ai1')">
                            <option value="openai">OpenAI (Recommended)</option>
                            <option value="anthropic">Anthropic Claude</option>
                            <option value="groq">Groq (Fast)</option>
                            <option value="together">Together AI</option>
                            <option value="ollama">Ollama (Local)</option>
                            <option value="openai_compatible">OpenAI-Compatible</option>
                            <option value="cohere">Cohere</option>
                            <option value="custom">Custom Provider</option>
                        </select>
                    </div>
                    <div class="config-row">
                        <label>Model:</label>
                        <select id="ai1Model">
                            <option value="gpt-4o">gpt-4o</option>
                        </select>
                    </div>
                    <div class="config-row">
                        <label>Endpoint:</label>
                        <input type="text" id="ai1Endpoint" placeholder="API endpoint URL">
                    </div>
                    <div class="config-row" style="margin-bottom: 1rem;">
                        <label>API Key:</label>
                        <input type="password" id="ai1ApiKey" placeholder="Enter API key">
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <button class="test-button" onclick="testConnection('ai1')">Test Connection</button>
                        <span id="ai1Status" class="api-status disconnected">Not Connected</span>
                    </div>
                </div>

                <!-- AI Model 2 (Pragmatic) -->
                <div class="config-section">
                    <h5 style="margin-bottom: 0.75rem; color: #198754;">AI Model 2 - Pragmatic Screening</h5>
                    <div class="config-row">
                        <label>Provider:</label>
                        <select id="ai2Provider" onchange="updateProviderConfig('ai2')">
                            <option value="openai">OpenAI (Recommended)</option>
                            <option value="anthropic">Anthropic Claude</option>
                            <option value="groq">Groq (Fast)</option>
                            <option value="together">Together AI</option>
                            <option value="ollama">Ollama (Local)</option>
                            <option value="openai_compatible">OpenAI-Compatible</option>
                            <option value="cohere">Cohere</option>
                            <option value="custom">Custom Provider</option>
                        </select>
                    </div>
                    <div class="config-row">
                        <label>Model:</label>
                        <select id="ai2Model">
                            <option value="gpt-4o">gpt-4o</option>
                        </select>
                    </div>
                    <div class="config-row">
                        <label>Endpoint:</label>
                        <input type="text" id="ai2Endpoint" placeholder="API endpoint URL">
                    </div>
                    <div class="config-row" style="margin-bottom: 1rem;">
                        <label>API Key:</label>
                        <input type="password" id="ai2ApiKey" placeholder="Enter API key">
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <button class="test-button" onclick="testConnection('ai2')">Test Connection</button>
                        <span id="ai2Status" class="api-status disconnected">Not Connected</span>
                    </div>
                </div>
            </div>

            <!-- File Upload and Controls -->
            <div class="controls">
                <div class="mode-selection">
                    <div class="mode-button active" id="singleModeBtn" onclick="setScreeningMode('single')">Single</div>
                    <div class="mode-button" id="batchModeBtn" onclick="setScreeningMode('batch')">Batch</div>
                    <div class="mode-button" id="aiModeBtn" onclick="setScreeningMode('ai-assisted')">AI-Assisted</div>
                </div>

                <div class="upload-area" id="uploadArea">
                    <div>Drop citation files here or click to browse</div>
                    <small>Supports: RIS, XML, EndNote (.enw), Mendeley (.bib), Zotero (.rdf, .json)</small>
                    <input type="file" id="fileInput" accept=".xml,.ris,.enw,.bib,.rdf,.json" multiple />
                </div>

                <button onclick="startScreening()" id="startBtn" class="ai-button" disabled>Start Screening</button>
                
                <div class="button-row">
                    <button onclick="pauseScreening()" id="pauseBtn" disabled>Pause</button>
                    <button onclick="exportResults()" id="exportBtn" disabled>Export</button>
                </div>
            </div>
        </div>

        </div>

        <div class="main-content">
            <div class="sidebar">
                <div class="stats-grid">
                    <div class="stat-item">
                        <span class="stat-number" id="totalCount">0</span>
                        <div class="stat-label">Total Abstracts</div>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number" id="processedCount">0</span>
                        <div class="stat-label">Processed</div>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number" id="includeCount">0</span>
                        <div class="stat-label">Included</div>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number" id="conflictCount">0</span>
                        <div class="stat-label">Conflicts</div>
                    </div>
                </div>
                
                <div class="progress-container">
                    <div class="progress-label">
                        <span>Processing Progress</span>
                        <span id="progressText">0%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill" style="width: 0%;"></div>
                    </div>
                </div>
            </div>

            <!-- Citations Section -->
            <div id="citationsSection" class="citations-section" style="display: none;">
                <h3>Uploaded Citations</h3>
                <div class="citations-controls">
                    <div class="search-box">
                        <input type="text" id="citationSearch" placeholder="Search citations..." onkeyup="filterCitations()">
                        <button onclick="clearSearch()">Clear</button>
                    </div>
                    <div class="sort-controls">
                        <label>Sort by:</label>
                        <select id="sortCitations" onchange="sortCitations()">
                            <option value="relevance">Relevance Score</option>
                            <option value="title">Title</option>
                            <option value="year">Year</option>
                            <option value="authors">Authors</option>
                            <option value="journal">Journal</option>
                        </select>
                    </div>
                </div>
                <div id="citationsCarousel" class="citations-carousel">
                    <!-- Citations will be dynamically loaded here -->
                </div>
            </div>

            <!-- Performance Metrics Panel -->
            <div id="metricsPanel" class="metrics-panel" style="display: none;">
                <h3>Performance Metrics</h3>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h4>Processing Overview</h4>
                        <div id="processingStats">
                            <div class="stat-item">
                                <span class="stat-label">Total Citations:</span>
                                <span class="stat-value" id="totalCitations">0</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Processed:</span>
                                <span class="stat-value" id="processedCitations">0</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Included:</span>
                                <span class="stat-value" id="includedCitations">0</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Excluded:</span>
                                <span class="stat-value" id="excludedCitations">0</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <h4>AI Performance</h4>
                        <div id="aiPerformanceStats">
                            <div class="stat-item">
                                <span class="stat-label">Avg Response Time:</span>
                                <span class="stat-value" id="avgResponseTime">-</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Avg Confidence:</span>
                                <span class="stat-value" id="avgConfidence">-</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Success Rate:</span>
                                <span class="stat-value" id="successRate">-</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Conflicts:</span>
                                <span class="stat-value" id="conflictCount">0</span>
                            </div>
                        </div>
                    </div>

                    <div class="metric-card">
                        <h4>Export & Storage</h4>
                        <div class="export-controls">
                            <div class="config-row">
                                <label>Export Format:</label>
                                <select id="exportFormat">
                                    <option value="json">JSON (Detailed)</option>
                                    <option value="csv">CSV (Tabular)</option>
                                    <option value="excel">Excel (Multi-sheet)</option>
                                </select>
                            </div>
                            <div class="config-row">
                                <label>Include Metadata:</label>
                                <input type="checkbox" id="includeMetadata" checked>
                            </div>
                            <div class="config-row">
                                <label>Save Processing Log:</label>
                                <input type="checkbox" id="saveProcessingLog" checked>
                            </div>
                            <button onclick="exportWithOptions()" class="export-btn">Export with Options</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="reference-list" id="referenceList">
                <div class="no-references">Configure LLM connections and upload files to begin screening</div>
            </div>
        </div>

        <!-- Citation Detail Modal -->
        <div id="citationModal" class="modal" style="display: none;">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Citation Details & AI Analysis</h3>
                    <span class="close" onclick="closeCitationModal()">&times;</span>
                </div>
                <div class="modal-body" id="modalBody">
                    <!-- Citation details and AI analysis will be populated here -->
                </div>
            </div>
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
        
        // Citation Management Variables
        let uploadedCitations = [];
        let filteredCitations = [];
        let performanceMetrics = {
            totalCitations: 0,
            processedCitations: 0,
            includedCitations: 0,
            excludedCitations: 0,
            avgResponseTime: 0,
            avgConfidence: 0,
            successRate: 0,
            conflictCount: 0
        };

        // --- Event Listeners ---
        document.addEventListener('DOMContentLoaded', () => {
            document.getElementById('fileInput').addEventListener('change', handleFileUpload);
            setupDragAndDrop();
            loadProviders();
            
            // Initialize metrics display
            updateMetricsDisplay();
        });

        // --- Provider Management ---
        async function loadProviders() {
            try {
                const response = await fetch('/api/providers');
                const data = await response.json();
                providerConfigs = data.providers;
                
                // Update provider dropdowns
                updateProviderOptions('ai1');
                updateProviderOptions('ai2');
                
                // Set initial configurations
                updateProviderConfig('ai1');
                updateProviderConfig('ai2');
            } catch (error) {
                console.error('Error loading providers:', error);
            }
        }

        function updateProviderOptions(aiModel) {
            const select = document.getElementById(`${aiModel}Provider`);
            select.innerHTML = '';
            
            Object.entries(providerConfigs).forEach(([key, config]) => {
                const option = document.createElement('option');
                option.value = key;
                option.textContent = config.display_name;
                select.appendChild(option);
            });
        }

        function updateProviderConfig(aiModel) {
            const providerSelect = document.getElementById(`${aiModel}Provider`);
            const modelSelect = document.getElementById(`${aiModel}Model`);
            const endpointInput = document.getElementById(`${aiModel}Endpoint`);
            const apiKeyInput = document.getElementById(`${aiModel}ApiKey`);
            
            const selectedProvider = providerSelect.value;
            const providerConfig = providerConfigs[selectedProvider];
            
            if (!providerConfig) return;
            
            // Update models
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
            
            // Update endpoint
            endpointInput.value = providerConfig.default_endpoint || '';
            
            // Add provider hints
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
            
            // Update button states safely
            document.querySelectorAll('.mode-button').forEach(btn => btn.classList.remove('active'));
            const modeBtn = document.getElementById(mode + 'ModeBtn');
            if (modeBtn) {
                modeBtn.classList.add('active');
            }
            
            updateStartButton();
        }

        // --- Status Message System ---
        function showStatusMessage(message, type = 'info') {
            // Create or update status display
            let statusDiv = document.getElementById('statusMessage');
            if (!statusDiv) {
                statusDiv = document.createElement('div');
                statusDiv.id = 'statusMessage';
                statusDiv.style.cssText = `
                    position: fixed; top: 20px; right: 20px; z-index: 1000;
                    padding: 1rem; border-radius: 6px; max-width: 400px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                `;
                document.body.appendChild(statusDiv);
            }
            
            const colors = {
                'info': 'background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb;',
                'success': 'background: #d4edda; color: #155724; border: 1px solid #c3e6cb;',
                'error': 'background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;'
            };
            
            statusDiv.style.cssText += colors[type] || colors.info;
            statusDiv.textContent = message;
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
                if (statusDiv.parentNode) {
                    statusDiv.parentNode.removeChild(statusDiv);
                }
            }, 5000);
        }

        // --- Activity Log System ---
        function addActivityLog(message, type = 'info') {
            const timestamp = new Date().toLocaleTimeString();
            console.log(`[${timestamp}] ${type.toUpperCase()}: ${message}`);
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
                showStatusMessage('Uploading and parsing files...', 'info');
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error('Upload failed');

                const result = await response.json();
                currentProject = result.project_id;
                
                // Store citations data
                uploadedCitations = result.citations || [];
                filteredCitations = [...uploadedCitations];
                references = uploadedCitations;
                
                // Update metrics
                performanceMetrics.totalCitations = uploadedCitations.length;
                updateMetricsDisplay();
                
                // Display citations immediately
                displayCitations();
                
                showStatusMessage(`Successfully uploaded ${result.citations_count} citations`, 'success');
                
                // Enable screening controls
                updateStartButton();
                
                // Log activity
                addActivityLog(`Uploaded ${result.citations_count} citations`, 'success');
                
            } catch (error) {
                console.error('Upload error:', error);
                showStatusMessage(`Upload failed: ${error.message}`, 'error');
                addActivityLog(`Upload failed: ${error.message}`, 'error');
            }
        }

        function displayCitations() {
            const citationsSection = document.getElementById('citationsSection');
            const citationsCarousel = document.getElementById('citationsCarousel');
            const metricsPanel = document.getElementById('metricsPanel');
            
            if (filteredCitations.length === 0) {
                citationsSection.style.display = 'none';
                metricsPanel.style.display = 'none';
                return;
            }
            
            citationsSection.style.display = 'block';
            metricsPanel.style.display = 'block';
            
            citationsCarousel.innerHTML = '';
            
            filteredCitations.forEach((citation, index) => {
                const card = createCitationCard(citation, index);
                citationsCarousel.appendChild(card);
            });
        }

        function createCitationCard(citation, index) {
            const card = document.createElement('div');
            card.className = 'citation-card';
            card.setAttribute('data-index', index);
            
            // Calculate relevance score display
            const relevanceScore = citation.relevance_score || 0.5;
            const scoreColor = relevanceScore > 0.7 ? '#27ae60' : relevanceScore > 0.4 ? '#f39c12' : '#e74c3c';
            
            card.innerHTML = `
                <div class="citation-title">${citation.title || 'No title available'}</div>
                <div class="citation-meta">
                    <strong>Authors:</strong> ${citation.authors || 'Unknown'}<br>
                    <strong>Journal:</strong> ${citation.journal || 'Unknown'} 
                    ${citation.year ? `(${citation.year})` : ''}
                    ${citation.doi ? `<br><strong>DOI:</strong> ${citation.doi}` : ''}
                </div>
                <div class="citation-abstract">
                    ${citation.abstract ? citation.abstract.substring(0, 200) + (citation.abstract.length > 200 ? '...' : '') : 'No abstract available'}
                </div>
                ${citation.keywords ? `<div class="citation-meta"><strong>Keywords:</strong> ${citation.keywords}</div>` : ''}
                <div class="citation-footer">
                    <span class="relevance-score" style="background-color: ${scoreColor}">
                        Relevance: ${(relevanceScore * 100).toFixed(0)}%
                    </span>
                    <span class="citation-status status-ready">Ready</span>
                    <button class="view-details-btn" onclick="showCitationDetails('${citation.id}')">View Details</button>
                </div>
            `;
            
            return card;
        }

        function filterCitations() {
            const searchTerm = document.getElementById('citationSearch').value.toLowerCase();
            
            if (!searchTerm.trim()) {
                filteredCitations = [...uploadedCitations];
            } else {
                filteredCitations = uploadedCitations.filter(citation => {
                    return (citation.title && citation.title.toLowerCase().includes(searchTerm)) ||
                           (citation.authors && citation.authors.toLowerCase().includes(searchTerm)) ||
                           (citation.journal && citation.journal.toLowerCase().includes(searchTerm)) ||
                           (citation.abstract && citation.abstract.toLowerCase().includes(searchTerm)) ||
                           (citation.keywords && citation.keywords.toLowerCase().includes(searchTerm)) ||
                           (citation.year && citation.year.toString().includes(searchTerm));
                });
            }
            
            displayCitations();
        }

        function clearSearch() {
            document.getElementById('citationSearch').value = '';
            filteredCitations = [...uploadedCitations];
            displayCitations();
        }

        function sortCitations() {
            const sortBy = document.getElementById('sortCitations').value;
            
            filteredCitations.sort((a, b) => {
                switch(sortBy) {
                    case 'relevance':
                        return (b.relevance_score || 0.5) - (a.relevance_score || 0.5);
                    case 'title':
                        return (a.title || '').localeCompare(b.title || '');
                    case 'year':
                        return (b.year || 0) - (a.year || 0);
                    case 'authors':
                        return (a.authors || '').localeCompare(b.authors || '');
                    case 'journal':
                        return (a.journal || '').localeCompare(b.journal || '');
                    default:
                        return 0;
                }
            });
            
            displayCitations();
        }

        function updateMetricsDisplay() {
            document.getElementById('totalCitations').textContent = performanceMetrics.totalCitations;
            document.getElementById('processedCitations').textContent = performanceMetrics.processedCitations;
            document.getElementById('includedCitations').textContent = performanceMetrics.includedCitations;
            document.getElementById('excludedCitations').textContent = performanceMetrics.excludedCitations;
            document.getElementById('avgResponseTime').textContent = 
                performanceMetrics.avgResponseTime > 0 ? `${performanceMetrics.avgResponseTime.toFixed(2)}s` : '-';
            document.getElementById('avgConfidence').textContent = 
                performanceMetrics.avgConfidence > 0 ? `${(performanceMetrics.avgConfidence * 100).toFixed(1)}%` : '-';
            document.getElementById('successRate').textContent = 
                performanceMetrics.successRate > 0 ? `${(performanceMetrics.successRate * 100).toFixed(1)}%` : '-';
            document.getElementById('conflictCount').textContent = performanceMetrics.conflictCount;
        }

        function exportWithOptions() {
            const format = document.getElementById('exportFormat').value;
            const includeMetadata = document.getElementById('includeMetadata').checked;
            const saveProcessingLog = document.getElementById('saveProcessingLog').checked;
            
            if (!currentProject) {
                showStatusMessage('No project to export', 'error');
                return;
            }
            
            // Build export URL with options
            const params = new URLSearchParams({
                format: format,
                include_metadata: includeMetadata,
                include_log: saveProcessingLog
            });
            
            const exportUrl = `/api/projects/${currentProject}/export?${params}`;
            
            // Trigger download
            const link = document.createElement('a');
            link.href = exportUrl;
            link.download = `otto-sr-results-${new Date().toISOString().split('T')[0]}.${format === 'excel' ? 'xlsx' : format}`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            showStatusMessage(`Exporting results in ${format.toUpperCase()} format...`, 'info');
            addActivityLog(`Exported results (${format})`, 'info');
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
                    body: JSON.stringify({ config })
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
            
            document.getElementById('progressFill').style.width = `${percentage}%`;
            document.getElementById('progressText').textContent = `${percentage}%`;
        }

        function updateReferenceList() {
            const listElement = document.getElementById('referenceList');
            
            if (!references.length) {
                listElement.innerHTML = '<div class="no-references">Configure LLM connections and upload files to begin screening</div>';
                return;
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
        }

        function renderAnalysisResults(ref) {
            let html = '<h4>AI Analysis Results</h4>';
            
            if (ref.ai1_result) {
                html += `
                    <div style="margin-bottom: 1rem;">
                        <strong>Conservative AI:</strong> ${ref.ai1_result.decision} 
                        (${ref.ai1_result.confidence}% confidence)
                        <br><em>${ref.ai1_result.reasoning}</em>
                    </div>
                `;
            }
            
            if (ref.ai2_result) {
                html += `
                    <div style="margin-bottom: 1rem;">
                        <strong>Pragmatic AI:</strong> ${ref.ai2_result.decision} 
                        (${ref.ai2_result.confidence}% confidence)
                        <br><em>${ref.ai2_result.reasoning}</em>
                    </div>
                `;
            }
            
            return html;
        }

        function toggleAnalysis(refId) {
            const analysisDiv = document.getElementById(`analysis-${refId}`);
            if (analysisDiv) {
                analysisDiv.classList.toggle('expanded');
            }
        }

        function updateReferenceStatus(data) {
            const refIndex = references.findIndex(r => r.id === data.citation_id);
            if (refIndex !== -1) {
                references[refIndex].status = data.status;
                if (data.ai1_result) references[refIndex].ai1_result = data.ai1_result;
                if (data.ai2_result) references[refIndex].ai2_result = data.ai2_result;
                if (data.final_decision) references[refIndex].final_decision = data.final_decision;
            }
        }

        function updateStartButton() {
            const startBtn = document.getElementById('startBtn');
            const pauseBtn = document.getElementById('pauseBtn');
            const exportBtn = document.getElementById('exportBtn');
            
            const ai1Connected = llmConfigs.ai1.connected;
            const ai2Connected = llmConfigs.ai2.connected;
            const hasReferences = references.length > 0;
            
            if (isProcessing) {
                startBtn.disabled = true;
                startBtn.textContent = 'Processing...';
                pauseBtn.disabled = false;
            } else if (!ai1Connected || !ai2Connected) {
                startBtn.disabled = true;
                startBtn.textContent = 'Connect LLMs First';
                pauseBtn.disabled = true;
            } else if (!hasReferences) {
                startBtn.disabled = true;
                startBtn.textContent = 'Upload Studies First';
                pauseBtn.disabled = true;
            } else {
                startBtn.disabled = false;
                startBtn.textContent = `Start ${screeningMode} Screening`;
                pauseBtn.disabled = true;
            }
            
            exportBtn.disabled = !hasReferences;
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

        // --- Citation Modal Functions ---
        async function showCitationDetails(citationId) {
            const citation = uploadedCitations.find(c => c.id === citationId);
            if (!citation) {
                showStatusMessage('Citation not found', 'error');
                return;
            }

            const modalBody = document.getElementById('modalBody');
            
            // Build citation detail HTML
            let html = `
                <div class="citation-detail">
                    <h4>📄 Citation Information</h4>
                    <div class="citation-meta-detail">
                        <p><strong>Title:</strong> ${citation.title || 'No title available'}</p>
                        <p><strong>Authors:</strong> ${citation.authors || 'Unknown'}</p>
                        <p><strong>Journal:</strong> ${citation.journal || 'Unknown'}</p>
                        <p><strong>Year:</strong> ${citation.year || 'Unknown'}</p>
                        ${citation.doi ? `<p><strong>DOI:</strong> ${citation.doi}</p>` : ''}
                        ${citation.keywords ? `<p><strong>Keywords:</strong> ${citation.keywords}</p>` : ''}
                        <p><strong>Relevance Score:</strong> ${((citation.relevance_score || 0.5) * 100).toFixed(0)}%</p>
                    </div>
                    ${citation.abstract ? `
                        <h4>📝 Abstract</h4>
                        <div class="citation-meta-detail">
                            <p>${citation.abstract}</p>
                        </div>
                    ` : ''}
                </div>
            `;

            // Check if this citation has been screened (has AI results)
            if (currentProject) {
                try {
                    const response = await fetch(`/api/projects/${currentProject}/citations/${citationId}/results`);
                    if (response.ok) {
                        const result = await response.json();
                        if (result.ai1_result || result.ai2_result) {
                            html += generateAIAnalysisHTML(result);
                        } else {
                            html += `
                                <div class="ai-analysis">
                                    <div class="no-analysis">
                                        <p><em>This citation has not been screened yet. Start the screening process to see AI analysis.</em></p>
                                    </div>
                                </div>
                            `;
                        }
                    }
                } catch (error) {
                    console.error('Error fetching screening results:', error);
                }
            }

            modalBody.innerHTML = html;
            document.getElementById('citationModal').style.display = 'flex';
        }

        function generateAIAnalysisHTML(result) {
            let html = '<div class="ai-analysis"><h4>🤖 AI Screening Analysis</h4>';
            
            if (result.ai1_result) {
                html += generateSingleAIResultHTML('AI Model 1 (Conservative)', result.ai1_result);
            }
            
            if (result.ai2_result) {
                html += generateSingleAIResultHTML('AI Model 2 (Pragmatic)', result.ai2_result);
            }

            if (result.final_decision) {
                const decisionClass = result.final_decision === 'include' ? 'decision-include' : 
                                    result.final_decision === 'exclude' ? 'decision-exclude' : 'decision-uncertain';
                
                html += `
                    <div class="final-decision" style="grid-column: 1 / -1; margin-top: 1.5rem; padding: 1rem; border: 2px solid #007bff; border-radius: 6px; background: #f8f9fa;">
                        <h5>🎯 Final Decision</h5>
                        <div class="decision-badge ${decisionClass}">${result.final_decision}</div>
                        ${result.confidence_score ? `<p><strong>Confidence:</strong> ${(result.confidence_score * 100).toFixed(1)}%</p>` : ''}
                        ${result.notes ? `<div class="reasoning-text">${result.notes}</div>` : ''}
                    </div>
                `;
            }
            
            html += '</div>';
            return html;
        }

        function generateSingleAIResultHTML(title, aiResult) {
            if (!aiResult || typeof aiResult !== 'object') return '';
            
            const decision = aiResult.decision || 'uncertain';
            const confidence = aiResult.confidence || 0;
            const reasoning = aiResult.reasoning || 'No reasoning provided';
            const evidenceQuotes = aiResult.evidence_quotes || [];
            const picoScores = aiResult.pico_scores || {};
            
            const decisionClass = decision === 'include' ? 'decision-include' : 
                                decision === 'exclude' ? 'decision-exclude' : 'decision-uncertain';
            
            return `
                <div class="ai-result">
                    <h5>${title}</h5>
                    <div class="decision-badge ${decisionClass}">${decision}</div>
                    <div class="confidence-score">
                        <strong>Confidence:</strong> ${confidence.toFixed(1)}%
                    </div>
                    
                    <div class="reasoning-text">
                        <strong>Reasoning:</strong><br>
                        ${reasoning}
                    </div>
                    
                    ${evidenceQuotes.length > 0 ? `
                        <div class="evidence-quotes">
                            <strong>Evidence Quotes:</strong>
                            <ul>
                                ${evidenceQuotes.map(quote => `<li>"${quote}"</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    
                    ${Object.keys(picoScores).length > 0 ? `
                        <div class="pico-scores">
                            ${Object.entries(picoScores).map(([key, value]) => `
                                <div class="pico-score">
                                    <div class="pico-score-value">${(value * 100).toFixed(0)}%</div>
                                    <div class="pico-score-label">${key}</div>
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
            `;
        }

        function closeCitationModal() {
            document.getElementById('citationModal').style.display = 'none';
        }

        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('citationModal');
            if (event.target === modal) {
                closeCitationModal();
            }
        }
    </script>
</body>
</html>
    """)

@app.get("/api/providers")
async def get_providers():
    """Get available LLM providers and their configurations"""
    providers = {
        "openai": ProviderConfig(
            name="openai",
            display_name="OpenAI (Recommended)",
            models=["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            default_model="gpt-4o",
            requires_api_key=True,
            default_endpoint=None,
            supports_streaming=True
        ),
        "anthropic": ProviderConfig(
            name="anthropic",
            display_name="Anthropic Claude (Recommended)",
            models=["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
            default_model="claude-3-5-sonnet-20241022",
            requires_api_key=True,
            default_endpoint=None,
            supports_streaming=True
        ),
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
        "cohere": ProviderConfig(
            name="cohere",
            display_name="Cohere",
            models=["command-r-plus", "command-r", "command-nightly"],
            default_model="command-r-plus",
            requires_api_key=True,
            default_endpoint=None,
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
    
    return {"providers": {k: v.dict() for k, v in providers.items()}}

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    """Enhanced file upload with immediate citation display"""
    if not files:
        raise HTTPException(400, "No files provided")
    
    # Create new project
    project = Project(name=f"Project {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    db.add(project)
    db.commit()
    db.refresh(project)
    
    all_citations = []
    
    for file in files:
        try:
            content = await file.read()
            content_str = content.decode('utf-8', errors='ignore')
            
            # Parse based on file extension
            filename = file.filename.lower()
            if filename.endswith('.ris'):
                citations = parse_ris_file(content_str)
            elif filename.endswith('.xml'):
                citations = parse_xml_file(content_str)
            elif filename.endswith('.enw'):
                citations = parse_endnote_file(content_str)
            elif filename.endswith('.bib'):
                citations = parse_mendeley_file(content_str)
            elif filename.endswith(('.rdf', '.json')):
                citations = parse_zotero_file(content_str)
            else:
                # Try auto-detection
                if content_str.strip().startswith('TY  -'):
                    citations = parse_ris_file(content_str)
                elif content_str.strip().startswith('<'):
                    citations = parse_xml_file(content_str)
                else:
                    citations = parse_ris_file(content_str)  # Fallback
            
            # Store citations in database
            for citation_data in citations:
                citation = CitationRecord(
                    project_id=project.id,
                    title=citation_data.get('title', ''),
                    authors=citation_data.get('authors', ''),
                    journal=citation_data.get('journal', ''),
                    year=citation_data.get('year'),
                    abstract=citation_data.get('abstract', ''),
                    doi=citation_data.get('doi', ''),
                    keywords=citation_data.get('keywords', ''),
                    relevance_score=citation_data.get('relevance_score', 0.5),
                    file_content=content_str[:1000]  # Store first 1000 chars for reference
                )
                db.add(citation)
                all_citations.append(citation_data)
        
        except Exception as e:
            db.rollback()
            raise HTTPException(400, f"Error processing file {file.filename}: {str(e)}")
    
    db.commit()
    
    return CitationUploadResponse(
        project_id=str(project.id),
        citations_count=len(all_citations),
        message=f"Successfully uploaded {len(all_citations)} citations",
        citations=all_citations,
        status="success"
    )

@app.get("/api/projects/{project_id}/citations")
async def get_citations(project_id: str, db: Session = Depends(get_db)):
    """Get citations for a project"""
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(400, "Invalid project ID format")
    
    citations = db.query(CitationRecord).filter(CitationRecord.project_id == project_uuid).all()
    
    citation_data = []
    for citation in citations:
        citation_data.append({
            'id': str(citation.id),
            'title': citation.title,
            'authors': citation.authors,
            'journal': citation.journal,
            'year': citation.year,
            'abstract': citation.abstract,
            'doi': citation.doi,
            'keywords': citation.keywords,
            'relevance_score': float(citation.relevance_score) if citation.relevance_score else 0.5
        })
    
    return citation_data

@app.post("/api/test-llm")
async def test_llm_connection(request: dict):
    """Test LLM connection"""
    config = request.get('config', {})
    
    try:
        # Create a simple test using the LLM provider factory directly
        llm_config = LLMConfig(**config)
        llm = LLMProviderFactory.create_llm(llm_config)
        
        # Simple test message
        test_message = "Respond with exactly: {'status': 'success', 'message': 'Connection working'}"
        
        # Test the connection
        response = await llm.ainvoke(test_message)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        return {"success": True, "response": response_text}
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
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(400, "Invalid project ID format")
    
    # Validate project exists
    project = db.query(Project).filter(Project.id == project_uuid).first()
    if not project:
        raise HTTPException(404, "Project not found")
    
    # Store criteria in project
    criteria = request.get('criteria', {})
    project.criteria = criteria
    db.commit()
    
    # Get citations
    citations = db.query(CitationRecord).filter(CitationRecord.project_id == project_uuid).all()
    if not citations:
        raise HTTPException(400, "No citations found for this project")
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Create screening results for each citation
    llm_configs = request.get('llm_configs', {})
    
    for citation in citations:
        # Create screening result record
        result = ScreeningResult(
            citation_id=citation.id,
            project_id=project_uuid,
            job_id=job_id,
            status="pending"
        )
        db.add(result)
        db.commit()
        db.refresh(result)
        
        # Add to background tasks
        background_tasks.add_task(
            advanced_screening_task,
            str(result.id),
            job_id,
            llm_configs
        )
    
    return {"job_id": job_id, "message": f"Started screening {len(citations)} citations"}

@app.get("/api/stream/{job_id}")
async def stream_progress(job_id: str):
    """Stream screening progress"""
    async def event_generator():
        processed_ids = set()
        
        while True:
            try:
                db_session = SessionLocal()
                
                # Get unprocessed results
                results = db_session.query(ScreeningResult).filter(
                    ScreeningResult.job_id == job_id,
                    ScreeningResult.status.in_(["completed", "error"])
                ).all()
                
                # Send progress updates for new completions
                for result in results:
                    if result.id not in processed_ids:
                        citation = db_session.query(CitationRecord).filter(CitationRecord.id == result.citation_id).first()
                        
                        progress_data = {
                            "citation_id": str(result.citation_id),
                            "status": result.status,
                            "ai1_result": result.ai1_result,
                            "ai2_result": result.ai2_result,
                            "final_decision": result.final_decision,
                            "confidence_score": float(result.confidence_score) if result.confidence_score else 0.0,
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

@app.get("/api/projects/{project_id}/citations/{citation_id}/results")
async def get_citation_results(project_id: str, citation_id: str, db: Session = Depends(get_db)):
    """Get screening results for a specific citation"""
    try:
        # Find the screening result for this citation
        result = db.query(ScreeningResult).filter(
            ScreeningResult.project_id == project_id,
            ScreeningResult.citation_id == citation_id
        ).first()
        
        if not result:
            return {"message": "No screening results found for this citation"}
        
        return {
            "citation_id": result.citation_id,
            "project_id": result.project_id,
            "status": result.status,
            "ai1_result": result.ai1_result,
            "ai2_result": result.ai2_result,
            "final_decision": result.final_decision,
            "confidence_score": result.confidence_score,
            "notes": result.notes,
            "processed_at": result.processed_at
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching citation results: {str(e)}")

@app.get("/api/projects/{project_id}/export")
async def export_results(project_id: str, db: Session = Depends(get_db)):
    """Export screening results"""
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(400, "Invalid project ID format")
    
    project = db.query(Project).filter(Project.id == project_uuid).first()
    if not project:
        raise HTTPException(404, "Project not found")
    
    citations = db.query(CitationRecord).filter(CitationRecord.project_id == project_uuid).all()
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
                "confidence_score": float(screening_result.confidence_score) if screening_result and screening_result.confidence_score else None,
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