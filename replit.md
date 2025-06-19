# Otto-SR: Production LLM Screening Tool v3.0

## Overview

Otto-SR is a comprehensive systematic review screening application built as a single-file FastAPI application with advanced multi-provider LLM support. The application combines a Python backend with an embedded HTML/CSS/JavaScript frontend to provide a complete solution for academic research screening workflows with sophisticated AI assistance.

The application uses a monolithic single-file architecture where both the server-side logic and client-side interface are contained within `main.py`, making it easy to deploy and maintain while supporting multiple LLM providers through LangChain integration.

## System Architecture

### Architecture Pattern
- **Monolithic Single-File**: All application logic, including both backend API and frontend UI, is contained in a single Python file
- **Full-Stack Web Application**: FastAPI backend with embedded HTML/CSS/JavaScript frontend
- **Database-Driven**: PostgreSQL database for persistent data storage
- **Multi-Provider AI-Enhanced**: Integration with multiple LLM providers through LangChain for intelligent screening assistance
- **Structured Output Processing**: Pydantic-based structured outputs for consistent AI responses

### Technology Stack
- **Backend**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL with SQLAlchemy ORM
- **AI Integration**: LangChain with multiple LLM providers (OpenAI, Anthropic, Ollama, LM Studio, Cohere, Hugging Face)
- **Structured Outputs**: Pydantic models for consistent AI response formatting
- **Frontend**: Vanilla HTML/CSS/JavaScript (embedded)
- **File Handling**: aiofiles for async file operations

## Key Components

### Backend Services
1. **FastAPI Application**: Main web server handling HTTP requests
2. **SQLAlchemy Models**: Database schema definitions for systematic review data
3. **Multi-Provider LLM Integration**: AI-powered screening with support for multiple providers
4. **File Upload System**: Enhanced parsing for RIS and XML citation formats
5. **Background Tasks**: Async processing for long-running screening operations
6. **Structured Output Processing**: Pydantic-based AI response validation

### Database Layer
- **SQLAlchemy ORM**: Object-relational mapping for database operations  
- **PostgreSQL**: Primary data store for application data
- **Connection Management**: Session-based database connections with proper cleanup
- **Activity Logging**: Comprehensive tracking of screening activities

### Multi-Provider AI Integration
- **LangChain Framework**: Unified interface for multiple LLM providers
- **Supported Providers**: OpenAI, Anthropic (Claude), Ollama, LM Studio, Cohere, Hugging Face
- **Dynamic Provider Factory**: Automatic LLM instance creation based on provider configuration
- **Structured Prompts**: Template-based prompting with PICO-TT criteria integration
- **Pydantic Output Parsing**: Consistent structured responses across all providers
- **Dual Screening Strategy**: Conservative vs pragmatic AI approaches for conflict detection

### Advanced Screening Features
- **PICO-TT Criteria Configuration**: Comprehensive systematic review criteria setup
- **Multiple Screening Modes**: Single citation, batch processing, and AI-assisted workflows
- **Real-time Progress Tracking**: Live updates during screening processes
- **Conflict Detection**: Automatic identification of AI disagreements for human review
- **Relevance Scoring**: Automated assessment of citation completeness and relevance

### Frontend Interface
- **Embedded HTML**: Complete user interface served directly from the Python application
- **Dynamic Provider Selection**: JavaScript-driven LLM provider configuration
- **Real-time Updates**: Progress tracking and status monitoring
- **Responsive Design**: CSS styling for cross-device compatibility
- **File Drag-and-Drop**: Enhanced file upload experience

## Data Flow

1. **Request Handling**: FastAPI receives HTTP requests from the frontend
2. **Database Operations**: SQLAlchemy manages data persistence and retrieval
3. **AI Processing**: LangChain orchestrates interactions with OpenAI models
4. **Response Generation**: Processed results are returned to the frontend
5. **UI Updates**: JavaScript updates the interface with new data

## External Dependencies

### Required Environment Variables
- `DATABASE_URL`: PostgreSQL connection string
- `OPENAI_API_KEY`: OpenAI API key for language model access

### Python Dependencies
- `fastapi`: Web framework for API development
- `uvicorn`: ASGI server for running the application
- `sqlalchemy`: Database ORM and connection management
- `psycopg2-binary`: PostgreSQL adapter for Python
- `langchain-core` & `langchain-openai`: AI integration framework
- `aiofiles`: Async file handling
- `python-multipart`: File upload support
- `pydantic`: Data validation and serialization

## Deployment Strategy

### Development Environment
- **Replit Configuration**: Configured for Replit deployment with Python 3.11
- **Auto-Installation**: Dependencies automatically installed via pip
- **Port Configuration**: Application runs on port 5000 with host binding

### Production Considerations
- **Environment Variables**: Database and API credentials must be configured
- **Database Setup**: PostgreSQL database must be provisioned and accessible
- **Scaling**: Single-file architecture may require refactoring for high-load scenarios

### Deployment Commands
```bash
pip install fastapi uvicorn sqlalchemy psycopg2-binary langchain-core langchain-openai aiofiles python-multipart pydantic
uvicorn main:app --host 0.0.0.0 --port 5000
```

## Recent Changes

### June 19, 2025 - Enhanced Screening Decision Structure
- **Improved Screening Output Schema**: Enhanced the ScreeningDecision model with better structured outputs
  - Added evidence_quotes field for specific citation quotes supporting decisions
  - Enhanced criteria_assessment with detailed per-criterion evaluation (meets/does_not_meet/unclear)
  - Added quality_indicators for structured quality assessment (sample_size_adequate, methodology_clear, outcomes_relevant)
  - Included processing_metadata field for timing and model information tracking
  - Added "uncertain" decision option alongside include/exclude for cases requiring human review
- **Quality Assessment Integration**: Structured quality indicators provide standardized assessment framework
  - Sample size adequacy evaluation based on available information
  - Methodology clarity assessment for study design evaluation
  - Outcome relevance scoring aligned with systematic review objectives
- **Evidence-Based Decision Making**: Enhanced reasoning with specific quote extraction
  - Direct quotes from citations supporting inclusion/exclusion decisions
  - Improved traceability of AI reasoning back to source material
  - Better support for human reviewers to validate AI decisions

### June 19, 2025 - Additional Citation Format Support
- **Enhanced Citation Format Support**: Added comprehensive support for major reference management systems
  - EndNote format support (.enw tagged format and XML)
  - Mendeley format support (.bib BibTeX format) with proper author and keyword parsing
  - Zotero format support (.rdf RDF format and .json CSL JSON format)
  - Enhanced XML parser with namespace support for multiple citation standards
  - Automatic format detection and appropriate parser selection based on file extension
- **Robust Citation Parsing**: Advanced field extraction across all supported formats
  - Standardized author name handling and formatting across different citation styles
  - Enhanced keyword extraction and DOI validation for all formats
  - Improved year extraction with multiple date field support and validation
  - Consistent relevance scoring algorithm applied across all citation formats
- **User Interface Updates**: Frontend updated to reflect expanded format compatibility
  - Upload area now shows all supported formats with clear file extension guidance
  - File input accepts .ris, .xml, .enw, .bib, .rdf, .json formats
  - Enhanced error handling with format-specific error messages for better user guidance

### June 19, 2025 - Citation Carousel Display Implementation
- **Immediate Citation Display**: Enhanced file upload with citation carousel showing uploaded citations instantly
  - Citations display immediately after file upload without requiring page refresh
  - Horizontal scrollable carousel with properly formatted citation cards
  - Each card shows title, authors, journal, year, truncated abstract, relevance score, and status
  - Hover effects and smooth scrolling with navigation controls for large citation sets
  - Database schema updated to support ai1_result and ai2_result columns for dual LLM workflow
  - Upload response enhanced to include citation data for immediate frontend display
- **User Experience Enhancement**: Eliminates waiting time between upload and citation visibility
  - Users can immediately review their uploaded citations in a clean, organized format
  - Cards show essential citation information at a glance with visual relevance scoring
  - Ready status indicators show citations are prepared for screening workflow

### June 19, 2025 - Advanced Dual LLM Judgment System Integration
- **Intelligent Conflict Resolution**: Sophisticated agreement scoring algorithm
  - Multi-factor analysis: decision agreement, confidence alignment, PICO scores, quality assessment
  - Three-tier resolution: consensus (80%+ agreement), higher confidence (60-80%), conflict detection (<60%)
  - Automated resolution strategies with detailed reasoning and methodology tracking
- **Performance Analytics System**: Comprehensive LLM optimization framework
  - Real-time tracking: response times, success rates, confidence scores, token usage
  - Provider comparison metrics with cost analysis and quality benchmarking
  - Automated recommendations for model selection and performance optimization
  - Session-based analytics with historical performance data
- **Enhanced Screening Intelligence**: Advanced dual AI evaluation with metadata
  - Performance tracking for each LLM response with timing and token metrics
  - Agreement score calculation using sophisticated multi-dimensional analysis
  - Conflict metadata storage including resolution method and reasoning
  - Quality-based model selection with specialization tracking

### June 19, 2025 - Advanced Search & Filtering + RIS Parser Enhancement
- **Advanced Search Interface**: Comprehensive search system with multiple criteria
  - Search by title, authors, journal, publication year, keywords, and abstract content
  - Year range search supporting formats like "2020-2024" or single years
  - Real-time search highlighting of matching terms in results
  - Filter chips showing active search criteria with easy removal
  - Sorting controls for relevance, title, authors, year, journal, decision, and confidence
  - Search results info panel with match statistics and guidance
- **Robust RIS Parser**: Integrated enhanced citation management system
  - Comprehensive field extraction supporting all standard RIS fields
  - Improved author name cleaning and formatting
  - Better handling of multiple values and field continuations
  - Enhanced year extraction from multiple date fields with validation
  - DOI format validation and keyword processing
  - Relevance scoring based on content completeness
- **Database Schema Fix**: Added missing screening_mode and relevance_score columns to resolve upload errors

### June 19, 2025 - Enhanced Multi-Provider LLM Support
- **Added Trusted Provider Tiers**: Organized 8 LLM providers into production-ready tiers
  - Tier 1 (Recommended): OpenAI GPT-4o, Anthropic Claude 3.5 Sonnet
  - Tier 2 (Local/Self-Hosted): Ollama, OpenAI-Compatible endpoints
  - Tier 3 (Fast Inference): Groq, Together AI, Cohere
  - Advanced: Custom provider configuration
- **Dynamic Provider Factory**: LangChain-based unified interface for all providers
- **Structured Output Processing**: Pydantic schemas ensure consistent AI responses
- **Enhanced Frontend**: Provider selection with helpful hints and automatic configuration
- **Provider-Agnostic Design**: Custom endpoint support for any OpenAI-compatible API

### Core Features Implemented
- **PICO-TT Criteria Configuration**: Comprehensive systematic review setup
- **Dual AI Screening**: Conservative vs pragmatic approaches with conflict detection
- **Abstract Navigation Interface**: Interactive component for reviewing screening results
- **Comprehensive Metrics Panel**: Real-time statistics and progress visualization
- **Advanced Filtering System**: Filter abstracts by inclusion/exclusion status
- **Detailed AI Evaluation Display**: Side-by-side comparison of AI reasoning
- **Real-time Progress Tracking**: Live updates during screening processes
- **Enhanced File Parsing**: Support for RIS and XML citation formats
- **Activity Logging**: Complete audit trail of screening decisions
- **Export Functionality**: JSON export of screening results with metadata

## LLM Provider Support

### Trusted Providers (Tested & Recommended)
1. **OpenAI** - Most reliable for production screening
2. **Anthropic Claude** - Excellent reasoning for complex decisions
3. **Groq** - Ultra-fast inference for high-volume processing
4. **Together AI** - Cost-effective with good quality
5. **Ollama** - Local deployment for privacy-sensitive research
6. **OpenAI-Compatible** - Works with LM Studio, Oobabooga, and other local servers
7. **Cohere** - Strong multilingual capabilities
8. **Custom Provider** - Fully configurable for any OpenAI-compatible endpoint

### Technical Implementation
- **LangChain Integration**: Unified API across all providers
- **Pydantic Output Parsing**: Structured responses with validation
- **Dynamic Model Selection**: Provider-specific model lists and defaults
- **Automatic Endpoint Configuration**: Smart defaults with custom override options
- **Secure API Key Management**: Environment variable support with secure handling

## Changelog
- June 19, 2025. Initial setup and multi-provider LLM enhancement

## User Preferences

Preferred communication style: Simple, everyday language.
Provider preference: Support for multiple LLM providers with provider-agnostic options.