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

## Changelog
- June 19, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.