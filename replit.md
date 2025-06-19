# Otto-SR: Complete Single-File Application

## Overview

Otto-SR is a comprehensive web application for systematic review screening built as a single-file FastAPI application. The application combines a Python backend with an embedded HTML/CSS/JavaScript frontend to provide a complete solution for academic research screening workflows.

The application uses a monolithic single-file architecture where both the server-side logic and client-side interface are contained within `main.py`, making it easy to deploy and maintain.

## System Architecture

### Architecture Pattern
- **Monolithic Single-File**: All application logic, including both backend API and frontend UI, is contained in a single Python file
- **Full-Stack Web Application**: FastAPI backend with embedded HTML/CSS/JavaScript frontend
- **Database-Driven**: PostgreSQL database for persistent data storage
- **AI-Enhanced**: Integration with OpenAI's language models for intelligent screening assistance

### Technology Stack
- **Backend**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL with SQLAlchemy ORM
- **AI Integration**: LangChain with OpenAI GPT models
- **Frontend**: Vanilla HTML/CSS/JavaScript (embedded)
- **File Handling**: aiofiles for async file operations

## Key Components

### Backend Services
1. **FastAPI Application**: Main web server handling HTTP requests
2. **SQLAlchemy Models**: Database schema definitions for systematic review data
3. **LangChain Integration**: AI-powered screening and analysis capabilities
4. **File Upload System**: Support for research paper uploads and processing
5. **Background Tasks**: Async processing for long-running operations

### Database Layer
- **SQLAlchemy ORM**: Object-relational mapping for database operations  
- **PostgreSQL**: Primary data store for application data
- **Connection Management**: Session-based database connections with proper cleanup

### AI Integration
- **OpenAI GPT Models**: Language model integration for intelligent screening
- **LangChain Framework**: Structured prompting and response handling
- **Async Processing**: Non-blocking AI operations for better user experience

### Frontend Interface
- **Embedded HTML**: Complete user interface served directly from the Python application
- **JavaScript**: Client-side interactivity and API communication
- **Responsive Design**: CSS styling for cross-device compatibility

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