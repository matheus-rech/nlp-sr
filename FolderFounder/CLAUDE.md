# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Otto-SR: Production LLM Screening Tool v3.0** is a comprehensive systematic review screening application with multi-provider LLM support. The entire application (backend + frontend) is contained in a single `main.py` file, following a monolithic architecture pattern.

## Key Architecture Points

- **Single-File Architecture**: All server and client code in `main.py` (147KB)
- **FastAPI Backend**: RESTful API with async support
- **PostgreSQL Database**: Using SQLAlchemy ORM for data persistence
- **Multi-Provider LLM Integration**: Support for 8+ LLM providers via LangChain
- **Embedded Frontend**: HTML/CSS/JavaScript served directly from Python

## Common Commands

### Running the Application
```bash
# Install dependencies and run server (single command)
pip install fastapi uvicorn sqlalchemy psycopg2-binary langchain-core langchain-openai aiofiles python-multipart pydantic anthropic && uvicorn main:app --host 0.0.0.0 --port 5000

# Or install dependencies separately
pip install -r requirements.txt  # If requirements.txt exists
# Or use pyproject.toml dependencies
pip install aiofiles anthropic fastapi langchain-core langchain-openai openai psycopg2-binary pydantic python-multipart sqlalchemy uvicorn

# Run the server
uvicorn main:app --host 0.0.0.0 --port 5000

# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

### Database Setup
```bash
# Application expects PostgreSQL database
# Set DATABASE_URL environment variable:
export DATABASE_URL="postgresql://user:password@localhost/ottosr"

# Tables are auto-created on first run via SQLAlchemy
```

### Environment Variables
```bash
# Required for LLM functionality
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"  # For Claude support
# Add other provider keys as needed
```

## Code Structure

Since this is a single-file application, the code organization within `main.py` follows this pattern:

1. **Imports & Configuration** (lines 1-50)
2. **Database Models** - SQLAlchemy table definitions
3. **Pydantic Models** - Request/response schemas
4. **LLM Integration** - Multi-provider factory and structured output parsing
5. **API Endpoints** - FastAPI routes for all functionality
6. **Frontend HTML** - Complete embedded UI served via root endpoint

## Key Technical Details

### Database Schema
- **systematic_reviews**: Main review configurations with PICO-TT criteria
- **citations**: Uploaded citations with screening results
- **activity_logs**: Audit trail of all screening activities

### LLM Provider Support
The application uses a factory pattern for LLM creation:
- Providers: OpenAI, Anthropic, Ollama, LM Studio, Groq, Together AI, Cohere, Custom
- All providers use LangChain for unified interface
- Structured outputs via Pydantic models ensure consistent responses

### Citation Format Support
- RIS (.ris) - Research Information Systems format
- XML (.xml) - Various citation XML formats with namespace handling
- EndNote (.enw) - Tagged format
- Mendeley (.bib) - BibTeX format
- Zotero (.rdf, .json) - RDF and CSL JSON formats

### API Endpoints Pattern
- `POST /api/reviews` - Create systematic review
- `POST /api/citations/upload` - Upload citation files
- `POST /api/screening/batch` - Batch AI screening
- `GET /api/citations/{citation_id}/screening` - Get individual screening results
- `GET /api/export` - Export results in various formats

## Development Tips

### Making Changes
1. **Frontend Changes**: Look for the HTML response in the root endpoint (`@app.get("/")`)
2. **API Changes**: All endpoints are defined as `@app.post()` or `@app.get()` decorators
3. **Database Changes**: Modify SQLAlchemy models and restart to auto-migrate
4. **LLM Changes**: Update the `create_llm()` factory function for provider modifications

### Adding New Features
1. **New Citation Format**: Add parser function following existing patterns (see `parse_ris_file`, `parse_xml_file`)
2. **New LLM Provider**: Add case in `create_llm()` factory with appropriate LangChain class
3. **New API Endpoint**: Add FastAPI route decorator and handler function
4. **Frontend Features**: Modify embedded HTML/JavaScript in root endpoint response

### Testing
Currently no automated tests are implemented. To test:
1. Use the provided `test_citations.ris` file for upload testing
2. Test LLM connections via the UI's "Test Connection" button
3. Monitor console output for debugging information

## Important Patterns

### Structured LLM Outputs
All LLM responses use Pydantic models for consistency:
```python
class ScreeningDecision(PydanticBaseModel):
    decision: str
    confidence: float
    reasoning: str
    # ... other fields
```

### Database Session Management
Always use dependency injection for database sessions:
```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Error Handling
The application uses FastAPI's HTTPException for consistent error responses.

## Deployment Notes

- Configured for Replit deployment (see `.replit` file)
- Runs on port 5000 by default
- PostgreSQL 16 required for database
- Python 3.11+ required for all features