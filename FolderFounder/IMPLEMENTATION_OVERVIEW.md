# Otto-SR Implementation Overview

## 🎯 Project Status

This implementation merges the best features from multiple approaches to create a comprehensive systematic review screening tool that prevents researcher burnout through intelligent automation.

## 🏗️ Architecture

### Backend (FastAPI)
```
backend/
├── app/
│   ├── api/
│   │   └── api_v1/
│   │       ├── api.py              # Main API router
│   │       └── endpoints/
│   │           ├── health.py       # Health checks
│   │           ├── projects.py     # Project management
│   │           ├── citations.py    # Citation upload/management
│   │           ├── screening.py    # AI screening process
│   │           └── export.py       # Export functionality
│   ├── core/
│   │   ├── config.py              # Configuration settings
│   │   └── database.py            # Database connection
│   ├── models/
│   │   └── models.py              # SQLAlchemy models
│   ├── schemas/
│   │   └── schemas.py             # Pydantic schemas
│   └── services/
│       ├── database_file_parser.py     # Multi-format citation parser
│       ├── dual_llm_evaluation_engine.py   # Dual AI screening
│       ├── file_parser.py              # Enhanced file parsing
│       └── llm_service.py              # LLM integration service
└── requirements.txt
```

### Frontend (React + TypeScript)
```
frontend/
├── src/
│   ├── components/
│   │   ├── AbstractNavigator.tsx   # Citation navigation UI
│   │   ├── CitationImport.jsx      # File upload component
│   │   └── [other components]
│   ├── pages/
│   ├── services/
│   └── types/
├── package.json
└── vite.config.ts
```

## 🚀 Key Features Implemented

### 1. **Multi-Format Citation Import**
- ✅ RIS, XML, EndNote, BibTeX, Zotero, CSV, PubMed formats
- ✅ Automatic duplicate detection
- ✅ Relevance scoring
- ✅ Batch processing with progress tracking

### 2. **Dual LLM Screening**
- ✅ Conservative vs Liberal AI strategies
- ✅ Multiple LLM provider support (OpenAI, Anthropic, etc.)
- ✅ Structured output with Pydantic validation
- ✅ Conflict detection and resolution
- ✅ Evidence quote extraction

### 3. **Abstract Navigation Interface**
- ✅ Sequential navigation through citations
- ✅ Real-time metrics dashboard
- ✅ Filter by status (included/excluded/conflicts)
- ✅ Detailed AI reasoning display
- ✅ PICO criteria assessment visualization

### 4. **Project Management**
- ✅ Create/update systematic review projects
- ✅ PICO-TT criteria configuration
- ✅ Activity logging and audit trail
- ✅ Project statistics and progress tracking

### 5. **Real-Time Updates**
- ✅ Server-Sent Events for screening progress
- ✅ Live metrics updates
- ✅ Background task processing
- ✅ Job status tracking

## 📊 Database Schema

### Core Tables:
1. **users** - Authentication and user management
2. **projects** - Systematic review projects
3. **citations** - Uploaded research papers
4. **screening_results** - AI evaluation results
5. **activity_logs** - Audit trail

## 🔧 API Endpoints

### Projects
- `GET /api/v1/projects` - List all projects
- `POST /api/v1/projects` - Create new project
- `GET /api/v1/projects/{id}` - Get project details
- `PUT /api/v1/projects/{id}` - Update project
- `DELETE /api/v1/projects/{id}` - Delete project

### Citations
- `POST /api/v1/citations/upload/{project_id}` - Upload citation files
- `GET /api/v1/citations/{project_id}` - List citations with filtering
- `GET /api/v1/citations/{project_id}/{citation_id}` - Get citation details
- `DELETE /api/v1/citations/{project_id}/{citation_id}` - Delete citation

### Screening
- `POST /api/v1/screening/{project_id}/start` - Start AI screening
- `GET /api/v1/screening/{project_id}/progress/{job_id}` - Get job progress
- `GET /api/v1/screening/{project_id}/stream/{job_id}` - Stream progress (SSE)
- `GET /api/v1/screening/{project_id}/results` - Get screening results
- `PUT /api/v1/screening/{project_id}/results/{result_id}` - Update with human review

## 🚦 Getting Started

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Environment Variables
```env
# Database
DATABASE_URL=postgresql://user:password@localhost/ottosr

# LLM Providers
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key

# Security
SECRET_KEY=your-secret-key
```

## 💡 Usage Flow

1. **Create Project** → Define PICO-TT criteria
2. **Upload Citations** → Import from multiple sources
3. **Start Screening** → Dual AI evaluation begins
4. **Navigate Results** → Review with AbstractNavigator
5. **Resolve Conflicts** → Human review when needed
6. **Export Results** → Multiple format options

## 🎯 Burnout Prevention Features

1. **Automated Import** - No manual data entry
2. **Intelligent Screening** - AI handles initial review
3. **Conflict Detection** - Only review disputed cases
4. **Progress Tracking** - Clear visibility of workload
5. **Batch Processing** - Handle thousands efficiently

## 🔄 Next Steps

### High Priority
- [ ] Complete authentication system
- [ ] Add export endpoints
- [ ] Implement WebSocket for real-time updates
- [ ] Create Docker configuration

### Medium Priority
- [ ] Add full-text PDF processing
- [ ] Implement PRISMA flow diagram
- [ ] Create data extraction forms
- [ ] Add team collaboration features

### Future Enhancements
- [ ] Machine learning model training
- [ ] Custom LLM fine-tuning
- [ ] Integration with reference managers
- [ ] Advanced analytics dashboard

## 📚 Documentation

- See `CLAUDE.md` for development guidance
- Check `otto_sr_documentation.md` for detailed specs
- Review `requirements_analysis.md` for feature requirements

## 🤝 Contributing

This project follows a modular architecture making it easy to:
- Add new citation formats
- Integrate additional LLM providers
- Create custom UI components
- Extend API functionality

The codebase is designed for collaboration and preventing developer burnout too!