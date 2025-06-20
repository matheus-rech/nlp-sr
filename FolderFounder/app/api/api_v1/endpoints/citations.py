"""
Citation management endpoints
"""
import uuid
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_

from app.core.database import get_db
from app.models.models import Citation, Project, ActivityLog
from app.schemas.schemas import (
    Citation as CitationSchema,
    CitationCreate,
    CitationInDB,
    FileUploadResponse
)
from app.services.database_file_parser import DatabaseFileParser

router = APIRouter()


@router.post("/upload/{project_id}", response_model=FileUploadResponse)
async def upload_citations(
    project_id: UUID,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Upload citation files for a project"""
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    parser = DatabaseFileParser()
    upload_batch_id = str(uuid.uuid4())
    total_citations = 0
    successful_imports = 0
    duplicates = 0
    errors = []
    
    for file in files:
        try:
            # Read file content
            content = await file.read()
            content_str = content.decode('utf-8', errors='ignore')
            
            # Parse file
            citations = parser.parse_file(
                file.filename,
                content_str,
                file.filename.split('.')[-1].lower()
            )
            
            # Process each citation
            for citation_data in citations:
                total_citations += 1
                
                # Check for duplicates based on DOI or title
                existing = None
                if citation_data.get('doi'):
                    existing = db.query(Citation).filter(
                        Citation.project_id == project_id,
                        Citation.doi == citation_data['doi']
                    ).first()
                
                if not existing and citation_data.get('title'):
                    # Check by title similarity
                    existing = db.query(Citation).filter(
                        Citation.project_id == project_id,
                        Citation.title == citation_data['title']
                    ).first()
                
                if existing:
                    duplicates += 1
                    continue
                
                # Create new citation
                db_citation = Citation(
                    project_id=project_id,
                    title=citation_data.get('title', ''),
                    authors=citation_data.get('authors', ''),
                    journal=citation_data.get('journal', ''),
                    year=citation_data.get('year'),
                    abstract=citation_data.get('abstract', ''),
                    doi=citation_data.get('doi', ''),
                    pmid=citation_data.get('pmid', ''),
                    keywords=citation_data.get('keywords', ''),
                    full_text_url=citation_data.get('url', ''),
                    relevance_score=citation_data.get('relevance_score', 0.5),
                    upload_batch_id=upload_batch_id
                )
                
                db.add(db_citation)
                successful_imports += 1
                
        except Exception as e:
            errors.append(f"Error processing {file.filename}: {str(e)}")
            continue
    
    # Commit all citations
    db.commit()
    
    # Log activity
    activity = ActivityLog(
        project_id=project_id,
        user_id=None,  # TODO: Get from auth context
        action="citations_uploaded",
        details={
            "upload_batch_id": upload_batch_id,
            "files": [f.filename for f in files],
            "total_citations": total_citations,
            "successful_imports": successful_imports,
            "duplicates": duplicates,
            "errors": errors
        }
    )
    db.add(activity)
    db.commit()
    
    return FileUploadResponse(
        project_id=project_id,
        citations_count=successful_imports,
        upload_batch_id=upload_batch_id,
        message=f"Successfully imported {successful_imports} citations. {duplicates} duplicates skipped."
    )


@router.get("/{project_id}", response_model=List[CitationInDB])
async def list_citations(
    project_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    has_abstract: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """List citations for a project with filtering"""
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    query = db.query(Citation).filter(Citation.project_id == project_id)
    
    # Apply filters
    if search:
        search_filter = or_(
            Citation.title.ilike(f"%{search}%"),
            Citation.authors.ilike(f"%{search}%"),
            Citation.abstract.ilike(f"%{search}%"),
            Citation.keywords.ilike(f"%{search}%"),
            Citation.journal.ilike(f"%{search}%")
        )
        query = query.filter(search_filter)
    
    if year_from:
        query = query.filter(Citation.year >= year_from)
    
    if year_to:
        query = query.filter(Citation.year <= year_to)
    
    if has_abstract is not None:
        if has_abstract:
            query = query.filter(Citation.abstract != '')
        else:
            query = query.filter(or_(Citation.abstract == '', Citation.abstract.is_(None)))
    
    # Get total count before pagination
    total = query.count()
    
    # Apply pagination
    citations = query.offset(skip).limit(limit).all()
    
    # Add screening results to response
    results = []
    for citation in citations:
        citation_dict = citation.__dict__.copy()
        citation_dict['screening_result'] = citation.screening_result
        results.append(CitationInDB(**citation_dict))
    
    return results


@router.get("/{project_id}/{citation_id}", response_model=CitationInDB)
async def get_citation(
    project_id: UUID,
    citation_id: UUID,
    db: Session = Depends(get_db)
):
    """Get a specific citation"""
    citation = db.query(Citation).filter(
        Citation.project_id == project_id,
        Citation.id == citation_id
    ).first()
    
    if not citation:
        raise HTTPException(status_code=404, detail="Citation not found")
    
    citation_dict = citation.__dict__.copy()
    citation_dict['screening_result'] = citation.screening_result
    
    return CitationInDB(**citation_dict)


@router.delete("/{project_id}/{citation_id}")
async def delete_citation(
    project_id: UUID,
    citation_id: UUID,
    db: Session = Depends(get_db)
):
    """Delete a citation"""
    citation = db.query(Citation).filter(
        Citation.project_id == project_id,
        Citation.id == citation_id
    ).first()
    
    if not citation:
        raise HTTPException(status_code=404, detail="Citation not found")
    
    db.delete(citation)
    db.commit()
    
    # Log activity
    activity = ActivityLog(
        project_id=project_id,
        user_id=None,  # TODO: Get from auth context
        action="citation_deleted",
        details={
            "citation_id": str(citation_id),
            "citation_title": citation.title
        }
    )
    db.add(activity)
    db.commit()
    
    return {"message": "Citation deleted successfully"}


@router.post("/{project_id}/deduplicate")
async def deduplicate_citations(
    project_id: UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Find and mark duplicate citations in a project"""
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # This would typically be a background task
    # For now, return a placeholder response
    background_tasks.add_task(
        _deduplicate_citations_task,
        project_id,
        db
    )
    
    return {
        "message": "Deduplication started",
        "project_id": str(project_id)
    }


async def _deduplicate_citations_task(project_id: UUID, db: Session):
    """Background task to find and mark duplicates"""
    # Implementation would go here
    # This would:
    # 1. Find citations with same DOI
    # 2. Find citations with similar titles
    # 3. Mark duplicates in database
    # 4. Log results
    pass