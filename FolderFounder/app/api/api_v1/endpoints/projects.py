"""
Project management endpoints
"""
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.database import get_db
from app.models.models import Project, Citation, ScreeningResult, ActivityLog
from app.schemas.schemas import (
    Project as ProjectSchema,
    ProjectCreate,
    ProjectUpdate,
    ProjectInDB,
    ActivityLogEntry
)

router = APIRouter()


@router.get("/", response_model=List[ProjectInDB])
async def list_projects(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all projects with statistics"""
    query = db.query(Project)
    
    if status:
        query = query.filter(Project.status == status)
    
    projects = query.offset(skip).limit(limit).all()
    
    # Add statistics to each project
    results = []
    for project in projects:
        project_dict = project.__dict__.copy()
        
        # Get citation statistics
        total_citations = db.query(func.count(Citation.id)).filter(
            Citation.project_id == project.id
        ).scalar()
        
        screened_citations = db.query(func.count(ScreeningResult.id)).filter(
            ScreeningResult.project_id == project.id,
            ScreeningResult.final_decision.isnot(None)
        ).scalar()
        
        included_citations = db.query(func.count(ScreeningResult.id)).filter(
            ScreeningResult.project_id == project.id,
            ScreeningResult.final_decision == "include"
        ).scalar()
        
        excluded_citations = db.query(func.count(ScreeningResult.id)).filter(
            ScreeningResult.project_id == project.id,
            ScreeningResult.final_decision == "exclude"
        ).scalar()
        
        conflict_citations = db.query(func.count(ScreeningResult.id)).filter(
            ScreeningResult.project_id == project.id,
            ScreeningResult.consensus == "dispute"
        ).scalar()
        
        project_dict.update({
            "total_citations": total_citations,
            "screened_citations": screened_citations,
            "included_citations": included_citations,
            "excluded_citations": excluded_citations,
            "conflict_citations": conflict_citations
        })
        
        results.append(ProjectInDB(**project_dict))
    
    return results


@router.post("/", response_model=ProjectSchema)
async def create_project(
    project: ProjectCreate,
    db: Session = Depends(get_db)
):
    """Create a new project"""
    # TODO: Get current user from auth context
    # For now, using a placeholder user_id
    
    db_project = Project(
        name=project.name,
        description=project.description,
        screening_mode=project.screening_mode,
        criteria=project.criteria.dict() if project.criteria else None,
        owner_id=None  # TODO: Set from auth context
    )
    
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    
    # Log activity
    activity = ActivityLog(
        project_id=db_project.id,
        user_id=None,  # TODO: Set from auth context
        action="project_created",
        details={"project_name": project.name}
    )
    db.add(activity)
    db.commit()
    
    return db_project


@router.get("/{project_id}", response_model=ProjectInDB)
async def get_project(
    project_id: UUID,
    db: Session = Depends(get_db)
):
    """Get a specific project with statistics"""
    project = db.query(Project).filter(Project.id == project_id).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project_dict = project.__dict__.copy()
    
    # Get citation statistics
    total_citations = db.query(func.count(Citation.id)).filter(
        Citation.project_id == project.id
    ).scalar()
    
    screened_citations = db.query(func.count(ScreeningResult.id)).filter(
        ScreeningResult.project_id == project.id,
        ScreeningResult.final_decision.isnot(None)
    ).scalar()
    
    included_citations = db.query(func.count(ScreeningResult.id)).filter(
        ScreeningResult.project_id == project.id,
        ScreeningResult.final_decision == "include"
    ).scalar()
    
    excluded_citations = db.query(func.count(ScreeningResult.id)).filter(
        ScreeningResult.project_id == project.id,
        ScreeningResult.final_decision == "exclude"
    ).scalar()
    
    conflict_citations = db.query(func.count(ScreeningResult.id)).filter(
        ScreeningResult.project_id == project.id,
        ScreeningResult.consensus == "dispute"
    ).scalar()
    
    project_dict.update({
        "total_citations": total_citations,
        "screened_citations": screened_citations,
        "included_citations": included_citations,
        "excluded_citations": excluded_citations,
        "conflict_citations": conflict_citations
    })
    
    return ProjectInDB(**project_dict)


@router.put("/{project_id}", response_model=ProjectSchema)
async def update_project(
    project_id: UUID,
    project_update: ProjectUpdate,
    db: Session = Depends(get_db)
):
    """Update a project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Update fields if provided
    if project_update.name is not None:
        project.name = project_update.name
    if project_update.description is not None:
        project.description = project_update.description
    if project_update.criteria is not None:
        project.criteria = project_update.criteria.dict()
    if project_update.status is not None:
        project.status = project_update.status
    
    db.commit()
    db.refresh(project)
    
    # Log activity
    activity = ActivityLog(
        project_id=project.id,
        user_id=None,  # TODO: Set from auth context
        action="project_updated",
        details={"updated_fields": list(project_update.dict(exclude_unset=True).keys())}
    )
    db.add(activity)
    db.commit()
    
    return project


@router.delete("/{project_id}")
async def delete_project(
    project_id: UUID,
    db: Session = Depends(get_db)
):
    """Delete a project and all associated data"""
    project = db.query(Project).filter(Project.id == project_id).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Delete project (cascade will handle related records)
    db.delete(project)
    db.commit()
    
    return {"message": "Project deleted successfully"}


@router.get("/{project_id}/activity", response_model=List[ActivityLogEntry])
async def get_project_activity(
    project_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get activity log for a project"""
    activities = db.query(ActivityLog).filter(
        ActivityLog.project_id == project_id
    ).order_by(
        ActivityLog.timestamp.desc()
    ).offset(skip).limit(limit).all()
    
    return [
        ActivityLogEntry(
            action=activity.action,
            details=activity.details,
            timestamp=activity.timestamp,
            user_id=activity.user_id,
            project_id=activity.project_id
        )
        for activity in activities
    ]