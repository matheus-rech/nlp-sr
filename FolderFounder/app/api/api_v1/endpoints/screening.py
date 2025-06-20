"""
Screening endpoints for citation evaluation
"""
import asyncio
import uuid
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import and_
import json

from app.core.database import get_db
from app.models.models import Citation, Project, ScreeningResult, ActivityLog
from app.schemas.schemas import (
    ScreeningJobRequest,
    ScreeningJobResponse,
    ScreeningResult as ScreeningResultSchema,
    ScreeningResultUpdate,
    ScreeningProgress
)
from app.services.dual_llm_evaluation_engine import DualLLMEvaluationEngine

router = APIRouter()

# Store active screening jobs (in production, use Redis or similar)
active_jobs = {}


@router.post("/{project_id}/start", response_model=ScreeningJobResponse)
async def start_screening(
    project_id: UUID,
    request: ScreeningJobRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start screening citations for a project"""
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Update project criteria if provided
    if request.criteria:
        project.criteria = request.criteria.dict()
        db.commit()
    
    # Get unscreened citations
    unscreened_citations = db.query(Citation).outerjoin(ScreeningResult).filter(
        Citation.project_id == project_id,
        ScreeningResult.id.is_(None)
    ).all()
    
    if not unscreened_citations:
        raise HTTPException(status_code=400, detail="No unscreened citations found")
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job tracking
    active_jobs[job_id] = {
        "project_id": str(project_id),
        "status": "running",
        "total": len(unscreened_citations),
        "processed": 0,
        "completed": 0,
        "errors": 0,
        "conflicts": 0
    }
    
    # Start background screening task
    background_tasks.add_task(
        _screen_citations_task,
        job_id,
        project_id,
        unscreened_citations,
        project.criteria,
        request.batch_size
    )
    
    # Log activity
    activity = ActivityLog(
        project_id=project_id,
        user_id=None,  # TODO: Get from auth context
        action="screening_started",
        details={
            "job_id": job_id,
            "total_citations": len(unscreened_citations),
            "batch_size": request.batch_size
        }
    )
    db.add(activity)
    db.commit()
    
    return ScreeningJobResponse(
        job_id=job_id,
        project_id=project_id,
        total_citations=len(unscreened_citations),
        message=f"Screening started for {len(unscreened_citations)} citations"
    )


@router.get("/{project_id}/progress/{job_id}", response_model=ScreeningProgress)
async def get_screening_progress(
    project_id: UUID,
    job_id: str
):
    """Get progress of a screening job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = active_jobs[job_id]
    if job_info["project_id"] != str(project_id):
        raise HTTPException(status_code=403, detail="Job does not belong to this project")
    
    return ScreeningProgress(
        job_id=job_id,
        project_id=project_id,
        **job_info
    )


@router.get("/{project_id}/stream/{job_id}")
async def stream_screening_progress(
    project_id: UUID,
    job_id: str
):
    """Stream screening progress using Server-Sent Events"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = active_jobs[job_id]
    if job_info["project_id"] != str(project_id):
        raise HTTPException(status_code=403, detail="Job does not belong to this project")
    
    async def event_generator():
        while True:
            # Check if job is still active
            if job_id not in active_jobs:
                yield f"event: job_complete\ndata: Job completed or cancelled\n\n"
                break
            
            job_status = active_jobs[job_id]
            
            # Send current status
            yield f"event: progress_update\ndata: {json.dumps(job_status)}\n\n"
            
            # Check if job is complete
            if job_status["status"] in ["completed", "failed"]:
                yield f"event: job_complete\ndata: {json.dumps(job_status)}\n\n"
                break
            
            # Wait before next update
            await asyncio.sleep(1)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@router.get("/{project_id}/results", response_model=List[ScreeningResultSchema])
async def get_screening_results(
    project_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    consensus_filter: Optional[str] = None,
    decision_filter: Optional[str] = None,
    human_review_only: bool = False,
    db: Session = Depends(get_db)
):
    """Get screening results for a project"""
    query = db.query(ScreeningResult).filter(
        ScreeningResult.project_id == project_id
    )
    
    # Apply filters
    if consensus_filter:
        query = query.filter(ScreeningResult.consensus == consensus_filter)
    
    if decision_filter:
        query = query.filter(ScreeningResult.final_decision == decision_filter)
    
    if human_review_only:
        query = query.filter(ScreeningResult.human_decision.is_(None))
    
    # Get results with pagination
    results = query.offset(skip).limit(limit).all()
    
    return results


@router.put("/{project_id}/results/{result_id}", response_model=ScreeningResultSchema)
async def update_screening_result(
    project_id: UUID,
    result_id: UUID,
    update: ScreeningResultUpdate,
    db: Session = Depends(get_db)
):
    """Update a screening result (human review)"""
    result = db.query(ScreeningResult).filter(
        ScreeningResult.project_id == project_id,
        ScreeningResult.id == result_id
    ).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Screening result not found")
    
    # Update human review fields
    if update.human_decision:
        result.human_decision = update.human_decision
        result.reviewed_by_id = None  # TODO: Get from auth context
        result.reviewed_at = datetime.utcnow()
    
    if update.human_notes:
        result.human_notes = update.human_notes
    
    db.commit()
    db.refresh(result)
    
    # Log activity
    activity = ActivityLog(
        project_id=project_id,
        user_id=None,  # TODO: Get from auth context
        action="screening_result_reviewed",
        details={
            "result_id": str(result_id),
            "citation_id": str(result.citation_id),
            "human_decision": update.human_decision
        }
    )
    db.add(activity)
    db.commit()
    
    return result


async def _screen_citations_task(
    job_id: str,
    project_id: UUID,
    citations: List[Citation],
    criteria: dict,
    batch_size: int
):
    """Background task to screen citations"""
    from datetime import datetime
    from app.core.database import SessionLocal
    
    db = SessionLocal()
    evaluation_engine = DualLLMEvaluationEngine()
    
    try:
        async with evaluation_engine:
            # Process citations in batches
            for i in range(0, len(citations), batch_size):
                batch = citations[i:i + batch_size]
                
                # Convert citations to dict format
                citation_dicts = []
                for citation in batch:
                    citation_dict = {
                        "citation_id": str(citation.id),
                        "title": citation.title,
                        "abstract": citation.abstract,
                        "authors": citation.authors,
                        "journal": citation.journal,
                        "year": citation.year,
                        "doi": citation.doi,
                        "keywords": citation.keywords
                    }
                    citation_dicts.append(citation_dict)
                
                # Evaluate batch
                results = await evaluation_engine.evaluate_batch(
                    citation_dicts,
                    criteria,
                    batch_size=batch_size
                )
                
                # Save results to database
                for result in results:
                    db_result = ScreeningResult(
                        citation_id=result.citation_id,
                        project_id=project_id,
                        ai1_result={
                            "decision": result.conservative_result.decision,
                            "confidence": result.conservative_result.confidence,
                            "reasoning": result.conservative_result.reasoning,
                            "pico_matches": result.conservative_result.pico_matches,
                            "quality_score": result.conservative_result.quality_score,
                            "evidence_quotes": result.conservative_result.evidence_quotes,
                            "model": result.conservative_result.model,
                            "strategy": result.conservative_result.strategy
                        },
                        ai2_result={
                            "decision": result.liberal_result.decision,
                            "confidence": result.liberal_result.confidence,
                            "reasoning": result.liberal_result.reasoning,
                            "pico_matches": result.liberal_result.pico_matches,
                            "quality_score": result.liberal_result.quality_score,
                            "evidence_quotes": result.liberal_result.evidence_quotes,
                            "model": result.liberal_result.model,
                            "strategy": result.liberal_result.strategy
                        },
                        consensus="dispute" if result.conflict_detected else f"agree_{result.final_decision}",
                        final_decision=result.final_decision,
                        confidence_score=result.confidence_score,
                        screening_time_ms=int(result.total_processing_time * 1000)
                    )
                    
                    db.add(db_result)
                
                db.commit()
                
                # Update job progress
                active_jobs[job_id]["processed"] += len(batch)
                active_jobs[job_id]["completed"] += len([r for r in results if not r.conservative_result.error])
                active_jobs[job_id]["errors"] += len([r for r in results if r.conservative_result.error])
                active_jobs[job_id]["conflicts"] += len([r for r in results if r.conflict_detected])
        
        # Mark job as complete
        active_jobs[job_id]["status"] = "completed"
        
        # Log completion
        activity = ActivityLog(
            project_id=project_id,
            user_id=None,
            action="screening_completed",
            details={
                "job_id": job_id,
                "total_processed": active_jobs[job_id]["processed"],
                "conflicts": active_jobs[job_id]["conflicts"],
                "errors": active_jobs[job_id]["errors"]
            }
        )
        db.add(activity)
        db.commit()
        
    except Exception as e:
        # Mark job as failed
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = str(e)
        
        # Log error
        activity = ActivityLog(
            project_id=project_id,
            user_id=None,
            action="screening_failed",
            details={
                "job_id": job_id,
                "error": str(e)
            }
        )
        db.add(activity)
        db.commit()
        
    finally:
        db.close()