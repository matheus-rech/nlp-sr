"""
Export endpoints for systematic review results
"""
import io
import csv
import json
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.core.database import get_db
from app.models.models import Project, Citation, ScreeningResult
from app.schemas.schemas import ExportRequest

router = APIRouter()


@router.post("/{project_id}")
async def export_results(
    project_id: UUID,
    request: ExportRequest,
    db: Session = Depends(get_db)
):
    """Export screening results in various formats"""
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get screening results with citations
    query = db.query(ScreeningResult, Citation).join(
        Citation, ScreeningResult.citation_id == Citation.id
    ).filter(ScreeningResult.project_id == project_id)
    
    # Filter by decision if requested
    if request.include_only:
        query = query.filter(ScreeningResult.final_decision == "include")
    
    results = query.all()
    
    # Generate export based on format
    if request.format == "csv":
        return _export_csv(results, request.include_screening_details)
    elif request.format == "json":
        return _export_json(results, request.include_screening_details)
    elif request.format == "ris":
        return _export_ris(results)
    elif request.format == "bibtex":
        return _export_bibtex(results)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")


def _export_csv(results, include_details: bool):
    """Export results as CSV"""
    output = io.StringIO()
    
    # Define columns
    columns = [
        'title', 'authors', 'journal', 'year', 'doi', 'abstract',
        'final_decision', 'confidence_score', 'consensus'
    ]
    
    if include_details:
        columns.extend([
            'ai1_decision', 'ai1_confidence', 'ai1_reasoning',
            'ai2_decision', 'ai2_confidence', 'ai2_reasoning',
            'human_decision', 'human_notes'
        ])
    
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    
    for result, citation in results:
        row = {
            'title': citation.title,
            'authors': citation.authors,
            'journal': citation.journal,
            'year': citation.year,
            'doi': citation.doi,
            'abstract': citation.abstract,
            'final_decision': result.final_decision,
            'confidence_score': result.confidence_score,
            'consensus': result.consensus
        }
        
        if include_details:
            row.update({
                'ai1_decision': result.ai1_result.get('decision'),
                'ai1_confidence': result.ai1_result.get('confidence'),
                'ai1_reasoning': result.ai1_result.get('reasoning'),
                'ai2_decision': result.ai2_result.get('decision'),
                'ai2_confidence': result.ai2_result.get('confidence'),
                'ai2_reasoning': result.ai2_result.get('reasoning'),
                'human_decision': result.human_decision,
                'human_notes': result.human_notes
            })
        
        writer.writerow(row)
    
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=screening_results.csv"}
    )


def _export_json(results, include_details: bool):
    """Export results as JSON"""
    data = []
    
    for result, citation in results:
        item = {
            'citation': {
                'id': str(citation.id),
                'title': citation.title,
                'authors': citation.authors,
                'journal': citation.journal,
                'year': citation.year,
                'doi': citation.doi,
                'abstract': citation.abstract,
                'keywords': citation.keywords
            },
            'screening': {
                'final_decision': result.final_decision,
                'confidence_score': result.confidence_score,
                'consensus': result.consensus
            }
        }
        
        if include_details:
            item['screening']['ai1_result'] = result.ai1_result
            item['screening']['ai2_result'] = result.ai2_result
            item['screening']['human_review'] = {
                'decision': result.human_decision,
                'notes': result.human_notes,
                'reviewed_at': result.reviewed_at.isoformat() if result.reviewed_at else None
            }
        
        data.append(item)
    
    return StreamingResponse(
        io.BytesIO(json.dumps(data, indent=2).encode()),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=screening_results.json"}
    )


def _export_ris(results):
    """Export included citations as RIS format"""
    output = io.StringIO()
    
    for result, citation in results:
        if result.final_decision != "include":
            continue
            
        output.write("TY  - JOUR\n")
        output.write(f"TI  - {citation.title}\n")
        
        if citation.authors:
            for author in citation.authors.split(';'):
                output.write(f"AU  - {author.strip()}\n")
        
        if citation.journal:
            output.write(f"JO  - {citation.journal}\n")
        
        if citation.year:
            output.write(f"PY  - {citation.year}\n")
        
        if citation.abstract:
            output.write(f"AB  - {citation.abstract}\n")
        
        if citation.doi:
            output.write(f"DO  - {citation.doi}\n")
        
        if citation.keywords:
            for keyword in citation.keywords.split(';'):
                output.write(f"KW  - {keyword.strip()}\n")
        
        output.write("ER  - \n\n")
    
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=included_citations.ris"}
    )


def _export_bibtex(results):
    """Export included citations as BibTeX format"""
    output = io.StringIO()
    
    for idx, (result, citation) in enumerate(results):
        if result.final_decision != "include":
            continue
        
        # Generate cite key
        first_author = citation.authors.split(',')[0].split(';')[0] if citation.authors else "Unknown"
        cite_key = f"{first_author}{citation.year or 'YYYY'}_{idx}"
        
        output.write(f"@article{{{cite_key},\n")
        output.write(f"  title = {{{citation.title}}},\n")
        
        if citation.authors:
            authors = " and ".join([a.strip() for a in citation.authors.split(';')])
            output.write(f"  author = {{{authors}}},\n")
        
        if citation.journal:
            output.write(f"  journal = {{{citation.journal}}},\n")
        
        if citation.year:
            output.write(f"  year = {{{citation.year}}},\n")
        
        if citation.doi:
            output.write(f"  doi = {{{citation.doi}}},\n")
        
        if citation.abstract:
            output.write(f"  abstract = {{{citation.abstract}}},\n")
        
        if citation.keywords:
            output.write(f"  keywords = {{{citation.keywords}}},\n")
        
        output.write("}\n\n")
    
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=included_citations.bib"}
    )