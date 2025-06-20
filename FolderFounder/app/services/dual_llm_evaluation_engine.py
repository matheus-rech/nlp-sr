"""
Dual LLM Evaluation Engine
Handles systematic review screening using two LLMs with different strategies
"""
import os
import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import aiohttp
from enum import Enum

logger = logging.getLogger(__name__)


class Decision(Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    UNCERTAIN = "uncertain"


@dataclass
class ScreeningResult:
    """Result from a single LLM screening"""
    decision: str
    confidence: float
    reasoning: str
    pico_matches: Dict[str, bool]
    quality_score: float
    evidence_quotes: List[str]
    processing_time: float
    model: str
    strategy: str
    error: Optional[str] = None


@dataclass
class DualScreeningResult:
    """Combined result from both LLMs"""
    citation_id: str
    conservative_result: ScreeningResult
    liberal_result: ScreeningResult
    final_decision: str
    confidence_score: float
    conflict_detected: bool
    human_review_required: bool
    total_processing_time: float
    timestamp: str


class DualLLMEvaluationEngine:
    """Engine for dual LLM evaluation of citations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the evaluation engine with configuration"""
        self.config = config or {}
        
        # API Keys
        self.openai_api_key = os.getenv('OPENAI_API_KEY') or self.config.get('openai_api_key')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY') or self.config.get('anthropic_api_key')
        
        # Model configuration
        self.conservative_model = self.config.get('conservative_model', 'gpt-4')
        self.liberal_model = self.config.get('liberal_model', 'gpt-3.5-turbo')
        self.temperature_conservative = 0.1
        self.temperature_liberal = 0.3
        
        # Thresholds
        self.confidence_threshold = self.config.get('confidence_threshold', 70)
        self.conflict_threshold = self.config.get('conflict_threshold', 30)
        
        # Rate limiting
        self.rate_limit_delay = self.config.get('rate_limit_delay', 1)
        self.max_retries = 3
        
        # Session management
        self._session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()
    
    def create_prompt(self, citation: Dict[str, Any], criteria: Dict[str, Any], strategy: str) -> str:
        """Create evaluation prompt based on strategy"""
        
        if strategy == "conservative":
            persona = """You are Dr. Sarah Chen, a meticulous systematic reviewer with 20 years of experience 
            in evidence-based medicine. You follow Cochrane guidelines strictly and believe in minimizing 
            false negatives. When uncertain, you prefer to include studies for human review."""
            approach = "err on the side of inclusion to avoid missing potentially relevant studies"
        else:
            persona = """You are Dr. Michael Rodriguez, an efficient systematic reviewer who values both 
            thoroughness and practicality. You have extensive experience in rapid reviews and understand 
            the importance of focusing resources on the most relevant studies."""
            approach = "balance comprehensiveness with efficiency, excluding studies that clearly don't meet criteria"
        
        prompt = f"""{persona}

You are screening citations for a systematic review with the following criteria:

**PICO/PICOTT Criteria:**
- Population: {criteria.get('population', 'Not specified')}
- Intervention: {criteria.get('intervention', 'Not specified')}
- Comparator: {criteria.get('comparison', 'Not specified')}
- Outcome: {criteria.get('outcome', 'Not specified')}
- Timeframe: {criteria.get('timeframe', 'Not specified')}
- Study Types: {criteria.get('studyTypes', 'Not specified')}

**Research Question:** {criteria.get('researchQuestion', 'Not specified')}

**Inclusion Criteria:**
- Language: {criteria.get('inclusionLanguage', 'Not specified')}
- Publication: {criteria.get('inclusionPublication', 'Not specified')}
- Sample Size: {criteria.get('inclusionSampleSize', 'Not specified')}
- Data Availability: {criteria.get('inclusionDataAvailability', 'Not specified')}
- Other: {criteria.get('otherInclusion', 'Not specified')}

**Exclusion Criteria:**
- Study Types: {criteria.get('exclusionStudyTypes', 'Not specified')}
- Populations: {criteria.get('exclusionPopulations', 'Not specified')}
- Interventions: {criteria.get('exclusionInterventions', 'Not specified')}
- Languages: {criteria.get('exclusionLanguages', 'Not specified')}
- Other: {criteria.get('otherExclusion', 'Not specified')}

**Citation to evaluate:**
Title: {citation.get('title', 'Not provided')}
Authors: {citation.get('authors', 'Not provided')}
Journal: {citation.get('journal', 'Not provided')}
Year: {citation.get('year', 'Not provided')}
Abstract: {citation.get('abstract', 'Not provided')}

Your approach is to {approach}.

Provide your evaluation in the following JSON format:
{{
    "decision": "include" or "exclude" or "uncertain",
    "confidence": 0-100 (percentage),
    "reasoning": "Detailed explanation of your decision",
    "pico_matches": {{
        "population": true/false,
        "intervention": true/false,
        "comparator": true/false,
        "outcome": true/false,
        "timeframe": true/false,
        "study_type": true/false
    }},
    "quality_score": 0-100 (overall quality/relevance score),
    "evidence_quotes": ["List of specific quotes from the abstract supporting your decision"]
}}

Ensure your response is valid JSON only, with no additional text."""
        
        return prompt
    
    async def evaluate_single_llm(
        self, 
        citation: Dict[str, Any], 
        criteria: Dict[str, Any], 
        strategy: str,
        model: str
    ) -> ScreeningResult:
        """Evaluate a citation using a single LLM"""
        start_time = time.time()
        
        prompt = self.create_prompt(citation, criteria, strategy)
        
        # Determine which API to use based on model
        if model.startswith('gpt'):
            result = await self._call_openai(prompt, model, strategy)
        elif model.startswith('claude'):
            result = await self._call_anthropic(prompt, model, strategy)
        else:
            # Default to OpenAI API
            result = await self._call_openai(prompt, model, strategy)
        
        result.processing_time = time.time() - start_time
        return result
    
    async def _call_openai(self, prompt: str, model: str, strategy: str) -> ScreeningResult:
        """Call OpenAI API"""
        if not self.openai_api_key:
            return self._create_error_result("OpenAI API key not configured", model, strategy)
        
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        temperature = self.temperature_conservative if strategy == "conservative" else self.temperature_liberal
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1500
        }
        
        for attempt in range(self.max_retries):
            try:
                if not self._session:
                    self._session = aiohttp.ClientSession()
                
                async with self._session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        
                        # Parse JSON response
                        try:
                            eval_data = json.loads(content)
                            return ScreeningResult(
                                decision=eval_data.get('decision', 'uncertain'),
                                confidence=float(eval_data.get('confidence', 0)),
                                reasoning=eval_data.get('reasoning', ''),
                                pico_matches=eval_data.get('pico_matches', {}),
                                quality_score=float(eval_data.get('quality_score', 0)),
                                evidence_quotes=eval_data.get('evidence_quotes', []),
                                processing_time=0,  # Will be set by caller
                                model=model,
                                strategy=strategy
                            )
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}")
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.rate_limit_delay)
                                continue
                            return self._create_error_result(f"Invalid JSON response: {e}", model, strategy)
                    else:
                        error_text = await response.text()
                        logger.error(f"API error {response.status}: {error_text}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.rate_limit_delay * (attempt + 1))
                            continue
                        return self._create_error_result(f"API error {response.status}: {error_text}", model, strategy)
            
            except Exception as e:
                logger.error(f"Request error: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.rate_limit_delay)
                    continue
                return self._create_error_result(f"Request error: {e}", model, strategy)
        
        return self._create_error_result("Max retries exceeded", model, strategy)
    
    async def _call_anthropic(self, prompt: str, model: str, strategy: str) -> ScreeningResult:
        """Call Anthropic API"""
        if not self.anthropic_api_key:
            return self._create_error_result("Anthropic API key not configured", model, strategy)
        
        headers = {
            "x-api-key": self.anthropic_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        temperature = self.temperature_conservative if strategy == "conservative" else self.temperature_liberal
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1500
        }
        
        # Similar implementation to OpenAI but with Anthropic's API endpoint
        # This is a placeholder - you would implement the actual Anthropic API call
        return self._create_error_result("Anthropic API not yet implemented", model, strategy)
    
    def _create_error_result(self, error: str, model: str, strategy: str) -> ScreeningResult:
        """Create an error result"""
        return ScreeningResult(
            decision="uncertain",
            confidence=0,
            reasoning=f"Error during evaluation: {error}",
            pico_matches={},
            quality_score=0,
            evidence_quotes=[],
            processing_time=0,
            model=model,
            strategy=strategy,
            error=error
        )
    
    async def evaluate_citation(
        self, 
        citation: Dict[str, Any], 
        criteria: Dict[str, Any]
    ) -> DualScreeningResult:
        """Evaluate a citation using both conservative and liberal strategies"""
        start_time = time.time()
        
        # Run both evaluations in parallel
        conservative_task = self.evaluate_single_llm(
            citation, criteria, "conservative", self.conservative_model
        )
        liberal_task = self.evaluate_single_llm(
            citation, criteria, "liberal", self.liberal_model
        )
        
        conservative_result, liberal_result = await asyncio.gather(
            conservative_task, liberal_task
        )
        
        # Determine final decision and conflicts
        final_decision, confidence_score, conflict_detected, human_review_required = \
            self._analyze_results(conservative_result, liberal_result)
        
        total_time = time.time() - start_time
        
        return DualScreeningResult(
            citation_id=citation.get('citation_id', citation.get('id', 'unknown')),
            conservative_result=conservative_result,
            liberal_result=liberal_result,
            final_decision=final_decision,
            confidence_score=confidence_score,
            conflict_detected=conflict_detected,
            human_review_required=human_review_required,
            total_processing_time=total_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _analyze_results(
        self, 
        conservative: ScreeningResult, 
        liberal: ScreeningResult
    ) -> Tuple[str, float, bool, bool]:
        """Analyze results from both LLMs to determine final decision"""
        
        # Handle errors
        if conservative.error or liberal.error:
            return "uncertain", 0, True, True
        
        # Check for agreement
        if conservative.decision == liberal.decision:
            # Both agree
            avg_confidence = (conservative.confidence + liberal.confidence) / 2
            return conservative.decision, avg_confidence, False, False
        
        # Disagreement cases
        conflict_detected = True
        
        # If one is uncertain
        if conservative.decision == "uncertain" or liberal.decision == "uncertain":
            # Go with the certain decision but flag for review
            if conservative.decision != "uncertain":
                return conservative.decision, conservative.confidence * 0.8, conflict_detected, True
            else:
                return liberal.decision, liberal.confidence * 0.8, conflict_detected, True
        
        # Direct conflict (include vs exclude)
        # Conservative approach: if conservative says include, include it
        if conservative.decision == "include":
            return "include", conservative.confidence * 0.7, conflict_detected, True
        else:
            # Both have strong opinions but disagree
            # Flag for human review
            return "uncertain", 50, conflict_detected, True
    
    async def evaluate_batch(
        self, 
        citations: List[Dict[str, Any]], 
        criteria: Dict[str, Any],
        batch_size: int = 5
    ) -> List[DualScreeningResult]:
        """Evaluate a batch of citations"""
        results = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(citations), batch_size):
            batch = citations[i:i + batch_size]
            
            # Evaluate batch in parallel
            batch_results = await asyncio.gather(
                *[self.evaluate_citation(citation, criteria) for citation in batch]
            )
            
            results.extend(batch_results)
            
            # Rate limiting between batches
            if i + batch_size < len(citations):
                await asyncio.sleep(self.rate_limit_delay)
        
        return results
    
    def get_statistics(self, results: List[DualScreeningResult]) -> Dict[str, Any]:
        """Calculate statistics from evaluation results"""
        total = len(results)
        if total == 0:
            return {}
        
        stats = {
            'total_citations': total,
            'included': sum(1 for r in results if r.final_decision == 'include'),
            'excluded': sum(1 for r in results if r.final_decision == 'exclude'),
            'uncertain': sum(1 for r in results if r.final_decision == 'uncertain'),
            'conflicts': sum(1 for r in results if r.conflict_detected),
            'human_review_required': sum(1 for r in results if r.human_review_required),
            'average_confidence': sum(r.confidence_score for r in results) / total,
            'average_processing_time': sum(r.total_processing_time for r in results) / total,
            'errors': sum(1 for r in results if r.conservative_result.error or r.liberal_result.error)
        }
        
        stats['inclusion_rate'] = (stats['included'] / total) * 100
        stats['exclusion_rate'] = (stats['excluded'] / total) * 100
        stats['conflict_rate'] = (stats['conflicts'] / total) * 100
        
        return stats