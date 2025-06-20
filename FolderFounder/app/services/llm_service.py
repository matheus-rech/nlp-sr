"""
LLM service for dual AI screening
"""
import os
from typing import Dict, Any, Literal, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama

from app.schemas.schemas import ScreeningDecision, ProjectCriteria
from app.core.config import settings


class LLMScreeningService:
    """Service for managing dual LLM screening"""
    
    def __init__(self):
        self.conservative_llm = self._create_llm("conservative")
        self.liberal_llm = self._create_llm("liberal")
        self.output_parser = PydanticOutputParser(pydantic_object=ScreeningDecision)
    
    def _create_llm(self, strategy: Literal["conservative", "liberal"]):
        """Create LLM instance based on strategy and configuration"""
        # For now, using OpenAI. Can be extended to support multiple providers
        model = settings.CONSERVATIVE_MODEL if strategy == "conservative" else settings.LIBERAL_MODEL
        temperature = 0.1 if strategy == "conservative" else 0.3
        
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=settings.MAX_TOKENS,
            api_key=settings.OPENAI_API_KEY
        )
    
    def create_screening_prompt(
        self, 
        criteria: ProjectCriteria, 
        strategy: Literal["conservative", "liberal"]
    ) -> ChatPromptTemplate:
        """Create a screening prompt based on criteria and strategy"""
        
        if strategy == "conservative":
            persona = """You are Dr. Sarah Chen, a meticulous Cochrane reviewer with 20 years of experience.
            You are known for your conservative approach - when in doubt, you include studies to avoid missing 
            potentially relevant research. You carefully examine every detail and err on the side of inclusion."""
            approach = "minimize false negatives (include when uncertain)"
        else:
            persona = """You are Dr. Michael Rodriguez, a pragmatic evidence synthesizer with extensive experience
            in rapid reviews. You focus on efficiency while maintaining quality. You exclude studies that don't 
            clearly meet the criteria to keep the review focused and manageable."""
            approach = "balance efficiency with comprehensiveness (exclude when uncertain)"
        
        criteria_text = f"""
        Population: {criteria.population}
        Intervention: {criteria.intervention}
        Comparison: {criteria.comparison}
        Outcome: {criteria.outcome}
        Timeframe: {criteria.timeframe}
        Study Types: {criteria.study_types}
        
        Inclusion Criteria:
        - Language: {criteria.inclusion_language}
        - Publication: {criteria.inclusion_publication}
        - Sample Size: {criteria.inclusion_sample_size}
        - Data Availability: {criteria.inclusion_data_availability}
        - Other: {criteria.other_inclusion}
        
        Exclusion Criteria:
        - Study Types: {criteria.exclusion_study_types}
        - Populations: {criteria.exclusion_populations}
        - Interventions: {criteria.exclusion_interventions}
        - Languages: {criteria.exclusion_languages}
        - Other: {criteria.other_exclusion}
        
        Research Question: {criteria.research_question}
        """
        
        return ChatPromptTemplate.from_messages([
            ("system", f"""{persona}
            
            Your task is to screen academic citations for a systematic review. Your approach is to {approach}.
            
            IMPORTANT SCREENING PRINCIPLES:
            1. Inclusion/Exclusion criteria are MANDATORY - any violation of exclusion criteria means automatic exclusion
            2. Not all PICO-TT elements need to be explicitly stated in the abstract - use your expertise to infer relevance
            3. Consider how well the study addresses the research question, even with incomplete information
            4. Abstracts often lack complete details - make reasonable inferences based on available information
            5. Weight the importance of each PICO-TT element based on the research question
            
            DECISION GUIDELINES:
            - INCLUDE: Study likely meets criteria based on available information
            - EXCLUDE: Clear violation of exclusion criteria OR clearly irrelevant to research question
            - UNCERTAIN: Insufficient information to make a confident decision (requires full-text review)
            
            You must respond with a structured JSON output that includes:
            1. A decision (include/exclude/uncertain)
            2. Confidence level (0-100)
            3. Detailed reasoning explaining your decision
            4. Relevant quotes from the abstract supporting your decision
            5. PICO element assessment (which elements are present/absent/unclear)
            6. Quality indicators based on available information
            
            {self.output_parser.get_format_instructions()}
            """),
            ("human", f"""
            Review the following citation against these criteria:
            
            {criteria_text}
            
            REMEMBER:
            - Exclusion criteria are absolute - if met, exclude regardless of other factors
            - Missing PICO-TT elements don't automatically mean exclusion - consider overall relevance
            - The research question is the ultimate guide - does this study help answer it?
            
            Citation to review:
            Title: {{title}}
            Authors: {{authors}}
            Journal: {{journal}}
            Year: {{year}}
            Abstract: {{abstract}}
            Keywords: {{keywords}}
            
            Provide your screening decision:
            """)
        ])
    
    async def screen_citation(
        self,
        citation: Dict[str, Any],
        criteria: ProjectCriteria
    ) -> Dict[str, ScreeningDecision]:
        """Screen a citation using both conservative and liberal strategies"""
        
        # Create prompts for both strategies
        conservative_prompt = self.create_screening_prompt(criteria, "conservative")
        liberal_prompt = self.create_screening_prompt(criteria, "liberal")
        
        # Create chains with output parsing
        conservative_chain = conservative_prompt | self.conservative_llm | self.output_parser
        liberal_chain = liberal_prompt | self.liberal_llm | self.output_parser
        
        # Prepare input data
        input_data = {
            "title": citation.get("title", ""),
            "authors": citation.get("authors", ""),
            "journal": citation.get("journal", ""),
            "year": citation.get("year", ""),
            "abstract": citation.get("abstract", ""),
            "keywords": citation.get("keywords", "")
        }
        
        # Run both screenings in parallel
        import asyncio
        conservative_result, liberal_result = await asyncio.gather(
            conservative_chain.ainvoke(input_data),
            liberal_chain.ainvoke(input_data),
            return_exceptions=True
        )
        
        # Handle errors
        if isinstance(conservative_result, Exception):
            conservative_result = ScreeningDecision(
                decision="uncertain",
                confidence=0,
                reasoning=f"Error in conservative screening: {str(conservative_result)}",
                evidence_quotes=[],
                pico_assessment={},
                quality_indicators={}
            )
        
        if isinstance(liberal_result, Exception):
            liberal_result = ScreeningDecision(
                decision="uncertain",
                confidence=0,
                reasoning=f"Error in liberal screening: {str(liberal_result)}",
                evidence_quotes=[],
                pico_assessment={},
                quality_indicators={}
            )
        
        return {
            "conservative": conservative_result,
            "liberal": liberal_result
        }
    
    def determine_consensus(
        self,
        conservative: ScreeningDecision,
        liberal: ScreeningDecision
    ) -> tuple[str, str, float]:
        """Determine consensus between two screening decisions
        
        Returns:
            consensus: agree_include, agree_exclude, or dispute
            final_decision: include, exclude, or uncertain
            confidence_score: combined confidence score
        """
        
        if conservative.decision == liberal.decision:
            # Both agree
            consensus = f"agree_{conservative.decision}"
            final_decision = conservative.decision
            # Average confidence when they agree
            confidence_score = (conservative.confidence + liberal.confidence) / 2
        else:
            # Disagreement
            consensus = "dispute"
            
            # In case of dispute:
            # - If one says include and other says exclude/uncertain -> needs human review
            # - If one says uncertain -> go with the other's decision but lower confidence
            if "uncertain" in [conservative.decision, liberal.decision]:
                # One is uncertain, go with the certain one
                if conservative.decision != "uncertain":
                    final_decision = conservative.decision
                    confidence_score = conservative.confidence * 0.7  # Lower confidence
                else:
                    final_decision = liberal.decision
                    confidence_score = liberal.confidence * 0.7
            else:
                # Direct conflict (include vs exclude)
                final_decision = "uncertain"
                confidence_score = 50.0  # Medium confidence, needs human review
        
        return consensus, final_decision, confidence_score