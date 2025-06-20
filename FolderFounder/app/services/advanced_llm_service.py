"""
Advanced LLM Service with GPT-4.1, O3, and Gemini 2.5 Flash-Lite Support
"""
import os
import asyncio
import aiohttp
import json
from typing import Dict, Any, Literal, Optional, List
from datetime import datetime
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from app.schemas.schemas import ScreeningDecision, ProjectCriteria
from app.core.config import settings


class AdvancedLLMService:
    """Service for managing advanced LLM screening with GPT-4.1, O3, and Gemini 2.5 Flash-Lite"""
    
    def __init__(self):
        # Model configurations for our super special LLMs
        self.models = {
            "gpt-4.1": {
                "provider": "openai",
                "model_name": "gpt-4.1",
                "context_window": 1000000,  # 1M tokens
                "features": ["long_context", "multimodal", "instruction_following"],
                "cost_per_1k_input": 0.015,
                "cost_per_1k_output": 0.06,
                "strategy": "ultra_precise"
            },
            "o3-pro": {
                "provider": "openai",
                "model_name": "o3-pro",
                "features": ["reasoning", "step_by_step", "tools", "web_search"],
                "cost_per_1k_input": 0.02,
                "cost_per_1k_output": 0.08,
                "strategy": "deep_reasoning"
            },
            "gemini-2.5-flash-lite": {
                "provider": "google",
                "model_name": "gemini-2.5-flash-lite",
                "context_window": 1000000,  # 1M tokens
                "features": ["ultra_fast", "cost_efficient", "multimodal"],
                "cost_per_1k_input": 0.0001,
                "cost_per_1k_output": 0.0004,
                "strategy": "rapid_screening"
            }
        }
        
        self.output_parser = PydanticOutputParser(pydantic_object=ScreeningDecision)
        self._sessions = {}
    
    async def __aenter__(self):
        """Async context manager entry"""
        # Initialize sessions for each provider
        self._sessions['openai'] = aiohttp.ClientSession()
        self._sessions['google'] = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        for session in self._sessions.values():
            await session.close()
    
    def _create_llm(self, model_key: str):
        """Create LLM instance based on model configuration"""
        config = self.models[model_key]
        
        if config["provider"] == "openai":
            if model_key == "o3-pro":
                # O3 requires special configuration for reasoning
                return ChatOpenAI(
                    model=config["model_name"],
                    temperature=0.1,
                    max_tokens=4000,
                    api_key=settings.OPENAI_API_KEY,
                    model_kwargs={
                        "reasoning_effort": "high",  # For deep analysis
                        "tools": ["search", "analyze_files", "python"]
                    }
                )
            else:
                # GPT-4.1 configuration
                return ChatOpenAI(
                    model=config["model_name"],
                    temperature=0.2,
                    max_tokens=8000,
                    api_key=settings.OPENAI_API_KEY,
                    model_kwargs={
                        "response_format": {"type": "json_object"}
                    }
                )
        
        elif config["provider"] == "google":
            # Gemini 2.5 Flash-Lite configuration
            return ChatGoogleGenerativeAI(
                model=config["model_name"],
                temperature=0.3,
                max_output_tokens=2000,
                google_api_key=settings.GOOGLE_API_KEY,
                convert_system_message_to_human=True
            )
    
    def create_advanced_screening_prompt(
        self, 
        criteria: ProjectCriteria, 
        model_key: str
    ) -> ChatPromptTemplate:
        """Create model-specific prompts optimized for each LLM's strengths"""
        
        if model_key == "gpt-4.1":
            # Ultra-precise prompt for GPT-4.1's superior instruction following
            system_prompt = """You are an expert systematic reviewer with decades of experience in evidence-based medicine.
            
You have been enhanced with GPT-4.1's advanced capabilities:
- Perfect instruction following with 1M token context
- Superior long-context comprehension
- State-of-the-art multimodal understanding
- Knowledge cutoff: June 2024

Your task is to perform an ultra-precise screening with zero ambiguity. Follow the PICO-TT criteria EXACTLY.
When uncertain, analyze every nuance of the abstract against the criteria."""

        elif model_key == "o3-pro":
            # Deep reasoning prompt for O3's step-by-step analysis
            system_prompt = """You are a reasoning-focused systematic reviewer using O3-pro's advanced capabilities.

Your unique features:
- Step-by-step reasoning through complex criteria
- Access to web search for verification
- File analysis capabilities
- Python execution for calculations

Think through each PICO-TT criterion methodically:
1. First, identify all relevant elements in the abstract
2. Then, compare each element against the criteria
3. Use reasoning tokens to work through edge cases
4. Provide your final decision with complete reasoning chain"""

        elif model_key == "gemini-2.5-flash-lite":
            # Rapid screening prompt for Gemini's speed
            system_prompt = """You are a rapid-screening specialist using Gemini 2.5 Flash-Lite.

Your advantages:
- Ultra-fast processing with 1.5x speed improvement
- Cost-efficient analysis
- Excellent for high-volume classification
- Strong multilingual capabilities

Focus on quick, accurate decisions. Prioritize clear inclusion/exclusion signals.
Be decisive - this is a first-pass screen where speed matters."""

        else:
            system_prompt = "You are a systematic review expert."
        
        # Common prompt structure
        criteria_text = f"""
PICO-TT Criteria for Screening:
- Population: {criteria.population}
- Intervention: {criteria.intervention}
- Comparison: {criteria.comparison}
- Outcome: {criteria.outcome}
- Timeframe: {criteria.timeframe}
- Study Types: {criteria.study_types}

Inclusion Requirements:
- Language: {criteria.inclusion_language}
- Publication: {criteria.inclusion_publication}
- Sample Size: {criteria.inclusion_sample_size}
- Data Availability: {criteria.inclusion_data_availability}
- Other: {criteria.other_inclusion}

Exclusion Criteria:
- Study Types to Exclude: {criteria.exclusion_study_types}
- Populations to Exclude: {criteria.exclusion_populations}
- Interventions to Exclude: {criteria.exclusion_interventions}
- Languages to Exclude: {criteria.exclusion_languages}
- Other Exclusions: {criteria.other_exclusion}

Research Question: {criteria.research_question}
"""
        
        human_prompt = f"""
{criteria_text}

Citation to Screen:
Title: {{title}}
Authors: {{authors}}
Journal: {{journal}}
Year: {{year}}
Abstract: {{abstract}}
Keywords: {{keywords}}

{self.output_parser.get_format_instructions()}

Provide your screening decision as valid JSON.
"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    async def screen_with_triple_llm(
        self,
        citation: Dict[str, Any],
        criteria: ProjectCriteria
    ) -> Dict[str, Any]:
        """Screen using all three advanced LLMs for maximum accuracy"""
        
        # Prepare input data
        input_data = {
            "title": citation.get("title", ""),
            "authors": citation.get("authors", ""),
            "journal": citation.get("journal", ""),
            "year": citation.get("year", ""),
            "abstract": citation.get("abstract", ""),
            "keywords": citation.get("keywords", "")
        }
        
        # Create tasks for parallel execution
        tasks = []
        for model_key in ["gpt-4.1", "o3-pro", "gemini-2.5-flash-lite"]:
            llm = self._create_llm(model_key)
            prompt = self.create_advanced_screening_prompt(criteria, model_key)
            chain = prompt | llm | self.output_parser
            tasks.append(self._screen_with_model(chain, input_data, model_key))
        
        # Execute all three models in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        screening_results = {}
        for i, model_key in enumerate(["gpt-4.1", "o3-pro", "gemini-2.5-flash-lite"]):
            if isinstance(results[i], Exception):
                screening_results[model_key] = {
                    "error": str(results[i]),
                    "decision": "uncertain",
                    "confidence": 0
                }
            else:
                screening_results[model_key] = results[i]
        
        # Determine consensus using weighted voting
        consensus = self._calculate_advanced_consensus(screening_results)
        
        return {
            "gpt_4_1_result": screening_results.get("gpt-4.1"),
            "o3_pro_result": screening_results.get("o3-pro"),
            "gemini_flash_lite_result": screening_results.get("gemini-2.5-flash-lite"),
            "consensus": consensus,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _screen_with_model(self, chain, input_data: Dict, model_key: str):
        """Screen with a specific model"""
        start_time = datetime.utcnow()
        
        try:
            result = await chain.ainvoke(input_data)
            
            # Add model-specific metadata
            result_dict = result.dict()
            result_dict["model"] = model_key
            result_dict["model_features"] = self.models[model_key]["features"]
            result_dict["processing_time_ms"] = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return result_dict
            
        except Exception as e:
            raise Exception(f"Error with {model_key}: {str(e)}")
    
    def _calculate_advanced_consensus(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate consensus using model strengths and confidence weights"""
        
        # Weight factors based on model characteristics
        weights = {
            "gpt-4.1": 0.4,  # Highest weight for precision
            "o3-pro": 0.35,  # High weight for reasoning
            "gemini-2.5-flash-lite": 0.25  # Lower weight but fast
        }
        
        decisions = []
        weighted_confidence = 0
        total_weight = 0
        
        for model_key, result in results.items():
            if "error" not in result:
                weight = weights[model_key]
                decisions.append({
                    "model": model_key,
                    "decision": result.get("decision", "uncertain"),
                    "confidence": result.get("confidence", 0),
                    "weight": weight
                })
                weighted_confidence += result.get("confidence", 0) * weight
                total_weight += weight
        
        if not decisions:
            return {
                "final_decision": "uncertain",
                "confidence": 0,
                "agreement": "no_valid_results"
            }
        
        # Count weighted votes
        include_score = sum(d["weight"] for d in decisions if d["decision"] == "include")
        exclude_score = sum(d["weight"] for d in decisions if d["decision"] == "exclude")
        uncertain_score = sum(d["weight"] for d in decisions if d["decision"] == "uncertain")
        
        # Determine final decision
        if include_score > exclude_score and include_score > uncertain_score:
            final_decision = "include"
        elif exclude_score > include_score and exclude_score > uncertain_score:
            final_decision = "exclude"
        else:
            final_decision = "uncertain"
        
        # Calculate agreement level
        max_score = max(include_score, exclude_score, uncertain_score)
        agreement_ratio = max_score / total_weight if total_weight > 0 else 0
        
        if agreement_ratio >= 0.8:
            agreement = "strong_consensus"
        elif agreement_ratio >= 0.6:
            agreement = "moderate_consensus"
        else:
            agreement = "weak_consensus"
        
        # Check for complete agreement
        unique_decisions = set(d["decision"] for d in decisions)
        if len(unique_decisions) == 1:
            agreement = "unanimous"
        
        return {
            "final_decision": final_decision,
            "confidence": weighted_confidence / total_weight if total_weight > 0 else 0,
            "agreement": agreement,
            "voting_details": {
                "include_score": include_score,
                "exclude_score": exclude_score,
                "uncertain_score": uncertain_score
            },
            "model_decisions": decisions
        }
    
    def estimate_screening_cost(self, num_citations: int) -> Dict[str, float]:
        """Estimate the cost of screening with each model"""
        
        # Average tokens per citation (title + abstract)
        avg_input_tokens = 500
        avg_output_tokens = 200
        
        costs = {}
        for model_key, config in self.models.items():
            input_cost = (avg_input_tokens * num_citations / 1000) * config["cost_per_1k_input"]
            output_cost = (avg_output_tokens * num_citations / 1000) * config["cost_per_1k_output"]
            costs[model_key] = {
                "total_cost": input_cost + output_cost,
                "per_citation": (input_cost + output_cost) / num_citations
            }
        
        # Triple screening cost
        triple_cost = sum(model["total_cost"] for model in costs.values())
        costs["triple_screening"] = {
            "total_cost": triple_cost,
            "per_citation": triple_cost / num_citations
        }
        
        return costs