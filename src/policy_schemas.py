"""
Policy schemas and evaluation functions for the AutoRAG project.

This module defines Pydantic models for structured policy data extraction
and scoring functions to evaluate city policy implementation.
"""

import logging
from typing import List, Dict, Optional, Union, Any, Literal
from pydantic import BaseModel, Field, field_validator

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Base Schema Models ---

class PolicyMeasure(BaseModel):
    """Base model for a policy measure or initiative."""
    title: str = Field(..., description="Title or name of the policy measure")
    description: str = Field(..., description="Brief description of the policy measure")
    timeframe: Optional[str] = Field(None, description="Implementation timeframe (e.g., '2020-2025')")
    source_document: Optional[str] = Field(None, description="Source document name")
    source_page: Optional[str] = Field(None, description="Page number or section in source document")

class TargetGoal(BaseModel):
    """Model for quantitative or qualitative targets in policies."""
    metric: str = Field(..., description="The metric or indicator being targeted")
    target_value: Union[str, float, int] = Field(..., description="Target value to achieve")
    baseline_value: Optional[Union[str, float, int]] = Field(None, description="Baseline value, if available")
    target_year: Optional[int] = Field(None, description="Year by which the target should be achieved")
    is_binding: Optional[bool] = Field(None, description="Whether the target is legally binding")

# --- City-Specific Schema Models ---

class BeijingLowCarbonPolicy(BaseModel):
    """Schema for Beijing's low-carbon policies."""
    city: Literal["北京"] = Field("北京", description="City name")
    policy_name: str = Field(..., description="Name of the low-carbon policy")
    implementation_period: str = Field(..., description="Implementation period of the policy")
    key_measures: List[PolicyMeasure] = Field(..., description="Key measures in the policy")
    carbon_reduction_targets: List[TargetGoal] = Field(..., description="Carbon reduction targets")
    energy_efficiency_measures: List[PolicyMeasure] = Field(..., description="Energy efficiency measures")
    renewable_energy_initiatives: List[PolicyMeasure] = Field(..., description="Renewable energy initiatives")
    
    # 不再需要验证器，因为使用了 Literal 类型

class HarbinLowCarbonPolicy(BaseModel):
    """Schema for Harbin's low-carbon policies."""
    city: Literal["哈尔滨"] = Field("哈尔滨", description="City name")
    policy_name: str = Field(..., description="Name of the low-carbon policy")
    implementation_period: str = Field(..., description="Implementation period of the policy")
    key_measures: List[PolicyMeasure] = Field(..., description="Key measures in the policy")
    environmental_protection_goals: List[TargetGoal] = Field(..., description="Environmental protection goals")
    industrial_transformation_measures: List[PolicyMeasure] = Field(..., description="Industrial transformation measures")
    ecological_development_initiatives: List[PolicyMeasure] = Field(..., description="Ecological development initiatives")
    
    # 不再需要验证器，因为使用了 Literal 类型

# --- Schema and Template Mapping ---

# Map city names to their corresponding schema classes
CITY_SCHEMA_MAP = {
    "北京": BeijingLowCarbonPolicy,
    "哈尔滨": HarbinLowCarbonPolicy,
    # Add more cities and their schemas as needed
}

# Map dimension keywords to their focus areas (for evaluation)
DIMENSION_FOCUS_MAP = {
    "低碳": "carbon_reduction",
    "绿色": "green_development",
    "能源": "energy",
    "生态": "ecology",
    "环保": "environmental_protection",
    "科技": "technology",
    "创新": "innovation",
    "发展": "development",
    # Add more dimensions as needed
}

# The template mapping is now handled by the KeywordsManager class in keywords_manager.py

# --- Policy Evaluation Functions ---

def score_policy_completeness(policy_data: Dict[str, Any], city: str) -> float:
    """
    Score the completeness of a policy based on how many fields are filled.
    
    Args:
        policy_data: The extracted policy data as a dictionary
        city: The city name to determine which schema to use
        
    Returns:
        A score between 0.0 and 1.0 representing completeness
    """
    if city not in CITY_SCHEMA_MAP:
        logging.error(f"No schema defined for city: {city}")
        return 0.0
    
    # Get the appropriate schema class
    schema_class = CITY_SCHEMA_MAP[city]
    
    try:
        # Count total fields in the schema
        schema_fields = len(schema_class.__fields__)
        
        # Count non-empty fields in the provided data
        non_empty_fields = 0
        for field_name, field_value in policy_data.items():
            if field_value is not None and field_value != [] and field_value != "":
                non_empty_fields += 1
                
                # For list fields, check if they have content
                if isinstance(field_value, list) and len(field_value) > 0:
                    # Add points for each non-empty item in lists
                    for item in field_value:
                        if isinstance(item, dict):
                            non_empty_subfields = sum(1 for v in item.values() if v is not None and v != "")
                            non_empty_fields += non_empty_subfields / len(item)
        
        # Calculate completeness score (normalized to 0.0-1.0)
        completeness_score = min(1.0, non_empty_fields / (schema_fields * 2))
        logging.info(f"Policy completeness score for {city}: {completeness_score:.2f}")
        return completeness_score
        
    except Exception as e:
        logging.error(f"Error scoring policy completeness: {e}")
        return 0.0

def score_policy_ambition(policy_data: Dict[str, Any], city: str) -> float:
    """
    Score the ambition level of a policy based on targets and measures.
    
    Args:
        policy_data: The extracted policy data as a dictionary
        city: The city name to determine which scoring criteria to use
        
    Returns:
        A score between 0.0 and 1.0 representing ambition level
    """
    if city not in CITY_SCHEMA_MAP:
        logging.error(f"No scoring criteria defined for city: {city}")
        return 0.0
    
    try:
        ambition_score = 0.0
        
        # Score based on city-specific criteria
        if city == "北京":
            # Get carbon reduction targets
            targets = policy_data.get("carbon_reduction_targets", [])
            if targets:
                # More targets = higher score (up to 0.5)
                ambition_score += min(0.5, len(targets) * 0.1)
                
                # Binding targets get extra points
                binding_targets = sum(1 for t in targets if t.get("is_binding", False))
                ambition_score += min(0.2, binding_targets * 0.05)
            
            # Count total measures
            measures_count = len(policy_data.get("key_measures", [])) + \
                             len(policy_data.get("energy_efficiency_measures", [])) + \
                             len(policy_data.get("renewable_energy_initiatives", []))
            
            # More measures = higher score (up to 0.3)
            ambition_score += min(0.3, measures_count * 0.02)
            
        elif city == "哈尔滨":
            # Get environmental protection goals
            goals = policy_data.get("environmental_protection_goals", [])
            if goals:
                # More goals = higher score (up to 0.4)
                ambition_score += min(0.4, len(goals) * 0.08)
                
                # Binding goals get extra points
                binding_goals = sum(1 for g in goals if g.get("is_binding", False))
                ambition_score += min(0.2, binding_goals * 0.05)
            
            # Count total measures
            measures_count = len(policy_data.get("key_measures", [])) + \
                             len(policy_data.get("industrial_transformation_measures", [])) + \
                             len(policy_data.get("ecological_development_initiatives", []))
            
            # More measures = higher score (up to 0.4)
            ambition_score += min(0.4, measures_count * 0.03)
        
        logging.info(f"Policy ambition score for {city}: {ambition_score:.2f}")
        return ambition_score
        
    except Exception as e:
        logging.error(f"Error scoring policy ambition: {e}")
        return 0.0

def evaluate_city_policy(policy_data: Dict[str, Any], city: str, dimension: Optional[str] = None) -> Dict[str, float]:
    """
    Evaluate a city's policy implementation based on multiple criteria.
    
    Args:
        policy_data: The extracted policy data as a dictionary
        city: The city name
        dimension: Optional dimension keyword to focus evaluation
        
    Returns:
        A dictionary with scores for different evaluation criteria
    """
    scores = {
        "completeness": score_policy_completeness(policy_data, city),
        "ambition": score_policy_ambition(policy_data, city),
        "feasibility": score_policy_feasibility(policy_data, city) if 'score_policy_feasibility' in globals() else 0.5,
        # Add more scoring dimensions as needed
    }
    
    # Apply dimension-specific weighting if a dimension is specified
    if dimension and dimension in DIMENSION_FOCUS_MAP:
        focus_area = DIMENSION_FOCUS_MAP[dimension]
        
        # Adjust weights based on dimension focus
        if focus_area == "carbon_reduction":
            weights = {"completeness": 0.3, "ambition": 0.5, "feasibility": 0.2}
        elif focus_area in ["green_development", "ecology", "environmental_protection"]:
            weights = {"completeness": 0.3, "ambition": 0.4, "feasibility": 0.3}
        elif focus_area in ["energy", "technology"]:
            weights = {"completeness": 0.2, "ambition": 0.3, "feasibility": 0.5}
        elif focus_area in ["innovation", "development"]:
            weights = {"completeness": 0.3, "ambition": 0.3, "feasibility": 0.4}
        else:
            weights = {"completeness": 0.4, "ambition": 0.6, "feasibility": 0.0}
            
        logging.info(f"Using dimension-specific weights for {dimension} (focus: {focus_area})")
    else:
        # Default weights
        weights = {"completeness": 0.4, "ambition": 0.6, "feasibility": 0.0}
    
    # Calculate overall score (weighted average)
    overall_score = sum(score * weights.get(criterion, 0) for criterion, score in scores.items())
    scores["overall"] = overall_score
    scores["weights_used"] = weights  # Include the weights used in the evaluation
    
    # Round scores for readability
    for key in scores:
        if isinstance(scores[key], float):
            scores[key] = round(scores[key], 2)
    
    return scores

# --- Helper Functions ---

def get_city_schema(city: str) -> Optional[type]:
    """Get the appropriate schema class for a city."""
    return CITY_SCHEMA_MAP.get(city)

def get_city_prompt_template(city: str) -> Optional[str]:
    """Get the appropriate prompt template for a city.
    
    Note: This function is deprecated and will be removed in a future version.
    Use the KeywordsManager.get_or_create_template() method instead.
    """
    logging.warning("get_city_prompt_template() is deprecated. Use KeywordsManager.get_or_create_template() instead.")
    # For backward compatibility only
    from .keywords_manager import create_keywords_manager
    import yaml
    
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        keywords_manager = create_keywords_manager(config)
        return keywords_manager.get_or_create_template(city)
    except Exception as e:
        logging.error(f"Error in get_city_prompt_template: {e}")
        return None
