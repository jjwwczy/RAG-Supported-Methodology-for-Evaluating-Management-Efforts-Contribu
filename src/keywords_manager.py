"""
Keywords Manager for AutoRAG project.

This module handles loading and managing city and dimension keywords
from configuration files or defaults.
"""

import os
import logging
import json
from typing import List, Dict, Optional, Any, Tuple

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KeywordsManager:
    """Manager for city and dimension keywords."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the KeywordsManager with configuration.
        
        Args:
            config: The configuration dictionary containing keywords settings
        """
        self.config = config
        self.keywords_config = config.get('keywords', {})
        self.cities = []
        self.dimensions = []
        self.templates = {}
        
        # Load cities and dimensions
        self._load_cities()
        self._load_dimensions()
        self._load_templates()
        
    def _load_cities(self) -> None:
        """Load cities from file or use defaults."""
        if not self.keywords_config.get('enabled', False):
            logging.info("Keywords functionality is disabled in config.")
            return
            
        cities_file = self.keywords_config.get('cities_file')
        default_cities = self.keywords_config.get('default_cities', [])
        
        if cities_file and os.path.exists(cities_file):
            try:
                with open(cities_file, 'r', encoding='utf-8') as f:
                    # Read cities from file, one per line, strip whitespace
                    self.cities = [line.strip() for line in f.readlines() if line.strip()]
                logging.info(f"Successfully loaded {len(self.cities)} cities from {cities_file}")
            except Exception as e:
                logging.error(f"Failed to load cities from {cities_file}: {e}")
                self.cities = default_cities
                logging.info(f"Using {len(self.cities)} default cities instead")
        else:
            self.cities = default_cities
            if cities_file:
                logging.warning(f"Cities file {cities_file} not found, using {len(self.cities)} default cities")
            else:
                logging.info(f"No cities file specified, using {len(self.cities)} default cities")
    
    def _load_dimensions(self) -> None:
        """Load dimensions from file or use defaults."""
        if not self.keywords_config.get('enabled', False):
            return
            
        dimensions_file = self.keywords_config.get('dimensions_file')
        default_dimensions = self.keywords_config.get('default_dimensions', [])
        
        if dimensions_file and os.path.exists(dimensions_file):
            try:
                with open(dimensions_file, 'r', encoding='utf-8') as f:
                    # Read dimensions from file, one per line, strip whitespace
                    self.dimensions = [line.strip() for line in f.readlines() if line.strip()]
                logging.info(f"Successfully loaded {len(self.dimensions)} dimensions from {dimensions_file}")
            except Exception as e:
                logging.error(f"Failed to load dimensions from {dimensions_file}: {e}")
                self.dimensions = default_dimensions
                logging.info(f"Using {len(self.dimensions)} default dimensions instead")
        else:
            self.dimensions = default_dimensions
            if dimensions_file:
                logging.warning(f"Dimensions file {dimensions_file} not found, using {len(self.dimensions)} default dimensions")
            else:
                logging.info(f"No dimensions file specified, using {len(self.dimensions)} default dimensions")
    
    def _load_templates(self) -> None:
        """Load templates from file."""
        policy_config = self.config.get('policy_evaluation', {})
        if not policy_config.get('enabled', False):
            return
            
        template_file = policy_config.get('template_file')
        
        if template_file and os.path.exists(template_file):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    self.templates = json.load(f)
                logging.info(f"Successfully loaded templates from {template_file}")
            except Exception as e:
                logging.error(f"Failed to load templates from {template_file}: {e}")
                self.templates = {}
        else:
            self.templates = {}
            if template_file:
                logging.warning(f"Template file {template_file} not found, using default templates")
            else:
                logging.info("No template file specified, using default templates")
    
    def get_cities(self) -> List[str]:
        """Get the list of cities."""
        return self.cities
    
    def get_dimensions(self) -> List[str]:
        """Get the list of dimensions."""
        return self.dimensions
    
    def get_template(self, city: str, dimension: Optional[str] = None) -> Optional[str]:
        """
        Get a template for a specific city and dimension.
        
        Args:
            city: The city name
            dimension: The dimension keyword (optional)
            
        Returns:
            The template string if found, None otherwise
        """
        # First try city+dimension specific template
        if dimension:
            template_key = f"{city}_{dimension}"
            if template_key in self.templates:
                return self.templates[template_key]
        
        # Fall back to city-only template
        if city in self.templates:
            return self.templates[city]
        
        # Fall back to dimension-only template
        if dimension and dimension in self.templates:
            return self.templates[dimension]
        
        # Fall back to default template
        if "default" in self.templates:
            return self.templates["default"]
        
        return None
    
    def detect_keywords_in_query(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect city and dimension keywords in a query.
        
        Args:
            query: The query string
            
        Returns:
            A tuple of (city, dimension) if found, (None, None) otherwise
        """
        detected_city = None
        detected_dimension = None
        
        # Check for cities
        for city in self.cities:
            if city in query:
                detected_city = city
                break
        
        # Check for dimensions
        for dimension in self.dimensions:
            if dimension in query:
                detected_dimension = dimension
                break
        
        return detected_city, detected_dimension
    
    def create_default_template(self, city: Optional[str] = None, dimension: Optional[str] = None) -> str:
        """
        Create a default template for a city and dimension.
        
        Args:
            city: The city name (optional)
            dimension: The dimension keyword (optional)
            
        Returns:
            A default template string
        """
        context_placeholder = "{context}"
        city_part = f"关于{city}的" if city else ""
        dimension_part = f"{dimension}相关的" if dimension else ""
        
        template = f"""
请根据以下上下文信息，提取{city_part}{dimension_part}政策相关内容，并按照JSON格式返回。

上下文信息：
{context_placeholder}

请确保返回的JSON格式正确，并且只包含上下文中明确提到的信息。如果某些字段在上下文中没有提及，请将其设置为null或空列表。
"""
        return template
    
    def get_or_create_template(self, city: Optional[str] = None, dimension: Optional[str] = None) -> str:
        """
        Get an existing template or create a default one.
        
        Args:
            city: The city name (optional)
            dimension: The dimension keyword (optional)
            
        Returns:
            A template string
        """
        template = None
        if city or dimension:
            template = self.get_template(city or "", dimension)
        
        if not template:
            template = self.create_default_template(city, dimension)
            
        return template


# Helper function to create a KeywordsManager instance
def create_keywords_manager(config: Dict[str, Any]) -> KeywordsManager:
    """
    Create a KeywordsManager instance from configuration.
    
    Args:
        config: The configuration dictionary
        
    Returns:
        A KeywordsManager instance
    """
    return KeywordsManager(config)
