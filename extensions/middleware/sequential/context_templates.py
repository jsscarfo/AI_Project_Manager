#!/usr/bin/env python
"""
Task-specific Context Templates for Sequential Thinking.

This module provides templates for different reasoning tasks,
enabling customized context selection and processing based on
the specific type of task being performed.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field, validator
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContextTemplate(BaseModel):
    """Base model for context templates."""
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    task_type: str = Field(..., description="Type of task this template is for")
    level_weights: Dict[str, float] = Field(..., description="Weights for different knowledge levels")
    step_definitions: Dict[str, Dict[str, Any]] = Field(..., description="Definitions for each step")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Customizable parameters")
    
    def get_step_config(self, step_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Step configuration dictionary
        """
        if step_name not in self.step_definitions:
            # Return defaults if step not explicitly defined
            return {
                "levels": list(self.level_weights.keys()),
                "query_count": 2,
                "context_limit": 5,
                "use_previous_step": True
            }
            
        return self.step_definitions[step_name]
    
    def get_level_weights(self, step_name: Optional[str] = None) -> Dict[str, float]:
        """
        Get weights for knowledge levels, optionally customized for a specific step.
        
        Args:
            step_name: Optional step name to get specialized weights
            
        Returns:
            Dictionary of level weights
        """
        if not step_name or step_name not in self.step_definitions:
            return self.level_weights
            
        # Check if step has custom level weights
        step_config = self.step_definitions[step_name]
        if "level_weights" in step_config:
            return step_config["level_weights"]
            
        return self.level_weights
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert template to dictionary.
        
        Returns:
            Dictionary representation of the template
        """
        return self.dict()
        
    def to_json(self, **kwargs) -> str:
        """
        Convert template to JSON.
        
        Args:
            **kwargs: Additional arguments to pass to json.dumps
            
        Returns:
            JSON representation of the template
        """
        return json.dumps(self.dict(), **kwargs)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextTemplate':
        """
        Create a template from a dictionary.
        
        Args:
            data: Dictionary representation of the template
            
        Returns:
            ContextTemplate instance
        """
        return cls(**data)
        
    @classmethod
    def from_json(cls, json_str: str) -> 'ContextTemplate':
        """
        Create a template from JSON.
        
        Args:
            json_str: JSON representation of the template
            
        Returns:
            ContextTemplate instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


# Define standard templates

# Planning template
PLANNING_TEMPLATE = ContextTemplate(
    name="planning",
    description="Template for planning tasks",
    task_type="planning",
    level_weights={
        "domain": 0.4,
        "techstack": 0.3,
        "project": 0.3
    },
    step_definitions={
        "problem_definition": {
            "levels": ["domain", "project"],
            "query_count": 3,
            "context_limit": 5,
            "use_previous_step": False,
            "level_weights": {
                "domain": 0.6,
                "project": 0.4,
                "techstack": 0.0
            }
        },
        "information_gathering": {
            "levels": ["domain", "techstack", "project"],
            "query_count": 4,
            "context_limit": 10,
            "use_previous_step": True
        },
        "analysis": {
            "levels": ["techstack", "project"],
            "query_count": 3,
            "context_limit": 8,
            "use_previous_step": True,
            "level_weights": {
                "domain": 0.2,
                "techstack": 0.5,
                "project": 0.3
            }
        },
        "solution_design": {
            "levels": ["techstack", "project"],
            "query_count": 3,
            "context_limit": 8,
            "use_previous_step": True,
            "level_weights": {
                "domain": 0.1,
                "techstack": 0.4,
                "project": 0.5
            }
        },
        "conclusion": {
            "levels": ["project"],
            "query_count": 2,
            "context_limit": 5,
            "use_previous_step": True,
            "level_weights": {
                "domain": 0.2,
                "techstack": 0.3,
                "project": 0.5
            }
        }
    }
)

# Coding template
CODING_TEMPLATE = ContextTemplate(
    name="coding",
    description="Template for code implementation tasks",
    task_type="coding",
    level_weights={
        "domain": 0.1,
        "techstack": 0.6,
        "project": 0.3
    },
    step_definitions={
        "problem_definition": {
            "levels": ["project", "techstack"],
            "query_count": 2,
            "context_limit": 5,
            "use_previous_step": False,
            "level_weights": {
                "domain": 0.1,
                "techstack": 0.4,
                "project": 0.5
            }
        },
        "information_gathering": {
            "levels": ["techstack", "project"],
            "query_count": 4,
            "context_limit": 12,
            "use_previous_step": True,
            "level_weights": {
                "domain": 0.0,
                "techstack": 0.7,
                "project": 0.3
            }
        },
        "solution_design": {
            "levels": ["techstack", "project"],
            "query_count": 3,
            "context_limit": 10,
            "use_previous_step": True,
            "level_weights": {
                "domain": 0.0,
                "techstack": 0.5,
                "project": 0.5
            }
        },
        "implementation": {
            "levels": ["techstack", "project"],
            "query_count": 3,
            "context_limit": 15,
            "use_previous_step": True,
            "level_weights": {
                "domain": 0.0,
                "techstack": 0.3,
                "project": 0.7
            }
        },
        "testing": {
            "levels": ["project"],
            "query_count": 2,
            "context_limit": 8,
            "use_previous_step": True,
            "level_weights": {
                "domain": 0.0,
                "techstack": 0.2,
                "project": 0.8
            }
        }
    }
)

# Debugging template
DEBUGGING_TEMPLATE = ContextTemplate(
    name="debugging",
    description="Template for debugging tasks",
    task_type="debugging",
    level_weights={
        "domain": 0.0,
        "techstack": 0.4,
        "project": 0.6
    },
    step_definitions={
        "problem_definition": {
            "levels": ["project"],
            "query_count": 2,
            "context_limit": 5,
            "use_previous_step": False,
            "level_weights": {
                "domain": 0.0,
                "techstack": 0.2,
                "project": 0.8
            }
        },
        "error_analysis": {
            "levels": ["techstack", "project"],
            "query_count": 3,
            "context_limit": 10,
            "use_previous_step": True,
            "level_weights": {
                "domain": 0.0,
                "techstack": 0.5,
                "project": 0.5
            }
        },
        "solution_exploration": {
            "levels": ["techstack", "project"],
            "query_count": 4,
            "context_limit": 12,
            "use_previous_step": True,
            "level_weights": {
                "domain": 0.0,
                "techstack": 0.6,
                "project": 0.4
            }
        },
        "implementation": {
            "levels": ["project"],
            "query_count": 2,
            "context_limit": 8,
            "use_previous_step": True,
            "level_weights": {
                "domain": 0.0,
                "techstack": 0.3,
                "project": 0.7
            }
        },
        "verification": {
            "levels": ["project"],
            "query_count": 2,
            "context_limit": 5,
            "use_previous_step": True,
            "level_weights": {
                "domain": 0.0,
                "techstack": 0.2,
                "project": 0.8
            }
        }
    }
)

# Research template
RESEARCH_TEMPLATE = ContextTemplate(
    name="research",
    description="Template for research tasks",
    task_type="research",
    level_weights={
        "domain": 0.7,
        "techstack": 0.2,
        "project": 0.1
    },
    step_definitions={
        "problem_definition": {
            "levels": ["domain"],
            "query_count": 3,
            "context_limit": 8,
            "use_previous_step": False,
            "level_weights": {
                "domain": 0.9,
                "techstack": 0.1,
                "project": 0.0
            }
        },
        "information_gathering": {
            "levels": ["domain", "techstack"],
            "query_count": 5,
            "context_limit": 15,
            "use_previous_step": True,
            "level_weights": {
                "domain": 0.8,
                "techstack": 0.2,
                "project": 0.0
            }
        },
        "analysis": {
            "levels": ["domain", "techstack"],
            "query_count": 4,
            "context_limit": 12,
            "use_previous_step": True,
            "level_weights": {
                "domain": 0.7,
                "techstack": 0.3,
                "project": 0.0
            }
        },
        "synthesis": {
            "levels": ["domain", "techstack", "project"],
            "query_count": 3,
            "context_limit": 10,
            "use_previous_step": True,
            "level_weights": {
                "domain": 0.6,
                "techstack": 0.3,
                "project": 0.1
            }
        },
        "conclusion": {
            "levels": ["domain", "project"],
            "query_count": 2,
            "context_limit": 5,
            "use_previous_step": True,
            "level_weights": {
                "domain": 0.7,
                "techstack": 0.1,
                "project": 0.2
            }
        }
    }
)


class TemplateManager:
    """
    Manager for context templates.
    
    This class provides methods for managing and selecting appropriate
    templates for different types of reasoning tasks.
    """
    
    def __init__(self, custom_templates: Optional[List[ContextTemplate]] = None):
        """
        Initialize the template manager.
        
        Args:
            custom_templates: Optional list of custom templates to add
        """
        # Initialize with standard templates
        self.templates = {
            "planning": PLANNING_TEMPLATE,
            "coding": CODING_TEMPLATE,
            "debugging": DEBUGGING_TEMPLATE,
            "research": RESEARCH_TEMPLATE
        }
        
        # Add custom templates if provided
        if custom_templates:
            for template in custom_templates:
                self.register_template(template)
    
    def register_template(self, template: ContextTemplate) -> None:
        """
        Register a new template.
        
        Args:
            template: Template to register
        """
        self.templates[template.name] = template
        logger.info(f"Registered template: {template.name} for task type: {template.task_type}")
    
    def get_template(self, template_name: str) -> Optional[ContextTemplate]:
        """
        Get a template by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template if found, None otherwise
        """
        return self.templates.get(template_name)
    
    def get_template_for_task(self, task_type: str) -> ContextTemplate:
        """
        Get an appropriate template for a task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            Best matching template (default to planning if no match)
        """
        # Look for exact match
        for name, template in self.templates.items():
            if template.task_type.lower() == task_type.lower():
                return template
        
        # Look for partial match
        for name, template in self.templates.items():
            if task_type.lower() in template.task_type.lower():
                return template
        
        # Default to planning template if no match
        logger.warning(f"No template found for task type: {task_type}, using planning template")
        return self.templates["planning"]
    
    def list_templates(self) -> List[Dict[str, str]]:
        """
        List all available templates.
        
        Returns:
            List of template summaries
        """
        return [
            {
                "name": template.name,
                "description": template.description,
                "task_type": template.task_type
            }
            for template in self.templates.values()
        ]
    
    def customize_template(self, 
                          base_template_name: str, 
                          new_name: str,
                          parameter_overrides: Dict[str, Any]) -> Optional[ContextTemplate]:
        """
        Create a customized version of an existing template.
        
        Args:
            base_template_name: Name of the template to customize
            new_name: Name for the customized template
            parameter_overrides: Parameter values to override
            
        Returns:
            Customized template if successful, None otherwise
        """
        base_template = self.get_template(base_template_name)
        if not base_template:
            logger.warning(f"Base template not found: {base_template_name}")
            return None
        
        # Create a copy of the base template
        template_dict = base_template.dict()
        template_dict["name"] = new_name
        template_dict["parameters"].update(parameter_overrides)
        
        # Create new template
        new_template = ContextTemplate(**template_dict)
        
        # Register the new template
        self.register_template(new_template)
        
        return new_template
    
    def evaluate_template_effectiveness(self, 
                                       template_name: str,
                                       task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of a template based on task results.
        
        Args:
            template_name: Name of the template
            task_results: List of task results using this template
            
        Returns:
            Evaluation metrics
        """
        template = self.get_template(template_name)
        if not template:
            logger.warning(f"Template not found for evaluation: {template_name}")
            return {"error": "Template not found"}
        
        if not task_results:
            return {"error": "No task results provided"}
        
        # Calculate success rate
        success_count = sum(1 for result in task_results if result.get("success", False))
        success_rate = success_count / len(task_results)
        
        # Calculate average context quality
        context_qualities = [
            result.get("context_quality", 0.0) 
            for result in task_results 
            if "context_quality" in result
        ]
        avg_context_quality = sum(context_qualities) / len(context_qualities) if context_qualities else 0.0
        
        # Calculate step effectiveness by step
        steps_data = {}
        for result in task_results:
            for step_data in result.get("steps", []):
                step_name = step_data.get("step_name")
                if not step_name:
                    continue
                    
                if step_name not in steps_data:
                    steps_data[step_name] = {
                        "count": 0,
                        "quality_sum": 0.0,
                        "time_sum": 0.0
                    }
                
                steps_data[step_name]["count"] += 1
                steps_data[step_name]["quality_sum"] += step_data.get("quality", 0.0)
                steps_data[step_name]["time_sum"] += step_data.get("time", 0.0)
        
        step_metrics = {}
        for step_name, data in steps_data.items():
            if data["count"] > 0:
                step_metrics[step_name] = {
                    "avg_quality": data["quality_sum"] / data["count"],
                    "avg_time": data["time_sum"] / data["count"]
                }
        
        return {
            "template_name": template_name,
            "task_count": len(task_results),
            "success_rate": success_rate,
            "avg_context_quality": avg_context_quality,
            "step_metrics": step_metrics
        } 