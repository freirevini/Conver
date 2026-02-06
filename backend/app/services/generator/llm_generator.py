"""
LLM Generator - Generate KNIME Node Code via Vertex AI

Uses Gemini 2.5 Pro to generate Python code for KNIME nodes
that don't have templates.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Load .env from backend directory
def _load_env():
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent.parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded .env from {env_path}")
        else:
            # Try backend directory
            env_path = Path(__file__).parent.parent.parent / '.env'
            if env_path.exists():
                load_dotenv(env_path)
                logger.info(f"Loaded .env from {env_path}")
    except ImportError:
        logger.debug("python-dotenv not installed, using system env vars")

_load_env()


class LLMGenerator:
    """
    LLM-based code generator for KNIME nodes.
    
    Uses Google Vertex AI (Gemini 2.5 Pro) as per rules 04-llm-config.md.
    """
    
    MODEL_ID = os.environ.get('LLM_MODEL_ID', "gemini-2.5-pro")
    
    def __init__(self):
        """Initialize LLM Generator with Vertex AI client."""
        self.client = None
        self.is_initialized = False
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Vertex AI client."""
        try:
            from google import genai
            
            project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
            location = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
            
            if not project_id:
                logger.warning("GOOGLE_CLOUD_PROJECT not set. LLM generation disabled.")
                return
            
            logger.info(f"Connecting to Vertex AI: project={project_id}, location={location}")
            
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location
            )
            self.is_initialized = True
            logger.info(f"LLM Generator initialized with {self.MODEL_ID}")
            
        except ImportError:
            logger.warning("google-genai not installed. LLM generation disabled.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
    
    def is_available(self) -> bool:
        """Check if LLM is available for generation."""
        return self.is_initialized and self.client is not None
    
    def generate_node_code(
        self,
        node_type: str,
        node_name: str,
        factory: str,
        settings: Dict[str, Any],
        input_var: str,
        output_var: str
    ) -> Optional[str]:
        """
        Generate Python code for a KNIME node.
        
        Args:
            node_type: KNIME node type name
            node_name: Human-readable node name
            factory: KNIME factory class
            settings: Node configuration settings
            input_var: Input DataFrame variable name
            output_var: Output DataFrame variable name
            
        Returns:
            Generated Python code or None on failure
        """
        if not self.is_available():
            logger.warning("LLM not available for code generation")
            return None
        
        prompt = self._build_generation_prompt(
            node_type=node_type,
            node_name=node_name,
            factory=factory,
            settings=settings,
            input_var=input_var,
            output_var=output_var
        )
        
        try:
            from google.genai import types
            
            response = self.client.models.generate_content(
                model=self.MODEL_ID,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,  # Deterministic for code
                    top_p=0.95,
                    max_output_tokens=2048,
                )
            )
            
            code = self._extract_code(response.text)
            
            if code:
                # Validate syntax
                import ast
                ast.parse(code)
                logger.info(f"LLM generated valid code for {node_name}")
                return code
            else:
                logger.warning(f"LLM returned empty code for {node_name}")
                return None
                
        except SyntaxError as e:
            logger.warning(f"LLM generated invalid syntax for {node_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM generation failed for {node_name}: {e}")
            return None
    
    def _build_generation_prompt(
        self,
        node_type: str,
        node_name: str,
        factory: str,
        settings: Dict[str, Any],
        input_var: str,
        output_var: str
    ) -> str:
        """Build structured prompt for code generation with behavior context."""
        
        simple_name = factory.split('.')[-1].replace('NodeFactory', '')
        
        # Get behavior context from catalog
        behavior_context = self._get_behavior_context(factory)
        
        # Convert settings to readable format
        settings_str = "\n".join([
            f"  - {k}: {v}" 
            for k, v in settings.items() 
            if v and k not in ['_internal', 'model']
        ][:15])  # Increased limit to 15 settings
        
        prompt = f"""You are a Python code generator for KNIME workflow transpilation.

## Task
Generate Python/Pandas code that replicates the behavior of this KNIME node.

## Node Information
- Name: {node_name}
- Type: {node_type}
- Factory: {simple_name}

## Settings
{settings_str if settings_str else "  (No specific settings)"}

## Variables
- Input DataFrame: `{input_var}`
- Output DataFrame: `{output_var}`
"""
        
        # Add behavior context if available
        if behavior_context:
            prompt += f"""
{behavior_context}
"""
        
        prompt += """
## Requirements
1. Use pandas and numpy for data operations
2. Input is a pandas DataFrame
3. Output must be a pandas DataFrame
4. Handle empty DataFrames gracefully
5. Code must be syntactically valid Python
6. Include brief comments explaining the logic
7. Follow the PADRÃO PYTHON section if provided above

## Output
Return ONLY valid Python code, no explanations:

```python
# Your code here
```"""
        return prompt
    
    def _get_behavior_context(self, factory: str) -> str:
        """Get behavior context from catalog."""
        try:
            from .node_behavior_catalog import get_behavior_context
            return get_behavior_context(factory)
        except ImportError:
            logger.debug("node_behavior_catalog not available")
            return ""
        except Exception as e:
            logger.debug(f"Error getting behavior context: {e}")
            return ""
    
    def _extract_code(self, response_text: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        if not response_text:
            return None
        
        # Try to extract from code block
        code_match = re.search(r'```python\n([\s\S]*?)\n```', response_text)
        if code_match:
            return code_match.group(1).strip()
        
        # Try generic code block
        code_match = re.search(r'```\n([\s\S]*?)\n```', response_text)
        if code_match:
            return code_match.group(1).strip()
        
        # Return response without backticks
        cleaned = response_text.strip()
        if cleaned.startswith('```'):
            lines = cleaned.split('\n')
            cleaned = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
        
        return cleaned if cleaned else None
