"""
Google Vertex AI Gemini Client

Provides LLM-based code generation for unsupported KNIME nodes.
Uses the Gemini model with structured prompts for Python code generation.
"""
import os
import logging
from typing import Dict, Optional, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Client for Google Vertex AI Gemini model.
    
    Used as fallback for KNIME nodes not covered by templates.
    Generates Python code using structured prompts.
    """
    
    def __init__(self):
        """Initialize Gemini client with GCP credentials."""
        self.client = None
        # MANDATORY: gemini-2.5-pro only (see .agent/rules/00-core/04-llm-config.md)
        self.model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')
        self.project = os.getenv('GOOGLE_CLOUD_PROJECT')
        self.location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Gemini client."""
        try:
            from google import genai
            
            self.client = genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location
            )
            
            logger.info(f"Gemini client initialized: model={self.model_name}")
            
        except ImportError as e:
            logger.error(f"Failed to import google-genai: {e}")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.client = None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=16)
    )
    def generate_node_code(
        self,
        node_type: str,
        node_config: Dict[str, Any],
        input_vars: List[str],
        output_var: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate Python code for a KNIME node using Gemini.
        
        Args:
            node_type: KNIME node factory class
            node_config: Node configuration/settings
            input_vars: List of input variable names
            output_var: Output variable name
            context: Additional context about the workflow
            
        Returns:
            Dictionary with:
            - code: Generated Python code
            - imports: Required imports
            - explanation: Brief explanation
        """
        if self.client is None:
            logger.warning("Gemini client not initialized, returning passthrough code")
            return self._fallback_response(input_vars, output_var)
        
        prompt = self._build_prompt(
            node_type=node_type,
            node_config=node_config,
            input_vars=input_vars,
            output_var=output_var,
            context=context
        )
        
        try:
            from google.genai import types
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.2,  # Low temperature for deterministic code
                    top_p=0.8,
                    max_output_tokens=2048,
                )
            )
            
            return self._parse_response(response.text, input_vars, output_var)
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise
    
    def _build_prompt(
        self,
        node_type: str,
        node_config: Dict,
        input_vars: List[str],
        output_var: str,
        context: Optional[str]
    ) -> str:
        """Build structured prompt for code generation."""
        
        # Extract simple node name from factory class
        simple_name = node_type.split('.')[-1].replace('NodeFactory', '').replace('Factory', '')
        
        prompt = f"""You are a Python code generator converting KNIME Analytics Platform nodes to Python/Pandas code.

## Task
Generate Python code equivalent to the KNIME node: **{simple_name}**

## KNIME Node Details
- **Factory Class**: `{node_type}`
- **Configuration**: 
```json
{self._format_config(node_config)}
```

## Input/Output Variables
- **Input DataFrames**: {', '.join(input_vars) if input_vars else 'None (this is a source node)'}
- **Output Variable**: `{output_var}`

## Requirements
1. Use pandas for DataFrame operations
2. Use sklearn for ML operations if needed
3. Use numpy for numerical operations
4. Generate clean, production-ready code
5. Include necessary comments

## Output Format
Respond with ONLY the following JSON format:
```json
{{
    "imports": ["import pandas as pd", ...],
    "code": "# Your generated code here",
    "explanation": "Brief explanation of what the code does"
}}
```

{f"## Additional Context: {context}" if context else ""}

Generate the Python code now:"""
        
        return prompt
    
    def _format_config(self, config: Dict, max_depth: int = 3) -> str:
        """Format configuration dict as readable string."""
        import json
        
        def truncate(obj, depth=0):
            if depth >= max_depth:
                return "..."
            if isinstance(obj, dict):
                return {k: truncate(v, depth+1) for k, v in list(obj.items())[:10]}
            elif isinstance(obj, list):
                return [truncate(v, depth+1) for v in obj[:5]]
            elif isinstance(obj, str) and len(obj) > 100:
                return obj[:100] + "..."
            return obj
        
        try:
            return json.dumps(truncate(config), indent=2)
        except:
            return str(config)[:500]
    
    def _parse_response(
        self,
        response_text: str,
        input_vars: List[str],
        output_var: str
    ) -> Dict[str, Any]:
        """Parse Gemini response to extract code and imports."""
        import json
        import re
        
        result = {
            'code': '',
            'imports': ['import pandas as pd'],
            'explanation': ''
        }
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                result['code'] = parsed.get('code', '')
                result['imports'] = parsed.get('imports', ['import pandas as pd'])
                result['explanation'] = parsed.get('explanation', '')
                return result
            except json.JSONDecodeError:
                pass
        
        # Extract code blocks if JSON parsing fails
        code_match = re.search(r'```python\n([\s\S]*?)\n```', response_text)
        if code_match:
            result['code'] = code_match.group(1)
        else:
            # Use any code-like content
            result['code'] = response_text.strip()
        
        # Extract imports from code
        import_matches = re.findall(r'^(?:from|import)\s+.+$', result['code'], re.MULTILINE)
        if import_matches:
            result['imports'] = import_matches
        
        return result
    
    def _fallback_response(self, input_vars: List[str], output_var: str) -> Dict[str, Any]:
        """Return passthrough code when Gemini is unavailable."""
        input_var = input_vars[0] if input_vars else 'df_input'
        
        return {
            'code': f"# TODO: Implement node logic\n{output_var} = {input_var}.copy()",
            'imports': ['import pandas as pd'],
            'explanation': 'Passthrough (Gemini unavailable)'
        }
    
    def is_available(self) -> bool:
        """Check if Gemini client is initialized and working."""
        return self.client is not None
    
    def test_connection(self) -> bool:
        """Test connection to Gemini API."""
        if self.client is None:
            return False
        
        try:
            from google.genai import types
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=["Say 'hello' in one word"],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=10,
                )
            )
            
            return len(response.text) > 0
            
        except Exception as e:
            logger.error(f"Gemini connection test failed: {e}")
            return False
