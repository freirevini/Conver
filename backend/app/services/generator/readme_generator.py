"""
README Generator - Generate documentation for transpiled KNIME workflows.

Creates a comprehensive README.md documenting:
- Workflow metadata and structure
- Node ID to Python function mapping
- Dependencies and usage instructions
- Known limitations and fallback nodes
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from app.models.ir_models import (
    GeneratedCode,
    NodeInstance,
    WorkflowIR,
    FallbackLevel,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkflowMetadata:
    """Metadata about the original KNIME workflow."""
    name: str
    source_file: str
    conversion_date: str
    knime_version: Optional[str] = None
    node_count: int = 0
    metanode_count: int = 0
    connection_count: int = 0
    max_nesting_depth: int = 0


@dataclass
class NodeMapping:
    """Mapping from KNIME node to Python function."""
    node_id: str
    node_name: str
    function_name: str
    category: str
    factory_class: str
    fallback_level: FallbackLevel = FallbackLevel.DETERMINISTIC


@dataclass 
class ReadmeContent:
    """Complete README content structure."""
    metadata: WorkflowMetadata
    node_mappings: List[NodeMapping] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    usage_instructions: str = ""


class ReadmeGenerator:
    """
    Generates comprehensive README.md documentation for transpiled workflows.
    
    Usage:
        generator = ReadmeGenerator()
        readme_md = generator.generate(ir, generated_code, source_file="workflow.knwf")
    """
    
    def __init__(self):
        self.template = self._get_template()
    
    def generate(
        self, 
        ir: WorkflowIR, 
        generated_code: GeneratedCode,
        source_file: str = "workflow.knwf"
    ) -> str:
        """
        Generate README.md content from workflow IR and generated code.
        
        Args:
            ir: WorkflowIR containing the workflow structure
            generated_code: GeneratedCode with function mappings
            source_file: Original KNIME file name
            
        Returns:
            Complete README.md content as string
        """
        # Build metadata
        metadata = self._extract_metadata(ir, source_file)
        
        # Build node mappings
        node_mappings = self._extract_node_mappings(ir.nodes, generated_code)
        
        # Extract dependencies
        dependencies = self._extract_dependencies(generated_code)
        
        # Collect warnings
        warnings = self._collect_warnings(ir.nodes, generated_code)
        
        # Build content structure
        content = ReadmeContent(
            metadata=metadata,
            node_mappings=node_mappings,
            dependencies=dependencies,
            warnings=warnings,
            usage_instructions=self._get_usage_instructions()
        )
        
        # Render README
        return self._render_readme(content)
    
    def _extract_metadata(self, ir: WorkflowIR, source_file: str) -> WorkflowMetadata:
        """Extract workflow metadata from IR."""
        # Count metanodes
        metanode_count = sum(
            1 for node in ir.nodes 
            if hasattr(node, 'is_metanode') and node.is_metanode
        )
        
        # Calculate max depth
        max_depth = max(
            (node.depth for node in ir.nodes if hasattr(node, 'depth')),
            default=0
        )
        
        return WorkflowMetadata(
            name=ir.workflow_name,
            source_file=source_file,
            conversion_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            knime_version="4.5.2+",
            node_count=len(ir.nodes),
            metanode_count=metanode_count,
            connection_count=len(ir.connections),
            max_nesting_depth=max_depth
        )
    
    def _extract_node_mappings(
        self, 
        nodes: List[NodeInstance],
        generated_code: GeneratedCode
    ) -> List[NodeMapping]:
        """Create node ID to function name mappings."""
        mappings = []
        
        for node in nodes:
            # Get function name - use the ID-based naming
            func_name = self._get_function_name(node)
            
            # Determine fallback level
            fallback = FallbackLevel.DETERMINISTIC
            if hasattr(generated_code, 'fallback_nodes'):
                for fb in generated_code.fallback_nodes:
                    if fb.get('node_id') == node.id:
                        fallback = FallbackLevel[fb.get('level', 'stub').upper()]
                        break
            
            mappings.append(NodeMapping(
                node_id=node.id,
                node_name=node.name,
                function_name=func_name,
                category=node.category.value if hasattr(node.category, 'value') else str(node.category),
                factory_class=node.factory_class,
                fallback_level=fallback
            ))
        
        return mappings
    
    def _get_function_name(self, node: NodeInstance) -> str:
        """Generate Python function name from node."""
        # Clean name for Python identifier
        clean_name = node.name.lower()
        clean_name = clean_name.replace(" ", "_")
        clean_name = clean_name.replace("-", "_")
        clean_name = clean_name.replace("(", "")
        clean_name = clean_name.replace(")", "")
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
        
        # Prefix with node ID for uniqueness
        return f"node_{node.id}_{clean_name}"
    
    def _extract_dependencies(self, generated_code: GeneratedCode) -> List[str]:
        """Extract Python dependencies from generated code."""
        dependencies = set()
        
        # Check imports in generated code
        code = generated_code.code if hasattr(generated_code, 'code') else ""
        
        # Common KNIME → Python dependencies
        dependency_patterns = {
            "pandas": ["import pandas", "from pandas"],
            "numpy": ["import numpy", "from numpy"],
            "datetime": ["import datetime", "from datetime"],
            "json": ["import json"],
            "logging": ["import logging"],
            "pathlib": ["from pathlib"],
            "typing": ["from typing"],
            "concurrent.futures": ["from concurrent.futures", "ThreadPoolExecutor"],
            "re": ["import re", "re."],
            "sqlalchemy": ["from sqlalchemy", "import sqlalchemy"],
            "google.cloud.bigquery": ["bigquery"],
        }
        
        for dep, patterns in dependency_patterns.items():
            for pattern in patterns:
                if pattern in code:
                    dependencies.add(dep)
                    break
        
        # Always require pandas for KNIME → Python
        dependencies.add("pandas")
        
        return sorted(list(dependencies))
    
    def _collect_warnings(
        self, 
        nodes: List[NodeInstance],
        generated_code: GeneratedCode
    ) -> List[str]:
        """Collect warnings about conversion limitations."""
        warnings = []
        
        # Check for fallback nodes
        stub_nodes = []
        llm_nodes = []
        
        if hasattr(generated_code, 'fallback_nodes'):
            for fb in generated_code.fallback_nodes:
                level = fb.get('level', 'stub')
                node_name = fb.get('node_name', 'Unknown')
                if level == 'stub':
                    stub_nodes.append(node_name)
                elif level in ['llm', 'gemini']:
                    llm_nodes.append(node_name)
        
        if stub_nodes:
            warnings.append(
                f"⚠️ {len(stub_nodes)} node(s) geraram stubs (implementação manual necessária): "
                f"{', '.join(stub_nodes[:5])}" + 
                (f" e mais {len(stub_nodes) - 5}..." if len(stub_nodes) > 5 else "")
            )
        
        if llm_nodes:
            warnings.append(
                f"🤖 {len(llm_nodes)} node(s) foram convertidos via LLM (requer revisão): "
                f"{', '.join(llm_nodes[:5])}" +
                (f" e mais {len(llm_nodes) - 5}..." if len(llm_nodes) > 5 else "")
            )
        
        # Check for database nodes
        db_nodes = [n for n in nodes if 'database' in n.factory_class.lower()]
        if db_nodes:
            warnings.append(
                f"🗄️ {len(db_nodes)} node(s) de banco de dados detectados - "
                "verifique as credenciais e strings de conexão."
            )
        
        # Check for legacy nodes
        legacy_nodes = [n for n in nodes if 'legacy' in n.factory_class.lower()]
        if legacy_nodes:
            warnings.append(
                f"📦 {len(legacy_nodes)} node(s) legados detectados - "
                "considere atualizar no KNIME original."
            )
        
        return warnings
    
    def _get_usage_instructions(self) -> str:
        """Get standard usage instructions."""
        return """
## Como Usar

### Pré-requisitos

1. Python 3.10 ou superior
2. Instale as dependências:

```bash
pip install pandas numpy sqlalchemy
```

### Execução

```bash
python converted_workflow.py
```

### Configuração

Edite as variáveis de configuração no início do script:
- `INPUT_PATH`: Caminho para os dados de entrada
- `OUTPUT_PATH`: Caminho para salvar resultados
- `DB_CONNECTION_STRING`: String de conexão se aplicável
"""
    
    def _render_readme(self, content: ReadmeContent) -> str:
        """Render the complete README.md."""
        
        # Header
        readme = f"""# {content.metadata.name}

> Código Python gerado automaticamente a partir de workflow KNIME

---

## 📋 Informações do Workflow

| Propriedade | Valor |
|------------|-------|
| **Arquivo Original** | `{content.metadata.source_file}` |
| **Data da Conversão** | {content.metadata.conversion_date} |
| **Versão KNIME** | {content.metadata.knime_version} |
| **Total de Nodes** | {content.metadata.node_count} |
| **Metanodes** | {content.metadata.metanode_count} |
| **Conexões** | {content.metadata.connection_count} |
| **Profundidade Máxima** | {content.metadata.max_nesting_depth} níveis |

---

## 📦 Dependências

```
{chr(10).join(content.dependencies)}
```

Instale com:
```bash
pip install {' '.join(content.dependencies)}
```

---

"""
        
        # Warnings
        if content.warnings:
            readme += "## ⚠️ Avisos Importantes\n\n"
            for warning in content.warnings:
                readme += f"- {warning}\n"
            readme += "\n---\n\n"
        
        # Node Mapping Table
        readme += """## 🗺️ Mapeamento de Nodes

A tabela abaixo mostra a correspondência entre os nodes KNIME originais e as funções Python geradas.

| Node ID | Nome Original | Função Python | Categoria | Fallback |
|---------|---------------|---------------|-----------|----------|
"""
        
        for mapping in content.node_mappings[:50]:  # Limit to 50 for readability
            fallback_icon = {
                FallbackLevel.DETERMINISTIC: "✅",
                FallbackLevel.LLM: "🤖",
                FallbackLevel.STUB: "⚠️"
            }.get(mapping.fallback_level, "❓")
            
            readme += f"| #{mapping.node_id} | {mapping.node_name} | `{mapping.function_name}()` | {mapping.category} | {fallback_icon} |\n"
        
        if len(content.node_mappings) > 50:
            readme += f"\n*... e mais {len(content.node_mappings) - 50} nodes*\n"
        
        readme += "\n**Legenda Fallback:**\n"
        readme += "- ✅ Determinístico (template)\n"
        readme += "- 🤖 Gerado via LLM (requer revisão)\n"
        readme += "- ⚠️ Stub (implementação manual necessária)\n"
        
        readme += "\n---\n"
        
        # Usage Instructions
        readme += content.usage_instructions
        
        # Footer
        readme += f"""
---

## 📁 Arquivos Gerados

- `converted_workflow.py` - Script Python principal
- `README.md` - Esta documentação

---

## 🔧 Estrutura do Código

O código gerado segue uma estrutura modular:

1. **Imports** - Bibliotecas necessárias
2. **Configurações** - Variáveis configuráveis
3. **Funções por Node** - Cada node KNIME é uma função Python
4. **Execução Principal** - Orquestragem na ordem topológica

---

*Gerado pelo KNIME to Python Converter*
*Data: {content.metadata.conversion_date}*
"""
        
        return readme
    
    def _get_template(self) -> str:
        """Get the base README template."""
        return "README template loaded"
    
    def save(self, readme_content: str, output_path: Path | str) -> None:
        """
        Save README.md to file.
        
        Args:
            readme_content: Complete README content
            output_path: Path to save the README.md
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"README.md saved to {output_path}")
