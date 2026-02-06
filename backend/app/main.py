"""
KNIME to Python Converter - FastAPI Main Application

This is the main entry point for the backend API.
Provides endpoints for:
- POST /api/upload - Upload KNIME workflow (.zip)
- GET /api/conversion/{job_id}/status - Check conversion status
- GET /api/conversion/{job_id}/download - Download generated Python code
"""
import os
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# In-memory job storage (use Redis/DB in production)
jobs: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info("Starting KNIME to Python Converter API...")
    
    # Create upload and output directories
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.output_dir, exist_ok=True)
    
    logger.info(f"Upload directory: {settings.upload_dir}")
    logger.info(f"Output directory: {settings.output_dir}")
    logger.info(f"GCP Project: {settings.google_cloud_project}")
    logger.info(f"GCP Location: {settings.google_cloud_location}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down KNIME to Python Converter API...")


# Create FastAPI application
app = FastAPI(
    title="KNIME to Python Converter API",
    description="Convert KNIME Analytics Platform workflows to executable Python code",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",  # Angular dev server
        "http://127.0.0.1:4200",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "KNIME to Python Converter API",
        "version": "1.0.0"
    }


@app.get("/api/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "gcp_project": settings.google_cloud_project,
        "gcp_location": settings.google_cloud_location,
        "active_jobs": len(jobs)
    }


@app.post("/api/upload")
async def upload_workflow(
    workflow: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload KNIME workflow (.zip or .knwf file).
    
    Returns job_id for tracking conversion progress.
    """
    # Validate file type
    filename = workflow.filename or ""
    if not (filename.endswith('.zip') or filename.endswith('.knwf')):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_file_type",
                "message": "Arquivo deve ser .zip ou .knwf",
                "suggestion": "Exporte seu workflow do KNIME no formato correto"
            }
        )
    
    # Validate file size
    contents = await workflow.read()
    file_size_mb = len(contents) / (1024 * 1024)
    
    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "file_too_large",
                "message": f"Arquivo muito grande: {file_size_mb:.1f}MB (máximo: {settings.max_file_size_mb}MB)",
                "suggestion": "Divida o workflow em partes menores"
            }
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Create job directory and save file
    job_dir = os.path.join(settings.upload_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Save with .zip extension for processing
    file_path = os.path.join(job_dir, "workflow.zip")
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Initialize job status
    jobs[job_id] = {
        "status": "uploaded",
        "progress": 0,
        "message": "Workflow carregado com sucesso",
        "error": None,
        "workflow_name": filename,
        "file_size_mb": round(file_size_mb, 2)
    }
    
    logger.info(f"Workflow uploaded: {filename} ({file_size_mb:.2f}MB) - Job ID: {job_id}")
    
    # Start background processing
    if background_tasks:
        background_tasks.add_task(process_workflow, job_id, file_path)
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Iniciando conversão do workflow"
    }


async def process_workflow(job_id: str, zip_path: str):
    """
    Process workflow in background.
    
    Steps:
    1. Extract ZIP
    2. Parse workflow.knime XML
    3. Parse settings.xml for each node (using node IDs)
    4. Build DAG (execution order)
    5. Generate Python code
    6. Generate README.md documentation
    7. Validate syntax
    8. Create ZIP package with both files
    """
    import zipfile
    
    try:
        # Update status: Processing
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 10
        jobs[job_id]["message"] = "Extraindo workflow..."
        
        logger.info(f"[{job_id}] Starting workflow processing...")
        
        # Import services (lazy import to avoid circular dependencies)
        from app.services.parser.workflow_parser import WorkflowParser
        from app.services.parser.node_parser import NodeParser
        from app.services.parser.topology_builder import TopologyBuilder
        from app.services.generator.code_generator import CodeGenerator
        from app.services.validator.python_validator import PythonValidator
        from app.utils.zip_extractor import ZipExtractor
        
        # Step 1: Extract ZIP
        extractor = ZipExtractor()
        workflow_dir = extractor.extract(zip_path)
        
        # Get original filename for documentation
        original_filename = jobs[job_id].get("workflow_name", "workflow.knwf")
        
        jobs[job_id]["progress"] = 20
        jobs[job_id]["message"] = "Analisando estrutura do workflow..."
        
        # Step 2: Parse workflow.knime
        parser = WorkflowParser()
        workflow_data = parser.parse_workflow(workflow_dir)
        
        jobs[job_id]["progress"] = 40
        jobs[job_id]["message"] = f"Encontrados {len(workflow_data['nodes'])} nós..."
        
        # Step 3: Parse node settings (using node IDs)
        node_parser = NodeParser()
        for node in workflow_data['nodes']:
            node['settings'] = node_parser.parse_node_settings(node.get('settings_path'))
            # Ensure node ID is tracked
            if 'id' not in node:
                node['id'] = node.get('node_id', 'unknown')
        
        jobs[job_id]["progress"] = 50
        jobs[job_id]["message"] = "Construindo grafo de execução..."
        
        # Step 4: Build DAG
        topology = TopologyBuilder()
        dag = topology.build_dag(workflow_data['nodes'], workflow_data['connections'])
        
        jobs[job_id]["progress"] = 60
        jobs[job_id]["message"] = "Gerando código Python..."
        
        # Step 5: Generate Python code with node ID-based function names
        generator = CodeGenerator()
        python_code = generator.generate_python_code(workflow_data, dag)
        
        jobs[job_id]["progress"] = 70
        jobs[job_id]["message"] = "Gerando documentação README.md..."
        
        # Step 6: Generate README.md documentation
        readme_content = _generate_readme(
            workflow_data, 
            dag, 
            original_filename,
            job_id
        )
        
        jobs[job_id]["progress"] = 80
        jobs[job_id]["message"] = "Validando sintaxe..."
        
        # Step 7: Validate syntax
        validator = PythonValidator()
        is_valid, errors = validator.validate_syntax(python_code)
        
        if not is_valid:
            logger.warning(f"[{job_id}] Generated code has syntax warnings: {errors}")
        
        jobs[job_id]["progress"] = 90
        jobs[job_id]["message"] = "Criando pacote de download..."
        
        # Step 8: Save results and create ZIP package
        output_dir = os.path.join(settings.output_dir, job_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Python code
        python_path = os.path.join(output_dir, "converted_workflow.py")
        with open(python_path, 'w', encoding='utf-8') as f:
            f.write(python_code)
        
        # Save README.md
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Create ZIP package
        zip_output_path = os.path.join(output_dir, "converted_workflow.zip")
        with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(python_path, "converted_workflow.py")
            zf.write(readme_path, "README.md")
        
        # Update final status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Conversão concluída com sucesso!"
        jobs[job_id]["output_path"] = python_path
        jobs[job_id]["readme_path"] = readme_path
        jobs[job_id]["zip_path"] = zip_output_path
        jobs[job_id]["node_count"] = len(workflow_data['nodes'])
        jobs[job_id]["connection_count"] = len(workflow_data.get('connections', []))
        
        logger.info(f"[{job_id}] Workflow processing completed successfully")
        
    except Exception as e:
        logger.error(f"[{job_id}] Error processing workflow: {e}", exc_info=True)
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = {
            "code": "processing_failed",
            "message": str(e),
            "suggestion": "Verifique se o arquivo é um workflow KNIME válido"
        }


def _generate_readme(
    workflow_data: Dict[str, Any], 
    dag: Any,
    original_filename: str,
    job_id: str
) -> str:
    """
    Generate README.md content for the converted workflow.
    
    Args:
        workflow_data: Parsed workflow data with nodes and connections
        dag: Directed Acyclic Graph of node execution order
        original_filename: Original KNIME file name
        job_id: Job ID for tracking
        
    Returns:
        README.md content as string
    """
    from datetime import datetime
    
    nodes = workflow_data.get('nodes', [])
    connections = workflow_data.get('connections', [])
    workflow_name = workflow_data.get('name', original_filename.replace('.knwf', '').replace('.zip', ''))
    
    # Count node categories
    category_counts = {}
    for node in nodes:
        category = node.get('category', 'Other')
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # Build node mapping table
    node_rows = []
    for node in nodes[:100]:  # Limit to 100 for readability
        node_id = node.get('id', node.get('node_id', 'unknown'))
        node_name = node.get('name', 'Unknown')
        factory = node.get('factory_class', node.get('factory', ''))
        category = node.get('category', 'Other')
        
        # Generate function name
        clean_name = node_name.lower().replace(" ", "_").replace("-", "_")
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
        func_name = f"node_{node_id}_{clean_name}"
        
        node_rows.append(f"| #{node_id} | {node_name} | `{func_name}()` | {category} |")
    
    # Collect dependencies based on node types
    dependencies = {"pandas", "numpy", "logging", "pathlib"}
    for node in nodes:
        factory = node.get('factory_class', node.get('factory', '')).lower()
        if 'database' in factory or 'db' in factory:
            dependencies.add("sqlalchemy")
        if 'bigquery' in factory:
            dependencies.add("google-cloud-bigquery")
        if 'datetime' in factory or 'date' in factory:
            dependencies.add("datetime")
        if 'json' in factory:
            dependencies.add("json")
    
    # Count metanodes
    metanode_count = sum(1 for n in nodes if n.get('is_metanode', False))
    
    # Calculate max depth
    max_depth = max((n.get('depth', 0) for n in nodes), default=0)
    
    readme = f"""# {workflow_name}

> Código Python gerado automaticamente a partir de workflow KNIME

---

## 📋 Informações do Workflow

| Propriedade | Valor |
|------------|-------|
| **Arquivo Original** | `{original_filename}` |
| **Data da Conversão** | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
| **Versão KNIME Compatível** | 4.5.2+ |
| **Total de Nodes** | {len(nodes)} |
| **Metanodes** | {metanode_count} |
| **Conexões** | {len(connections)} |
| **Profundidade Máxima** | {max_depth} níveis |

---

## 📊 Distribuição por Categoria

| Categoria | Quantidade |
|-----------|------------|
"""
    
    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        readme += f"| {category} | {count} |\n"
    
    readme += f"""

---

## 📦 Dependências

```
{chr(10).join(sorted(dependencies))}
```

Instale com:
```bash
pip install {' '.join(sorted(dependencies))}
```

---

## 🗺️ Mapeamento de Nodes (KNIME → Python)

A tabela abaixo mostra a correspondência entre os nodes KNIME originais (identificados por **Node ID**) e as funções Python geradas.

> **Nota**: O identificador único (Node ID) é usado em vez do nome exibido, pois nodes copiados podem ter o mesmo nome mas IDs distintos.

| Node ID | Nome Original | Função Python | Categoria |
|---------|---------------|---------------|-----------|
"""
    
    readme += "\n".join(node_rows)
    
    if len(nodes) > 100:
        readme += f"\n\n*... e mais {len(nodes) - 100} nodes*\n"
    
    readme += f"""

---

## 🚀 Como Usar

### Pré-requisitos

1. Python 3.10 ou superior
2. Instale as dependências:

```bash
pip install {' '.join(sorted(dependencies))}
```

### Execução

```bash
python converted_workflow.py
```

### Configuração

Edite as seguintes variáveis no início do script conforme necessário:

```python
# Configurações de entrada/saída
INPUT_PATH = "caminho/para/dados/entrada"
OUTPUT_PATH = "caminho/para/dados/saida"

# Configurações de banco de dados (se aplicável)
DB_CONNECTION_STRING = "postgresql://user:pass@host:port/database"
```

---

## 📁 Arquivos Gerados

| Arquivo | Descrição |
|---------|-----------|
| `converted_workflow.py` | Script Python principal executável |
| `README.md` | Esta documentação |

---

## 🔧 Estrutura do Código Gerado

O código Python gerado segue uma estrutura modular e organizada:

1. **Imports** - Bibliotecas necessárias (pandas, numpy, etc.)
2. **Configurações** - Variáveis configuráveis para caminhos e conexões
3. **Funções por Node** - Cada node KNIME é convertido em uma função Python
   - Nomeadas como `node_<ID>_<nome>()` para garantir unicidade
   - Documentadas com docstrings indicando o node KNIME original
4. **Execução Principal** - Orquestragem seguindo a ordem topológica do DAG original

### Exemplo de Função Gerada

```python
def node_1648_column_filter(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    Column Filter (Node ID: #1648)
    
    Filtra colunas do DataFrame baseado na configuração KNIME original.
    Converte: org.knime.base.node.preproc.filter.column.DataColumnSpecFilterNodeFactory
    \"\"\"
    columns_to_keep = ["col1", "col2", "col3"]
    return df[columns_to_keep]
```

---

## ⚠️ Considerações Importantes

1. **Nodes não suportados**: Alguns nodes podem gerar stubs que requerem implementação manual
2. **Conexões de banco**: Verifique e atualize as credenciais de conexão
3. **Caminhos de arquivo**: Ajuste os caminhos de entrada/saída conforme seu ambiente
4. **Testes**: Execute testes com dados de amostra antes de usar em produção

---

*Gerado pelo KNIME to Python Converter - Job ID: {job_id}*
*Data: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    return readme




@app.get("/api/conversion/{job_id}/status")
async def get_job_status(job_id: str):
    """Get conversion job status."""
    if job_id not in jobs:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "job_not_found",
                "message": f"Job não encontrado: {job_id}",
                "suggestion": "Verifique o ID do job ou faça um novo upload"
            }
        )
    
    return jobs[job_id]


@app.get("/api/conversion/{job_id}/download")
async def download_result(job_id: str):
    """Download generated Python code."""
    if job_id not in jobs:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "job_not_found",
                "message": f"Job não encontrado: {job_id}"
            }
        )
    
    job = jobs[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail={
                "error": "job_not_completed",
                "message": f"Job ainda não concluído. Status atual: {job['status']}",
                "suggestion": "Aguarde a conclusão do processamento"
            }
        )
    
    output_path = job.get('output_path')
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(
            status_code=500,
            detail={
                "error": "file_not_found",
                "message": "Arquivo de saída não encontrado",
                "suggestion": "Tente fazer o upload novamente"
            }
        )
    
    return FileResponse(
        output_path,
        media_type='text/x-python',
        filename='converted_workflow.py'
    )


@app.get("/api/conversion/{job_id}/download-package")
async def download_package(job_id: str):
    """
    Download complete package with Python code and README.md.
    
    Returns a ZIP file containing:
    - converted_workflow.py: The generated Python script
    - README.md: Documentation with node mapping, dependencies, and usage
    """
    if job_id not in jobs:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "job_not_found",
                "message": f"Job não encontrado: {job_id}"
            }
        )
    
    job = jobs[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail={
                "error": "job_not_completed",
                "message": f"Job ainda não concluído. Status atual: {job['status']}",
                "suggestion": "Aguarde a conclusão do processamento"
            }
        )
    
    zip_path = job.get('zip_path')
    if not zip_path or not os.path.exists(zip_path):
        raise HTTPException(
            status_code=500,
            detail={
                "error": "package_not_found",
                "message": "Pacote ZIP não encontrado",
                "suggestion": "Tente fazer o upload novamente"
            }
        )
    
    # Use workflow name for the download filename
    workflow_name = job.get('workflow_name', 'workflow').replace('.knwf', '').replace('.zip', '')
    download_filename = f"{workflow_name}_python.zip"
    
    return FileResponse(
        zip_path,
        media_type='application/zip',
        filename=download_filename
    )


@app.get("/api/conversion/{job_id}/download-readme")
async def download_readme(job_id: str):
    """Download only the README.md documentation."""
    if job_id not in jobs:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "job_not_found",
                "message": f"Job não encontrado: {job_id}"
            }
        )
    
    job = jobs[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail={
                "error": "job_not_completed",
                "message": f"Job ainda não concluído. Status atual: {job['status']}",
                "suggestion": "Aguarde a conclusão do processamento"
            }
        )
    
    readme_path = job.get('readme_path')
    if not readme_path or not os.path.exists(readme_path):
        raise HTTPException(
            status_code=500,
            detail={
                "error": "readme_not_found",
                "message": "README.md não encontrado",
                "suggestion": "Tente fazer o upload novamente"
            }
        )
    
    return FileResponse(
        readme_path,
        media_type='text/markdown',
        filename='README.md'
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )

