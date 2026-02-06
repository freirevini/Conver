"""
API Documentation and OpenAPI Schema Enhancement.

Provides enhanced API docs for the KNIME to Python Converter.
"""
from typing import Dict, Any

# API Metadata
API_TITLE = "KNIME to Python Converter API"
API_DESCRIPTION = """
## 🔄 KNIME Workflow to Python Code Converter

Converte workflows do KNIME Analytics Platform para código Python executável.

### Funcionalidades

* **Upload de Workflow** - Suporta `.knwf` e `.zip`
* **Conversão Automática** - Gera código Python idiomático
* **Documentação** - README.md com mapeamento de nodes
* **Validação** - Sintaxe Python verificada automaticamente

### Fluxo de Uso

1. `POST /api/upload` - Envie o arquivo do workflow
2. `GET /api/conversion/{job_id}/status` - Acompanhe o progresso
3. `GET /api/conversion/{job_id}/download-package` - Baixe o resultado

### Limites

| Recurso | Limite |
|---------|--------|
| Tamanho do arquivo | 100 MB |
| Requisições por minuto | 60 |
| Jobs simultâneos | 10 |

### Códigos de Status

| Código | Significado |
|--------|-------------|
| 200 | Sucesso |
| 400 | Erro de validação |
| 404 | Job não encontrado |
| 429 | Rate limit excedido |
| 500 | Erro interno |
"""

API_VERSION = "1.0.0"

API_CONTACT = {
    "name": "ChatKnime Team",
    "email": "support@chatknime.io"
}

API_LICENSE = {
    "name": "MIT",
    "url": "https://opensource.org/licenses/MIT"
}

# OpenAPI Tags
API_TAGS = [
    {
        "name": "Health",
        "description": "Endpoints de verificação de saúde da API"
    },
    {
        "name": "Workflow",
        "description": "Upload e conversão de workflows KNIME"
    },
    {
        "name": "Download",
        "description": "Download de arquivos gerados"
    },
    {
        "name": "Metrics",
        "description": "Métricas e observabilidade"
    }
]

# Response Examples
UPLOAD_RESPONSE_EXAMPLE = {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "processing",
    "message": "Iniciando conversão do workflow"
}

STATUS_PROCESSING_EXAMPLE = {
    "status": "processing",
    "progress": 60,
    "message": "Gerando código Python...",
    "workflow_name": "meu_workflow.knwf",
    "file_size_mb": 12.5
}

STATUS_COMPLETED_EXAMPLE = {
    "status": "completed",
    "progress": 100,
    "message": "Conversão concluída com sucesso!",
    "workflow_name": "meu_workflow.knwf",
    "node_count": 25,
    "connection_count": 30
}

ERROR_RESPONSE_EXAMPLE = {
    "error": "invalid_file_type",
    "message": "Arquivo deve ser .zip ou .knwf",
    "suggestion": "Exporte seu workflow do KNIME no formato correto"
}


def get_openapi_config() -> Dict[str, Any]:
    """Get OpenAPI configuration for FastAPI."""
    return {
        "title": API_TITLE,
        "description": API_DESCRIPTION,
        "version": API_VERSION,
        "contact": API_CONTACT,
        "license_info": API_LICENSE,
        "openapi_tags": API_TAGS
    }


# Response Models Description
RESPONSE_MODELS = {
    "UploadResponse": {
        "description": "Resposta do upload de workflow",
        "example": UPLOAD_RESPONSE_EXAMPLE
    },
    "StatusResponse": {
        "description": "Status atual do processamento",
        "examples": {
            "processing": STATUS_PROCESSING_EXAMPLE,
            "completed": STATUS_COMPLETED_EXAMPLE
        }
    },
    "ErrorResponse": {
        "description": "Resposta de erro padronizada",
        "example": ERROR_RESPONSE_EXAMPLE
    }
}
