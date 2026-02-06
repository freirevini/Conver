"""
KNIME Node Behavior Catalog

Provides contextual information about KNIME node behaviors to enhance
LLM code generation. Each node category has specific patterns and 
expected behaviors that the LLM should understand.
"""

from typing import Dict, Optional


# =============================================================================
# Node Behavior Definitions
# =============================================================================

NODE_BEHAVIORS: Dict[str, Dict] = {
    # =========================================================================
    # DATE/TIME NODES
    # =========================================================================
    "CreateDateTimeNodeFactory": {
        "category": "DateTime",
        "description": "Cria intervalos de datas/horas baseado em configurações",
        "behavior": """
COMPORTAMENTO KNIME:
- Gera uma tabela com uma coluna de datas/horas
- Parâmetros: data início, data fim, granularidade (dia, mês, ano)
- Pode usar "now" como referência para data atual

PADRÃO PYTHON:
```python
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

# Criar range de datas
start_date = datetime.now().replace(day=1)  # Primeiro dia do mês
end_date = datetime.now()
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
df_output = pd.DataFrame({'date': date_range})
```
""",
        "imports": ["datetime", "dateutil.relativedelta", "pandas"],
    },
    
    "DateTimeShiftNodeFactory": {
        "category": "DateTime",
        "description": "Desloca datas por período especificado",
        "behavior": """
COMPORTAMENTO KNIME:
- Adiciona/subtrai período de uma coluna de data
- Parâmetros: coluna, valor do shift, unidade (dias, meses, anos)

PADRÃO PYTHON:
```python
from dateutil.relativedelta import relativedelta

# Shift por meses (exemplo: -1 mês)
df_output = df_input.copy()
df_output['date_shifted'] = df_input['date_column'].apply(
    lambda x: x - relativedelta(months=1)
)
```
""",
        "imports": ["dateutil.relativedelta"],
    },
    
    "ExtractDateTimeFieldsNodeFactory": {
        "category": "DateTime", 
        "description": "Extrai componentes de data (ano, mês, dia, etc)",
        "behavior": """
COMPORTAMENTO KNIME:
- Extrai campos numéricos de uma coluna datetime
- Campos possíveis: year, month, day, hour, minute, second, weekday

PADRÃO PYTHON:
```python
df_output = df_input.copy()
df_output['year'] = df_input['date_column'].dt.year
df_output['month'] = df_input['date_column'].dt.month
df_output['day'] = df_input['date_column'].dt.day
```
""",
        "imports": [],
    },
    
    "DurationPeriodFormatNodeFactory": {
        "category": "DateTime",
        "description": "Calcula diferença entre duas datas",
        "behavior": """
COMPORTAMENTO KNIME:
- Calcula diferença entre duas colunas de data
- Resultado pode ser em dias, meses, anos ou duração

PADRÃO PYTHON:
```python
df_output = df_input.copy()
# Diferença em dias
df_output['diff_days'] = (df_input['end_date'] - df_input['start_date']).dt.days
# Diferença em meses (aproximado)
df_output['diff_months'] = df_output['diff_days'] / 30.44
```
""",
        "imports": [],
    },

    # =========================================================================
    # LOOP NODES
    # =========================================================================
    "LoopStartParNodeFactory": {
        "category": "Loop",
        "description": "Inicia loop com parametrização",
        "behavior": """
COMPORTAMENTO KNIME:
- Itera sobre cada linha da tabela de entrada
- Cada iteração processa uma linha como variáveis de fluxo
- Loop body recebe variáveis para usar em queries, cálculos, etc.

PADRÃO PYTHON:
```python
results = []
for idx, row in df_input.iterrows():
    # Variáveis da iteração atual
    current_value = row['column_name']
    
    # Processar (body do loop)
    result = process_row(row)
    results.append(result)

df_output = pd.concat(results, ignore_index=True)
```
""",
        "imports": ["pandas"],
    },
    
    "GroupLoopStartNodeFactory": {
        "category": "Loop",
        "description": "Itera sobre grupos de dados",
        "behavior": """
COMPORTAMENTO KNIME:
- Agrupa dados por coluna(s) especificada(s)
- Itera sobre cada grupo separadamente
- Útil para processamento por categoria

PADRÃO PYTHON:
```python
results = []
for group_key, group_df in df_input.groupby('group_column'):
    # Processar cada grupo
    processed = process_group(group_df)
    results.append(processed)

df_output = pd.concat(results, ignore_index=True)
```
""",
        "imports": ["pandas"],
    },
    
    "LoopEndNodeFactory": {
        "category": "Loop",
        "description": "Finaliza loop e concatena resultados",
        "behavior": """
COMPORTAMENTO KNIME:
- Coleta resultados de todas as iterações
- Concatena em uma única tabela
- Opções: append rows, unique concatenate

PADRÃO PYTHON:
```python
# Assumindo _loop_results contém resultados de cada iteração
df_output = pd.concat(_loop_results, ignore_index=True)
```
""",
        "imports": ["pandas"],
    },
    
    # =========================================================================
    # RULE ENGINE NODES
    # =========================================================================
    "RuleEngineNodeFactory": {
        "category": "Transform",
        "description": "Aplica regras condicionais para criar/modificar colunas",
        "behavior": """
COMPORTAMENTO KNIME:
- Avalia regras na ordem definida (primeira que match ganha)
- Sintaxe: $column$ > 10 => "HIGH"
- TRUE => valor_default (última regra)
- Cria nova coluna ou substitui existente

PADRÃO PYTHON:
```python
import numpy as np

df_output = df_input.copy()

# Criar condições e valores correspondentes
conditions = [
    df_input['value'] > 100,
    df_input['value'] > 50,
    df_input['value'] > 0,
]
choices = ['HIGH', 'MEDIUM', 'LOW']
default = 'UNKNOWN'

df_output['result'] = np.select(conditions, choices, default=default)
```
""",
        "imports": ["numpy"],
    },
    
    # =========================================================================
    # STRING MANIPULATION
    # =========================================================================
    "StringManipulationNodeFactory": {
        "category": "TextProcessing",
        "description": "Manipulação de strings com funções KNIME",
        "behavior": """
COMPORTAMENTO KNIME:
- Funções: substr(), upperCase(), lowerCase(), trim(), replace()
- join(): concatena strings
- regexReplace(): substituição com regex
- $column$: referência a coluna

PADRÃO PYTHON (mapeamento de funções):
```python
df_output = df_input.copy()

# substr(string, start, length)
df_output['result'] = df_input['col'].str[start:start+length]

# upperCase/lowerCase
df_output['result'] = df_input['col'].str.upper()

# trim
df_output['result'] = df_input['col'].str.strip()

# replace
df_output['result'] = df_input['col'].str.replace('old', 'new')

# join (concatenar colunas)
df_output['result'] = df_input['col1'] + ' - ' + df_input['col2']

# regexReplace
df_output['result'] = df_input['col'].str.replace(r'pattern', 'new', regex=True)
```
""",
        "imports": [],
    },
    
    # =========================================================================
    # MATH FORMULA (JEP)
    # =========================================================================
    "JEPNodeFactory": {
        "category": "Math",
        "description": "Fórmulas matemáticas com sintaxe JEP",
        "behavior": """
COMPORTAMENTO KNIME:
- Sintaxe JEP: $column$ para referência a colunas
- Operadores: +, -, *, /, ^, %
- Funções: abs(), sqrt(), round(), floor(), ceil(), log(), exp()
- Constantes: PI, E
- Condicionais: if(condition, then, else)
- Missing values: $column$ == null → isna()

PADRÃO PYTHON (mapeamento):
```python
import numpy as np

df_output = df_input.copy()

# Operações básicas
df_output['result'] = df_input['a'] + df_input['b']
df_output['result'] = df_input['a'] * 100 / df_input['b'].replace(0, np.nan)

# Funções
df_output['result'] = np.abs(df_input['col'])
df_output['result'] = np.sqrt(df_input['col'])
df_output['result'] = np.round(df_input['col'], 2)

# Condicionais (if)
df_output['result'] = np.where(df_input['a'] > 10, 'HIGH', 'LOW')
```
""",
        "imports": ["numpy"],
    },
    
    # =========================================================================
    # AGGREGATION
    # =========================================================================
    "ColumnAggregatorNodeFactory": {
        "category": "Aggregation",
        "description": "Agrega múltiplas colunas em uma única coluna",
        "behavior": """
COMPORTAMENTO KNIME:
- Combina valores de múltiplas colunas em uma lista ou concatenação
- Tipos de agregação: List, Set, Concatenate, Sum, Mean

PADRÃO PYTHON:
```python
df_output = df_input.copy()

# Agregar colunas em lista
columns_to_aggregate = ['col1', 'col2', 'col3']
df_output['aggregated'] = df_input[columns_to_aggregate].values.tolist()

# Ou concatenar como string
df_output['concatenated'] = df_input[columns_to_aggregate].astype(str).agg(' | '.join, axis=1)
```
""",
        "imports": [],
    },
    
    "GroupByNodeFactory": {
        "category": "Aggregation",
        "description": "Agrupa e calcula estatísticas",
        "behavior": """
COMPORTAMENTO KNIME:
- Agrupa por coluna(s) especificada(s)
- Aplica funções de agregação: SUM, MEAN, COUNT, MIN, MAX, FIRST, LAST
- Pode renomear colunas resultantes

PADRÃO PYTHON:
```python
df_output = df_input.groupby(['group_col']).agg({
    'value_col': 'sum',
    'count_col': 'count',
    'avg_col': 'mean'
}).reset_index()

# Renomear colunas se necessário
df_output.columns = ['group', 'total', 'count', 'average']
```
""",
        "imports": [],
    },
    
    # =========================================================================
    # DATABASE NODES
    # =========================================================================
    "DBLoopingNodeFactory": {
        "category": "Database",
        "description": "Executa query SQL para cada linha de entrada usando variáveis",
        "behavior": """
COMPORTAMENTO KNIME:
- Para cada linha da tabela de entrada, executa query SQL
- Variáveis de fluxo substituem placeholders na query
- Coleta resultados de todas as queries

PADRÃO PYTHON:
```python
import sqlalchemy
import pandas as pd

engine = sqlalchemy.create_engine(connection_string)
results = []

for idx, row in df_input.iterrows():
    # Substituir variáveis na query
    query = query_template.format(**row.to_dict())
    result = pd.read_sql(query, engine)
    results.append(result)

df_output = pd.concat(results, ignore_index=True)
```
""",
        "imports": ["sqlalchemy", "pandas"],
    },
    
    # =========================================================================
    # FLOW CONTROL
    # =========================================================================
    "EmptyTableSwitchNodeFactory": {
        "category": "FlowControl",
        "description": "Direciona fluxo baseado se tabela está vazia ou não",
        "behavior": """
COMPORTAMENTO KNIME:
- Porta 1: Tabela se NÃO vazia
- Porta 2: Tabela se vazia
- Usado para tratamento de casos especiais

PADRÃO PYTHON:
```python
if df_input.empty:
    df_output_empty = df_input
    df_output_nonempty = pd.DataFrame()
else:
    df_output_empty = pd.DataFrame()
    df_output_nonempty = df_input
```
""",
        "imports": ["pandas"],
    },
    
    "TableRowToVariableNodeFactory": {
        "category": "FlowControl",
        "description": "Converte primeira linha da tabela em variáveis de fluxo",
        "behavior": """
COMPORTAMENTO KNIME:
- Pega a primeira (ou todas) linhas da tabela
- Converte cada coluna em uma variável de fluxo
- Variáveis ficam disponíveis para nodes seguintes

PADRÃO PYTHON:
```python
# Extrair primeira linha como dicionário de variáveis
if not df_input.empty:
    flow_vars = df_input.iloc[0].to_dict()
    # Usar variáveis: flow_vars['column_name']
else:
    flow_vars = {}

df_output = df_input  # Passa tabela adiante
```
""",
        "imports": [],
    },
}


def get_node_behavior(factory: str) -> Optional[Dict]:
    """
    Get behavior documentation for a KNIME node factory.
    
    Args:
        factory: Full or partial factory class name
        
    Returns:
        Behavior dict or None if not found
    """
    # Try exact match
    simple_name = factory.split('.')[-1]
    if simple_name in NODE_BEHAVIORS:
        return NODE_BEHAVIORS[simple_name]
    
    # Try partial match
    for key in NODE_BEHAVIORS:
        if key.lower() in factory.lower():
            return NODE_BEHAVIORS[key]
    
    return None


def get_behavior_context(factory: str) -> str:
    """
    Get behavior context string for LLM prompt enrichment.
    
    Args:
        factory: Node factory class name
        
    Returns:
        Formatted context string for LLM prompt
    """
    behavior = get_node_behavior(factory)
    
    if not behavior:
        return ""
    
    context = f"""
## Node Behavior Reference
Category: {behavior['category']}
Description: {behavior['description']}

{behavior['behavior']}

Required imports: {', '.join(behavior['imports']) if behavior['imports'] else 'pandas (standard)'}
"""
    return context


def get_category_behaviors(category: str) -> Dict[str, Dict]:
    """
    Get all behaviors for a specific category.
    
    Args:
        category: Category name (DateTime, Loop, etc.)
        
    Returns:
        Dict of behaviors in that category
    """
    return {
        key: val for key, val in NODE_BEHAVIORS.items()
        if val['category'] == category
    }
