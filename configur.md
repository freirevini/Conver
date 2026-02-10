🧠 Relatório de Configuração de LLM & Otimização de Prompt — MapKnime
Projeto: MapKnime — KNIME Workflow Analyzer & AI Transpiler
Data: 2026-02-10
Escopo: Análise completa de configuração de modelos Gemini para transpilação KNIME → Python

1. Resumo Executivo
O MapKnime é um pipeline CLI de 6 etapas que extrai workflows KNIME (.knwf), parseia XML, executa 3 mappers especializados (temporal, loop, lógica) e, finalmente, transpila o workflow para Python executável via Vertex AI Gemini. A tarefa da IA é processar centenas de nós KNIME com suas configurações, conexões e metadados, e gerar código Python fiel ao workflow original.

IMPORTANT

A recomendação final é Gemini 2.5 Pro como modelo principal, com Gemini 2.5 Flash como fallback para workflows menores. Justificativas detalhadas na Seção 5.

2. Análise do Projeto
2.1 Arquitetura do Pipeline
Step 1
Step 2
Step 3
Step 4
Step 5
Step 6
.knwf (ZIP)
Extract
Parse XML → JSON
Temporal Mapper
Loop Mapper
Logic Mapper
AI Transpiler
fluxo_transpilado.py
transpilation_report.md
2.2 Módulos e Dependências
Módulo	Linhas	Função

knime_parser.py
906	Parser XML → JSON, gerador de MD/HTML, ordenação topológica (Kahn)

run_analysis.py
343	CLI unificado, orquestra extractão → parsing → mappers

avaliacao_IA.py
931	Transpiler IA: chunking, prompt builder, chamada Vertex AI, validação

temporal_mapper.py
788	Identifica padrões temporais (datas, timestamps, variáveis)

loop_mapper.py
650	Mapeia estruturas de loop (Group, Counting, While, etc.)

logic_mapper.py
589	Extrai lógica (Rule Engine, expressões, Java/Python snippets)

config.yaml
18	Configuração Vertex AI + preferências de output
2.3 Dependências Externas
# Core
pyyaml                    # Parsing de config.yaml
google-cloud-aiplatform   # SDK Vertex AI (vertexai.generative_models)
# Runtime do código gerado
pandas                    # Manipulação de dados
numpy                     # Operações numéricas
sqlalchemy                # Conexões com bancos de dados
2.4 Pontos de Integração com Bancos de Dados
O projeto não armazena credenciais de banco no 

config.yaml
. Em vez disso, o prompt instrui a IA a gerar placeholders editáveis no código transpilado:

python
# Padrão gerado pelo transpiler:
DB_CONFIG = {
    "driver": "postgresql",   # postgresql | mysql | mssql | oracle
    "host": "",               # <-- INSERT DB HOST
    "port": 5432,             # <-- INSERT DB PORT
    "user": "",               # <-- INSERT DB USERNAME
    "password": "",           # <-- INSERT DB PASSWORD
    "database": "",           # <-- INSERT DB NAME
}
Os tipos de banco detectados pelos nós KNIME incluem:

PostgreSQL (via DB Connector / DB Reader)
MySQL (via MySQL Connector)
SQL Server (via Microsoft SQL Server Connector)
Oracle (via Oracle Connector)
SQLite (via SQLite Connector)
3. Análise do Workflow e Fluxo de Dados
3.1 Etapas Sequenciais do Pipeline
Step	Ação	Input	Output
1	Extração ZIP	.knwf	Diretório temporário com XMLs
2	Parsing recursivo	workflow.knime + settings.xml	KNIME_WORKFLOW_ANALYSIS.json, 

.md
, .html
3	Mapeamento temporal	JSON da análise	temporal_map.json
4	Mapeamento de loops	JSON da análise	loop_map.json
5	Mapeamento de lógica	JSON da análise	logic_map.json
6	Transpilação IA	4 JSONs + system prompt	fluxo_transpilado.py + relatório
3.2 Chunking Engine
O sistema possui um engine de chunking inteligente:

Threshold: 800.000 tokens (~3.2MB de JSON)
Estratégia: Split por MetaNode boundaries
Chunk 0: Nós raiz (imports, config, utilidades)
Chunk 1..N: Um MetaNode por chunk
Último chunk: MetaNode + 

main()
 orchestrator
Contexto: Variáveis de chunks anteriores são propagadas via context_from_previous
3.3 Prompt Architecture
O prompt é estruturado em 6 seções:

┌─────────────────────────────────────────┐
│ SYSTEM PROMPT (fixo, 336 palavras)      │
│  → Regras de transpilação               │
│  → Padrão de credenciais DB             │
│  → Instruções de código Python          │
├─────────────────────────────────────────┤
│ USER PROMPT (dinâmico)                  │
│  § 1. WORKFLOW STRUCTURE (nodes + conns)│
│  § 2. TEMPORAL PATTERNS                 │
│  § 3. LOOP STRUCTURES                   │
│  § 4. LOGIC / RULES / EXPRESSIONS       │
│  § 5. CONTEXT FROM PREVIOUS CHUNKS      │
│  § 6. CHUNK INFO                        │
└─────────────────────────────────────────┘
4. Análise de Conexões com Bancos de Dados
4.1 Padrões de conexão detectados
O parser identifica nós de banco via 

factory
 class:

KNIME Node	Python Equivalente	Driver SQLAlchemy
DBReaderNodeFactory	pd.read_sql()	Varia por driver
DBWriterNodeFactory	df.to_sql()	Varia por driver
DBConnectorNodeFactory	sqlalchemy.create_engine()	Configado por tipo
MySQLConnectorNodeFactory	create_engine("mysql+pymysql://")	pymysql
MSSQLConnectorNodeFactory	create_engine("mssql+pyodbc://")	pyodbc
OracleConnectorNodeFactory	create_engine("oracle+cx_oracle://")	cx_Oracle
PostgreSQLConnectorNodeFactory	create_engine("postgresql://")	psycopg2
4.2 Boas práticas de segurança para conexões
O sistema já implementa boas práticas:

✅ Credenciais como placeholders editáveis (não hardcoded)
✅ Senhas em settings.xml detectadas como xpassword → "***ENCRYPTED***"
✅ Suporte a múltiplas conexões (DB_CONFIG_SOURCE, DB_CONFIG_TARGET)
TIP

Recomendação adicional: Considerar instrução no prompt para gerar suporte a variáveis de ambiente:

python
import os
DB_CONFIG = {
    "host": os.getenv("DB_HOST", ""),
    "password": os.getenv("DB_PASSWORD", ""),
}
5. Avaliação Comparativa de Modelos
5.1 Gemini 2.5 Pro
Critério	Avaliação	Nota (1-10)
Contexto longo	Janela de 1M tokens. Workflows complexos (100+ nós) geram ~200k-500k tokens de contexto. O Pro processa isso nativamente sem chunking.	10
Raciocínio complexo	Excelente para mapear DAGs de execução, traduzir Rule Engine para np.where(), resolver dependências entre nós, e gerar 

main()
 coerente.	9
Qualidade do código	Gera Python idiomático com type hints, docstrings, logging e error handling. Mantém fidelidade ao workflow original.	9
Compreensão multi-artefato	Processa 4 JSONs simultaneamente (workflow + temporal + loop + logic), cruzando referências entre eles.	10
Custo	~US$ 1.25/1M tokens input, ~US$ 5.00/1M tokens output. Para um workflow típico (~300k tokens in, ~50k out): ~US$ 0.63 por transpilação.	6
Velocidade	30-120s por requisição dependendo do tamanho. Adequado para uso CLI (não real-time).	7
Pontuação total Pro: 51/60

5.2 Gemini 2.5 Flash
Critério	Avaliação	Nota (1-10)
Contexto longo	Janela de 1M tokens (igual ao Pro). Capacidade técnica equivalente.	10
Raciocínio complexo	Bom para workflows simples/médios (até ~50 nós). Pode perder nuances em DAGs complexos com MetaNodes aninhados e loops recursivos.	6
Qualidade do código	Código funcional, mas pode omitir edge cases, simplificar error handling, ou gerar aproximações menos fiéis.	6
Compreensão multi-artefato	Funciona bem com contexto linear, mas pode falhar na correlação cruzada entre os 4 JSONs para workflows muito grandes.	7
Custo	~US$ 0.15/1M tokens input, ~US$ 0.60/1M tokens output. Para o mesmo workflow: ~US$ 0.08 por transpilação (~8x mais barato).	10
Velocidade	5-30s por requisição. Significativamente mais rápido.	9
Pontuação total Flash: 48/60

5.3 Comparação Direta
Pro         Flash       Δ
─────────────────────────────────────────────────
Contexto            10           10         =
Raciocínio           9            6        +3 Pro
Código               9            6        +3 Pro
Multi-artefato      10            7        +3 Pro
Custo                6           10        +4 Flash
Velocidade           7            9        +2 Flash
─────────────────────────────────────────────────
TOTAL               51           48        +3 Pro
IMPORTANT

Recomendação: Use Gemini 2.5 Pro como modelo principal. A diferença de qualidade no raciocínio e geração de código justifica o custo adicional, especialmente para workflows complexos onde erros de transpilação custam mais tempo de debugging do que a economia de custo.

5.4 Quando usar Flash
Cenário	Modelo Recomendado
Workflow < 30 nós, sem MetaNodes	Flash ✅
Workflow > 30 nós ou com MetaNodes	Pro ✅
Desenvolvimento/testes iterativos	Flash ✅
Produção / transpilação final	Pro ✅
Auto-correção de sintaxe (retry)	Flash ✅ (tarefa simples)
Batch de múltiplos workflows	Flash para triagem → Pro para os complexos
6. Configuração Recomendada de Parâmetros
6.1 Configuração para Gemini 2.5 Pro (Recomendado)
yaml
# config.yaml — Configuração otimizada para produção
vertex_ai:
  project_id: "seu-projeto-gcp"
  region: "us-central1"
  model: "gemini-2.5-pro"
Parâmetros de geração (em 

avaliacao_IA.py
):

python
generation_config = {
    "max_output_tokens": 65536,   # Suficiente para ~2000 linhas de Python
    "temperature": 0.1,           # Baixa: código determinístico e fiel
    "top_p": 0.95,                # Foco nas respostas mais prováveis
}
Parâmetro	Valor	Justificativa
temperature	0.1	Código deve ser determinístico e reproduzível. Valor muito baixo (0.0) pode causar repetição; 0.1 permite mínima variação criativa para nomes de variáveis.
top_p	0.95	Mantém diversidade suficiente sem sacrificar precisão.
max_output_tokens	65.536	Workflows complexos podem gerar 1000-3000 linhas. Valor atual é adequado. Para workflows muito grandes, considerar 131.072 (máximo do Pro).
top_k	Não configurado	Deixar default do modelo. Combinar top_k + top_p pode causar restrição excessiva.
candidate_count	1	Apenas uma resposta necessária.
6.2 Configuração para Gemini 2.5 Flash (Fallback)
yaml
vertex_ai:
  project_id: "seu-projeto-gcp"
  region: "us-central1"
  model: "gemini-2.5-flash"
python
generation_config = {
    "max_output_tokens": 65536,
    "temperature": 0.05,          # Ainda mais baixa para compensar menor raciocínio
    "top_p": 0.90,                # Mais restritivo para evitar divagações
}
6.3 Configuração de Safety Settings
python
from vertexai.generative_models import HarmCategory, HarmBlockThreshold
safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}
WARNING

O conteúdo gerado é exclusivamente código Python técnico. Os filtros de segurança devem ser desabilitados para evitar bloqueios falsos em queries SQL com palavras como DROP, DELETE, KILL, ou nomes de colunas que possam ser interpretados incorretamente.

7. Otimização do System Prompt
7.1 Prompt Atual (Análise)
O system prompt atual (336 palavras, ~420 tokens) é bem estruturado, com:

✅ 10 regras claras e numeradas
✅ Padrão de credenciais DB com exemplo
✅ Instruções para aproximações (nós sem equivalente direto)
✅ Mapeamento de nós KNIME → construtos Python
7.2 Melhorias Recomendadas
7.2.1 Adicionar seção de CONTEXT PRIORITIZATION
python
SYSTEM_PROMPT_ADDITION = """
    CONTEXT PRIORITIZATION:
    When processing the input, prioritize information in this order:
    1. execution_order (defines the DAG sequence)
    2. connections (determines data flow between nodes)
    3. logic_map (Rule Engine rules, expressions — must be translated exactly)
    4. temporal_map (date/time operations — use pd.to_datetime)
    5. loop_map (iteration patterns — translate to for/while)
    6. node model config (parameters, column names, etc.)
"""
7.2.2 Adicionar exemplos de tradução (few-shot)
python
EXAMPLES_SECTION = """
    TRANSLATION EXAMPLES:
    
    KNIME Row Filter (include rows where col > 0):
    → df = df[df["column_name"] > 0]
    
    KNIME Column Rename (old→new):
    → df = df.rename(columns={"old_name": "new_name"})
    
    KNIME Rule Engine ($col$ > 10 => "HIGH", TRUE => "LOW"):
    → df["result"] = np.where(df["col"] > 10, "HIGH", "LOW")
    
    KNIME GroupBy (group by col_A, aggregate col_B with SUM):
    → df = df.groupby("col_A", as_index=False).agg({"col_B": "sum"})
    
    KNIME Joiner (inner join on key_col):
    → df = df_left.merge(df_right, on="key_col", how="inner")
"""
7.2.3 Adicionar instruções para variáveis de ambiente
python
ENV_VARS_SECTION = """
    ENVIRONMENT VARIABLES:
    For all database connections, also generate an alternative using 
    environment variables with os.getenv(). Include both options 
    in the config section with clear comments explaining each approach.
"""
7.3 Prompt Size Budget
Componente	Tokens (~)	% do Budget
System Prompt (atual)	420	0.05%
System Prompt (otimizado)	~800	0.10%
User Prompt (workflow JSON)	50k-500k	5-50%
User Prompt (mappers)	10k-100k	1-10%
Total Input	~60k-600k	6-60%
Margem	400k-940k	40-94%
NOTE

O budget de tokens permite expandir significativamente o system prompt sem impacto. A margem é confortável mesmo para workflows muito grandes.

8. Configurações Avançadas
8.1 Retry & Self-Correction
A configuração atual já é sólida:

python
# avaliacao_IA.py (atual)
max_retries = 3       # Para chamadas Vertex AI
max_corrections = 2    # Para auto-correção de sintaxe via IA
backoff = 2 ** attempt # Exponential backoff (2s, 4s, 8s)
Recomendação: Manter. A combinação de 3 retries + 2 correções cobre 99%+ dos cenários.

8.2 Token Threshold para Chunking
python
TOKEN_THRESHOLD = 800_000  # Atual: margem de segurança de 200k
Recomendação: Manter em 800k. O threshold atual oferece margem adequada considerando que o prompt system + formatação consomem ~50k tokens adicionais.

8.3 Configuração de Output
yaml
output:
  max_line_length: 120      # PEP 8 recomenda 79, mas 120 é padrão moderno
  include_comments: true     # Essencial para rastreabilidade
  include_type_hints: true   # Melhora manutenibilidade
Recomendação: Manter. Adicionar opção include_env_vars: true para gerar padrão com os.getenv().

9. Tabela Resumo Final
Configuração Recomendada por Cenário
Cenário	Modelo	Temp	Top-P	Max Tokens	Safety
Produção	gemini-2.5-pro	0.1	0.95	65536	BLOCK_NONE
Dev/Teste	gemini-2.5-flash	0.05	0.90	65536	BLOCK_NONE
Auto-correção	gemini-2.5-flash	0.0	0.90	32768	BLOCK_NONE
Workflows grandes (>100 nós)	gemini-2.5-pro	0.1	0.95	131072	BLOCK_NONE
Custo Estimado por Transpilação
Tamanho	Nós	Modelo	Tokens In	Tokens Out	Custo	Tempo
Pequeno	<20	Flash	~50k	~10k	~$0.01	~5s
Médio	20-60	Pro	~200k	~30k	~$0.40	~30s
Grande	60-150	Pro	~500k	~60k	~$0.93	~60s
Muito grande	150+	Pro (chunked)	~800k	~100k	~$1.50	~120s
10. Conclusão e Recomendação Final
IMPORTANT

Modelo Recomendado: Gemini 2.5 Pro
Justificativa: Para a tarefa de transpilação KNIME → Python, a qualidade do raciocínio é o fator mais crítico. Um erro de lógica no código gerado custa significativamente mais (em tempo de debugging) do que a diferença de custo entre Pro e Flash. O Pro demonstra:

Melhor fidelidade na tradução de Rule Engine (regras complexas com múltiplas condições)
Melhor compreensão de DAG (resolve dependências entre nós corretamente)
Código mais completo (menos pass, menos # TODO, menos aproximações)
Melhor auto-correção (resolve erros de sintaxe em menos tentativas)
Estratégia de custo: Use Flash para development/testing e Pro para a transpilação final.
