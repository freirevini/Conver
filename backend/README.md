# 🔄 ChatKnime — Transpilador KNIME para Python

## O que é isso?

O **ChatKnime Backend** é uma ferramenta que **converte automaticamente** workflows do KNIME Analytics Platform em código Python puro.

Imagine que você tem um fluxo de dados criado no KNIME (um arquivo `.knwf`) e precisa que ele rode em Python — sem precisar do KNIME instalado. Este programa faz essa conversão para você.

```
📥 Arquivo KNIME (.knwf)  →  🔄 ChatKnime  →  📤 Código Python (.py)
```

### Para que serve?

| Cenário | Descrição |
|---------|-----------|
| 🏢 **Automação** | Rodar workflows KNIME como scripts Python em servidores |
| 📊 **Migração** | Converter processos KNIME existentes para Python |
| 🔍 **Análise SQL** | Extrair todas as queries SQL embutidas dentro de um workflow |
| 🤖 **IA Assistida** | Nós desconhecidos são traduzidos automaticamente por Inteligência Artificial |

> [!TIP]
> Você **não** precisa saber Python para usar esta ferramenta. Basta seguir este guia passo a passo e ela fará o trabalho por você.

---

## Pré-requisitos

Antes de começar, você precisa ter dois softwares instalados no seu computador:

### 1. Python (versão 3.10 ou superior)

Python é a linguagem de programação que roda a ferramenta. Pense nele como o "motor" que faz tudo funcionar.

**Verificando se já está instalado:**

Abra o terminal do seu computador e digite:

```bash
python --version
```

Você deve ver algo como `Python 3.12.4`. Se o número for **3.10** ou maior, está tudo certo.

> [!NOTE]
> **O que é um terminal?**
>
> - **Windows:** Pressione `Win + R`, digite `cmd` e pressione Enter. Ou procure por "Prompt de Comando" no menu Iniciar.
> - **macOS:** Procure por "Terminal" no Spotlight (Cmd + Espaço).
> - **Linux:** Procure por "Terminal" nos seus aplicativos.

**Se o Python NÃO estiver instalado:**

1. Acesse [python.org/downloads](https://www.python.org/downloads/) e clique no botão amarelo de download.
2. Execute o instalador.

> [!CAUTION]
> **Windows:** Durante a instalação, **marque a caixa** "Add Python to PATH" na primeira tela do instalador. Sem isso, os comandos não funcionarão.

1. Após instalar, feche e reabra o terminal, depois verifique novamente com `python --version`.

### 2. pip (gerenciador de pacotes)

O `pip` vem instalado automaticamente com o Python. Para confirmar:

```bash
pip --version
```

Você verá algo como `pip 24.0 from ...`. Se aparecer uma versão, está tudo certo.

### 3. Git (opcional)

Apenas necessário se quiser baixar o projeto usando `git clone`. Caso contrário, basta baixar o ZIP do repositório.

---

## Instalação

Siga **todos** os passos abaixo na ordem. Cada passo depende do anterior.

### Passo 1 — Baixar o projeto

**Opção A:** Se você tem o Git instalado:

```bash
git clone https://github.com/freirevini/Conver.git
```

**Opção B:** Sem Git — baixe o arquivo ZIP do repositório e descompacte em uma pasta da sua preferência.

### Passo 2 — Navegar até a pasta do backend

Abra o terminal e navegue até a pasta `backend2`:

**Windows:**

```cmd
cd C:\caminho\para\ChatKnime\backend2
```

**macOS / Linux:**

```bash
cd /caminho/para/ChatKnime/backend2
```

> [!TIP]
> **Dica prática:** No Windows, abra a pasta `backend2` no Explorador de Arquivos, clique na barra de endereço, digite `cmd` e pressione Enter. O terminal abrirá diretamente naquela pasta.

### Passo 3 — Criar um ambiente virtual

O ambiente virtual é como uma "caixa isolada" que mantém as dependências do projeto separadas de outros programas no seu computador.

**Windows:**

```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

Após ativar, o nome `(venv)` aparecerá no início da linha do terminal. Isso confirma que o ambiente virtual está ativo.

```
(venv) C:\...\backend2>
```

> [!IMPORTANT]
> **Sempre que abrir um novo terminal**, você precisará ativar o ambiente virtual novamente usando o comando `activate` acima.

### Passo 4 — Instalar as dependências

Com o ambiente virtual ativo (verifique o `(venv)` no terminal), execute:

```bash
pip install -r requirements.txt
```

Esse comando lê o arquivo `requirements.txt` e instala automaticamente todos os pacotes que a aplicação precisa. A instalação pode levar alguns minutos na primeira vez.

### Passo 5 — Configurar variáveis de ambiente (opcional)

Este passo é **opcional**. A ferramenta funciona sem LLM (Inteligência Artificial), mas com qualidade reduzida — nós KNIME desconhecidos ficarão como `pass  # TODO`.

Se quiser habilitar a tradução por IA:

1. Copie o arquivo de exemplo:

   **Windows:**

   ```cmd
   copy .env.example .env
   ```

   **macOS / Linux:**

   ```bash
   cp .env.example .env
   ```

2. Abra o arquivo `.env` com qualquer editor de texto e preencha:

   ```env
   GOOGLE_CLOUD_PROJECT=seu-project-id-aqui
   GOOGLE_CLOUD_LOCATION=us-central1
   ```

3. Configure as credenciais do Google Cloud:

   ```bash
   gcloud auth application-default login
   ```

> [!NOTE]
> A IA utiliza o modelo **Gemini 2.5 Pro** via Google Vertex AI. Você precisa de um projeto Google Cloud com a API do Vertex AI habilitada. Se não tiver, a ferramenta funciona normalmente sem IA.

---

## Como Executar

A aplicação oferece duas ferramentas de linha de comando: o **transpilador principal** e o **extrator de SQL**.

### Ferramenta 1 — Transpilador (principal)

Converte um arquivo `.knwf` completo em código Python.

**Sintaxe:**

```bash
python transpile.py caminho/para/seu_arquivo.knwf
```

**Exemplo prático:**

```bash
python transpile.py ../fluxo_knime_exemplo.knwf
```

**Arquivos gerados:**

| Arquivo | Conteúdo |
|---------|----------|
| `fluxo_knime_exemplo.py` | Código Python gerado — pronto para execução |
| `fluxo_knime_exemplo_log.md` | Relatório detalhado da transpilação |

O terminal exibirá um resumo ao final:

```
============================================================
COMPLETE
============================================================
Nodes:    299
Matched:  299
Fallback: 0
Coverage: 100.0%
============================================================
Output:   C:\...\fluxo_knime_exemplo.py
Log:      C:\...\fluxo_knime_exemplo_log.md
============================================================
```

> [!TIP]
> O **Coverage** indica a porcentagem de nós KNIME que foram traduzidos com sucesso. Quanto mais próximo de 100%, melhor a tradução.

### Ferramenta 2 — Extrator de SQL

Extrai apenas as queries SQL presentes no workflow, sem transpilação completa.

**Sintaxe:**

```bash
python extract_sql.py caminho/para/seu_arquivo.knwf
```

**Exemplo prático:**

```bash
python extract_sql.py ../fluxo_knime_exemplo.knwf
```

**Arquivo gerado:**

| Arquivo | Conteúdo |
|---------|----------|
| `fluxo_knime_exemplo_sql_queries.py` | Funções Python com todas as queries SQL extraídas |

---

## Estrutura do Projeto

O projeto possui 6 arquivos, cada um com uma responsabilidade específica:

```
backend2/
├── transpile.py              # 🔄 Motor principal — converte .knwf → .py
├── extract_sql.py            # 🔍 Extrator de queries SQL dos nós de banco
├── llm_fallback.py           # 🤖 IA para nós desconhecidos (Vertex AI)
├── llm_string_translator.py  # 🧠 Traduz expressões Java/String para pandas
├── requirements.txt          # 📦 Lista de dependências do projeto
└── .env.example              # ⚙️ Modelo de configuração de variáveis
```

### Descrição de cada arquivo

| Arquivo | O que faz | Quando é usado |
|---------|-----------|----------------|
| **transpile.py** | Abre o `.knwf`, identifica cada nó KNIME, e gera o código Python equivalente usando templates pré-definidos. É o coração da aplicação. | Sempre — é o comando principal |
| **extract_sql.py** | Percorre o `.knwf` buscando nós de banco de dados (DB Reader, DB Query Reader, etc.) e extrai as queries SQL. | Quando você quer apenas as queries SQL |
| **llm_fallback.py** | Quando um nó KNIME não tem template, envia para a IA Gemini 2.5 Pro gerar o código Python. Inclui proteções de retry e circuit breaker. | Automaticamente, se configurado |
| **llm_string_translator.py** | Traduz expressões de manipulação de texto do KNIME (tipo `substr($Col$, 0, 5)`) para pandas (tipo `df["Col"].str.slice(0, 5)`). | Automaticamente, se configurado |
| **requirements.txt** | Lista de todos os pacotes Python necessários. O pip usa este arquivo para instalá-los. | Uma vez, durante a instalação |
| **.env.example** | Modelo com as variáveis de ambiente. Copie para `.env` e preencha com seus dados. | Uma vez, durante a configuração |

---

## Uso da Aplicação

### Uso Básico — Transpilação simples

O cenário mais comum: converter um workflow KNIME para Python.

1. Coloque seu arquivo `.knwf` em uma pasta acessível.
2. Abra o terminal na pasta `backend2` (com `venv` ativado).
3. Execute:

```bash
python transpile.py C:\meus_workflows\relatorio_mensal.knwf
```

1. Dois arquivos serão criados na **mesma pasta** do `.knwf`:
   - `relatorio_mensal.py` — Seu código Python.
   - `relatorio_mensal_log.md` — Relatório da transpilação.

2. Abra o arquivo `_log.md` para verificar se houve problemas.

### Uso Intermediário — Extrair apenas as queries SQL

Quando você quer apenas ver as queries SQL que estão dentro do workflow:

```bash
python extract_sql.py C:\meus_workflows\relatorio_mensal.knwf
```

O terminal exibirá cada query encontrada e salvará tudo em `relatorio_mensal_sql_queries.py`.

### Uso Avançado — Transpilação com IA habilitada

Para obter a melhor qualidade possível na conversão:

1. Configure o `.env` com suas credenciais Google Cloud (veja a seção [Configuração](#configuração)).
2. Execute a transpilação normalmente — a IA será ativada automaticamente quando necessário:

```bash
python transpile.py C:\meus_workflows\workflow_complexo.knwf
```

Nós que não possuem template pré-definido serão enviados para o Gemini 2.5 Pro, que gerará o código Python equivalente.

> [!NOTE]
> **Com IA vs. Sem IA:**
>
> - **Sem IA:** Nós desconhecidos geram `pass  # TODO: Implement NomeDoNo`. Funcional, mas requer ajuste manual.
> - **Com IA:** Nós desconhecidos recebem código Python gerado automaticamente, aumentando a cobertura.

### Uso Avançado — Executar o código gerado

Após a transpilação, o arquivo `.py` gerado pode ser executado diretamente:

```bash
python relatorio_mensal.py
```

> [!WARNING]
> O código gerado geralmente precisa de uma **conexão com banco de dados** para os nós SQL. Você precisará configurar a string de conexão dentro do arquivo gerado antes de executá-lo.

---

## Configuração

### Variáveis de Ambiente

Todas as configurações são feitas via variáveis de ambiente, definidas no arquivo `.env`:

| Variável | Obrigatória | Valores | Descrição |
|----------|-------------|---------|-----------|
| `GOOGLE_CLOUD_PROJECT` | Não* | ID do projeto GCP | Habilita tradução por IA |
| `GOOGLE_CLOUD_LOCATION` | Não | `us-central1` (padrão) | Região do Vertex AI |
| `GOOGLE_API_KEY` | Não* | Chave da API Gemini | Alternativa ao Vertex AI |

> \* Obrigatória apenas se quiser habilitar a tradução por IA. Sem essas variáveis, a ferramenta funciona normalmente com templates pré-definidos.

### Configurar autenticação Google Cloud (para IA)

**Passo 1:** Instale o [Google Cloud CLI](https://cloud.google.com/sdk/docs/install).

**Passo 2:** Faça login:

```bash
gcloud auth application-default login
```

Uma janela do navegador abrirá para você fazer login com sua conta Google.

**Passo 3:** Configure o projeto:

```bash
gcloud config set project seu-project-id
```

**Passo 4:** Verifique se a API do Vertex AI está habilitada:

```bash
gcloud services enable aiplatform.googleapis.com
```

### Arquivo `.env` completo

```env
# Credenciais Google Cloud
GOOGLE_CLOUD_PROJECT=meu-projeto-gcp
GOOGLE_CLOUD_LOCATION=us-central1
```

### Parâmetros internos do LLM

Estes valores são configurados internamente e **não precisam ser alterados** para uso normal:

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| Modelo | `gemini-2.5-pro` | Modelo de IA (fixo, não alterável) |
| Temperatura | `0.0` | Respostas determinísticas |
| Max tokens | `2048` | Limite de resposta |
| Timeout | `30s` | Tempo máximo por requisição |
| Retries | `3` | Tentativas em caso de falha |
| Circuit Breaker | `5 falhas` | Pausa requisições após 5 erros |

---

## Perguntas Frequentes

### "Recebi um erro `python: command not found`"

O Python não está no PATH do sistema. Veja a seção [Pré-requisitos](#1-python-versão-310-ou-superior) para instruções de instalação.

No **Windows**, tente `py` em vez de `python`:

```cmd
py transpile.py seu_arquivo.knwf
```

### "O `pip install` falhou com erro de permissão"

Certifique-se de que o ambiente virtual está ativo (o `(venv)` deve aparecer no terminal). Se o problema persistir:

```bash
python -m pip install -r requirements.txt
```

### "O arquivo gerado tem muitos `pass  # TODO`"

Isso significa que esses nós KNIME não possuem template e a IA não está configurada. Veja a seção [Configuração](#configuração) para habilitar a tradução por IA.

### "Como desativo o ambiente virtual?"

```bash
deactivate
```

---

## Requisitos de Sistema

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| **Python** | 3.10 | 3.12+ |
| **RAM** | 2 GB | 4 GB |
| **Disco** | 500 MB | 1 GB |
| **SO** | Windows 10, macOS 12, Ubuntu 20.04 | Qualquer versão recente |
| **Internet** | Não necessário* | Necessário para IA |

> \* A internet é necessária apenas para instalar dependências (`pip install`) e para a funcionalidade de tradução por IA.
