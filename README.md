# 🐍 Sistema de Orquestração Python

## Guia Completo de Configuração e Uso

Este guia explica **passo a passo** como configurar e utilizar o sistema de orquestração, composto por dois arquivos Python que trabalham juntos para automatizar a execução de scripts.

---

## 📋 Índice

1. [O que é este sistema?](#-o-que-é-este-sistema)
2. [Pré-requisitos](#-pré-requisitos)
3. [Estrutura dos Arquivos](#-estrutura-dos-arquivos)
4. [Configurando o Orquestrador Filho](#-configurando-o-orquestrador-filho)
5. [Configurando o Orquestrador Pai](#-configurando-o-orquestrador-pai)
6. [Configurando o Envio de E-mail](#-configurando-o-envio-de-e-mail)
7. [Agendando Execução Automática](#-agendando-execução-automática)
8. [Como Executar](#-como-executar)
9. [Onde Encontrar os Resultados](#-onde-encontrar-os-resultados)
10. [Perguntas Frequentes](#-perguntas-frequentes)
11. [Solução de Problemas](#-solução-de-problemas)

---

## 🤔 O que é este sistema?

Imagine que você tem vários scripts Python que precisam ser executados em uma ordem específica, todos os dias, sem que você precise abrir cada um manualmente. Este sistema faz exatamente isso:

- O **Orquestrador Filho** (`orquestrador_filho.py`) é um "gerente local" — ele fica dentro de cada projeto e cuida de executar os scripts daquele projeto específico, um após o outro.
- O **Orquestrador Pai** (`orquestrador_pai.py`) é o "gerente geral" — ele dispara todos os orquestradores filhos de diferentes projetos, coleta os resultados e gera um relatório.

**Exemplo prático:**

```
Orquestrador Pai (arquivo central)
├── Orquestrador Filho do Projeto A (executa: extrair_dados.py → tratar_dados.py → gerar_planilha.py)
├── Orquestrador Filho do Projeto B (executa: baixar_relatorio.py → enviar_email.py)
└── Orquestrador Filho do Projeto C (executa: backup_banco.py → limpar_temporarios.py)
```

---

## ✅ Pré-requisitos

### 1. Python instalado

Antes de tudo, você precisa ter o Python instalado no seu computador.

**Como verificar se o Python está instalado:**

1. Pressione `Windows + R` no teclado
2. Digite `cmd` e pressione Enter
3. Na janela preta que abrir, digite:

   ```
   python --version
   ```

4. Se aparecer algo como `Python 3.10.5`, está instalado ✅
5. Se aparecer um erro, você precisa instalar o Python. Baixe em: <https://www.python.org/downloads/>

> ⚠️ **IMPORTANTE:** Durante a instalação do Python, marque a opção **"Add Python to PATH"** (Adicionar Python ao PATH).

### 2. Biblioteca pywin32 (somente se for usar e-mail)

Se você quiser que o sistema envie e-mails pelo Outlook automaticamente, precisa instalar uma biblioteca extra. Abra o Prompt de Comando (cmd) e digite:

```
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pywin32
```

> ⚠️ **REGRA DE INSTALAÇÃO:** Sempre que precisar instalar qualquer pacote Python, use o comando com os flags `--trusted-host`:
>
> ```
> pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org nome_do_pacote
> ```

---

## 📦 Instalação de Dependências

Abra o **Prompt de Comando** (pressione `Windows + R`, digite `cmd` e pressione Enter) e copie e cole os comandos abaixo conforme necessário:

**Instalar pywin32** (necessário apenas se for usar envio de e-mail pelo Outlook):

```
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pywin32
```

> 💡 **Dica:** O `orquestrador_filho.py` usa apenas bibliotecas padrão do Python e **não precisa de instalação extra**.

> ⚠️ **IMPORTANTE:** Caso precise instalar qualquer outro pacote Python no futuro, **sempre** use o formato:
>
> ```
> pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org nome_do_pacote
> ```

---

## 📁 Estrutura dos Arquivos

Ao executar o sistema, a seguinte estrutura de pastas será criada automaticamente:

```
📂 Sua Pasta do Projeto
├── 📄 orquestrador_pai.py        ← Arquivo principal (gerente geral)
├── 📄 orquestrador_filho.py      ← Template (copie para cada projeto)
├── 📂 logs/                      ← Criada automaticamente
│   ├── OrquestradorPai_20260209_060000.log
│   └── OrquestradorFilho_20260209_060001.log
└── 📂 relatorios/                ← Criada automaticamente
    └── relatorio_20260209_060030.txt
```

As pastas `logs/` e `relatorios/` são criadas automaticamente na primeira execução. Você não precisa criá-las manualmente.

---

## 👶 Configurando o Orquestrador Filho

O orquestrador filho é o arquivo que você vai **copiar para cada projeto** que precisa ser automatizado. Siga os passos:

### Passo 1 — Copie o arquivo para o seu projeto

Copie o arquivo `orquestrador_filho.py` para a pasta do projeto onde estão os scripts que você quer executar.

**Exemplo:** Se seus scripts estão na pasta `C:\Projetos\Relatorios\`, copie o arquivo para lá:

```
📂 C:\Projetos\Relatorios\
├── 📄 orquestrador_filho.py     ← Você copiou para cá
├── 📄 extrair_dados.py          ← Seu script 1
├── 📄 tratar_dados.py           ← Seu script 2
└── 📄 gerar_planilha.py         ← Seu script 3
```

### Passo 2 — Abra o arquivo e localize a seção de configuração

Abra o arquivo `orquestrador_filho.py` em qualquer editor de texto (Bloco de Notas, VS Code, Notepad++, etc.).

Procure a seção que começa com:

```
# ╔═══════════════════════════════════════════════════════════════╗
# ║                   CONFIGURAÇÃO DO USUÁRIO                      ║
# ╚═══════════════════════════════════════════════════════════════╝
```

Toda a configuração que você precisa editar está **entre essa linha e a linha que diz "FIM DA CONFIGURAÇÃO DO USUÁRIO"**.

### Passo 3 — Adicione seus scripts na lista

Localize a variável `SCRIPTS_A_EXECUTAR` e adicione os nomes dos seus scripts Python. Existem duas formas:

#### Forma simples (apenas o nome do arquivo)

Se os scripts estão na **mesma pasta** que o orquestrador filho:

```python
SCRIPTS_A_EXECUTAR = [
    "extrair_dados.py",
    "tratar_dados.py",
    "gerar_planilha.py",
]
```

#### Forma com caminho completo

Se os scripts estão em **pastas diferentes**:

```python
SCRIPTS_A_EXECUTAR = [
    r"C:\Projetos\Relatorios\extrair_dados.py",
    r"C:\Projetos\Relatorios\tratar_dados.py",
    r"C:\Outros\gerar_planilha.py",
]
```

> 💡 **Dica:** O `r` antes das aspas (chamado de "raw string") evita problemas com as barras invertidas `\` nos caminhos do Windows. **Sempre use** `r"..."` para caminhos no Windows.

#### Forma avançada (com argumentos)

Se algum script precisa receber parâmetros extras:

```python
SCRIPTS_A_EXECUTAR = [
    "extrair_dados.py",
    {"caminho": "tratar_dados.py", "argumentos": ["--verbose", "--ano", "2026"]},
    "gerar_planilha.py",
]
```

> 📌 **A ordem importa!** Os scripts são executados de cima para baixo, na ordem que você colocou na lista.

### Passo 4 — Configure as opções adicionais

#### Timeout (tempo máximo por script)

```python
TIMEOUT_SEGUNDOS = 300  # 300 segundos = 5 minutos
```

Se um script demorar mais do que esse tempo, ele será encerrado automaticamente. Aumente esse valor se seus scripts demoram muito:

```python
TIMEOUT_SEGUNDOS = 1800  # 1800 segundos = 30 minutos
```

#### Comportamento quando um script falhar

```python
COMPORTAMENTO_EM_ERRO = "continuar"
```

Existem três opções:

| Opção | O que acontece |
|-------|----------------|
| `"parar"` | Se um script falhar, todos os seguintes **não serão executados** |
| `"continuar"` | Se um script falhar, o sistema **pula para o próximo** e continua |
| `"reiniciar"` | Se um script falhar, o sistema **tenta executar novamente** (até 3 vezes) |

#### Número de tentativas (apenas para modo "reiniciar")

```python
MAX_TENTATIVAS = 3
```

#### Nome do orquestrador (aparece nos logs)

```python
NOME_ORQUESTRADOR = "Relatorios_Diarios"
```

Dê um nome descritivo para facilitar a identificação nos relatórios.

#### Pasta dos logs

```python
PASTA_LOGS = "logs"
```

Pode ser alterada se preferir outra localização.

### Passo 5 — Salve o arquivo

Após fazer todas as alterações, salve o arquivo (`Ctrl + S`).

---

## 👨‍👧‍👦 Configurando o Orquestrador Pai

O orquestrador pai é o arquivo central que executa todos os orquestradores filhos.

### Passo 1 — Abra o arquivo orquestrador_pai.py

Abra o arquivo `orquestrador_pai.py` no editor de texto.

### Passo 2 — Adicione os orquestradores filhos na lista

Localize a variável `ORQUESTRADORES_FILHOS` e adicione os caminhos dos orquestradores filhos que você configurou no passo anterior:

```python
ORQUESTRADORES_FILHOS = [
    {
        "nome": "Relatorios_Diarios",
        "caminho": r"C:\Projetos\Relatorios\orquestrador_filho.py"
    },
    {
        "nome": "Backup_Banco",
        "caminho": r"C:\Projetos\Backup\orquestrador_filho.py"
    },
    {
        "nome": "Limpeza_Temporarios",
        "caminho": r"C:\Projetos\Limpeza\orquestrador_filho.py"
    },
]
```

**Explicação de cada campo:**

| Campo | O que é | Obrigatório? |
|-------|---------|--------------|
| `"nome"` | Um nome curto e descritivo para identificar o orquestrador filho | Sim |
| `"caminho"` | O caminho completo para o arquivo `orquestrador_filho.py` do projeto | Sim |
| `"argumentos"` | Lista de argumentos extras (raro de usar) | Não |

> 💡 **Dica:** Sempre use `r"..."` nos caminhos e use barras invertidas `\` como no Windows.

### Passo 3 — Escolha o modo de execução

```python
MODO_EXECUCAO = "sequencial"
```

| Modo | O que acontece |
|------|----------------|
| `"sequencial"` | Executa um orquestrador filho por vez, aguardando cada um terminar antes de iniciar o próximo. **Mais seguro.** |
| `"paralelo"` | Executa todos ao mesmo tempo. **Mais rápido**, mas usa mais recursos do computador. |

> 💡 **Recomendação:** Use `"sequencial"` se não sabe qual escolher.

### Passo 4 — Configure o timeout

```python
TIMEOUT_SEGUNDOS = 1800  # 30 minutos por padrão
```

Este é o tempo máximo que **cada orquestrador filho** pode levar para finalizar. Se passar desse tempo, ele será encerrado.

### Passo 5 — Dê um nome ao orquestrador pai

```python
NOME_ORQUESTRADOR = "OrquestradorPai"
```

### Passo 6 — Salve o arquivo

Após fazer todas as alterações, salve o arquivo (`Ctrl + S`).

---

## 📧 Configurando o Envio de E-mail

O sistema pode enviar automaticamente o relatório por e-mail usando o Microsoft Outlook.

> ⚠️ **Requisito:** O Microsoft Outlook precisa estar instalado e configurado na máquina.

### Passo 1 — Instale a biblioteca pywin32

Abra o Prompt de Comando (cmd) e execute:

```
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pywin32
```

### Passo 2 — Habilite o envio de e-mail

No arquivo `orquestrador_pai.py`, altere:

```python
ENVIAR_EMAIL = True  # Mude de False para True
```

### Passo 3 — Configure os destinatários

```python
DESTINATARIOS_EMAIL = [
    "joao.silva@empresa.com",
    "maria.santos@empresa.com",
]
```

Para adicionar cópia (CC):

```python
DESTINATARIOS_CC = [
    "gestor@empresa.com",
]
```

### Passo 4 — Personalize o assunto e corpo do e-mail

```python
ASSUNTO_EMAIL = "Relatório de Execução - Orquestrador - {data}"
```

O `{data}` será automaticamente substituído pela data atual (ex: "06/02/2026").

```python
CORPO_EMAIL = """
Prezados,

Segue em anexo o relatório de execução do orquestrador.

{resumo}

Atenciosamente,
Sistema de Orquestração
"""
```

O `{resumo}` será substituído por um resumo automático com as estatísticas da execução.

> 💡 **Se o e-mail falhar:** O sistema salva o relatório localmente na pasta `relatorios/`. Você não perde nenhuma informação.

---

## ⏰ Agendando Execução Automática

Para que o sistema execute automaticamente (ex: todos os dias às 6h da manhã), configure o Agendador de Tarefas do Windows.

### Passo 1 — Configure as opções de agendamento

No arquivo `orquestrador_pai.py`, configure:

```python
# Nome que aparecerá no Agendador de Tarefas do Windows
NOME_TAREFA_AGENDADA = "Orquestrador_Automatico"

# Horário de execução (formato HH:MM)
HORARIO_EXECUCAO = "06:00"

# Frequência: "diaria", "semanal" ou "mensal"
FREQUENCIA_EXECUCAO = "diaria"
```

**Para execução semanal** — defina também os dias:

```python
FREQUENCIA_EXECUCAO = "semanal"

# 1=Segunda, 2=Terça, 3=Quarta, 4=Quinta, 5=Sexta, 6=Sábado, 7=Domingo
DIAS_SEMANA = [1, 2, 3, 4, 5]  # Segunda a Sexta
```

**Para execução mensal** — defina o dia do mês:

```python
FREQUENCIA_EXECUCAO = "mensal"
DIA_MES = 1  # Todo dia 1 do mês
```

### Passo 2 — Execute o comando de agendamento

Abra o **Prompt de Comando como Administrador** (clique com botão direito → "Executar como administrador") e execute:

```
python orquestrador_pai.py --agendar
```

> ⚠️ **É necessário rodar como Administrador** para criar tarefas no Agendador de Tarefas.

Se tudo correr bem, aparecerá a mensagem:

```
Tarefa 'Orquestrador_Automatico' criada com sucesso!
```

### Como verificar o agendamento

1. Pressione `Windows + R`
2. Digite `taskschd.msc` e pressione Enter
3. Procure a tarefa com o nome que você definiu (ex: "Orquestrador_Automatico")

---

## 🚀 Como Executar

### Execução Normal (todos os orquestradores filhos)

Abra o Prompt de Comando, navegue até a pasta do orquestrador pai e execute:

```
python orquestrador_pai.py
```

### Executar apenas um filho específico (modo de teste)

Para depurar ou testar um orquestrador filho sem executar todos:

```
python orquestrador_pai.py --teste Relatorios_Diarios
```

Substitua `Relatorios_Diarios` pelo nome do orquestrador filho que você definiu.

### Listar todos os filhos configurados

```
python orquestrador_pai.py --listar
```

Exibe uma lista com todos os orquestradores filhos e seus caminhos.

### Executar o orquestrador filho diretamente

Se quiser testar um orquestrador filho de forma isolada:

```
python orquestrador_filho.py
```

Execute esse comando dentro da pasta onde o orquestrador filho está localizado.

---

## 📊 Onde Encontrar os Resultados

### Logs de execução

Os logs ficam na pasta `logs/` e contêm o registro detalhado de tudo que aconteceu:

```
logs/OrquestradorPai_20260209_060000.log
logs/OrquestradorFilho_20260209_060001.log
```

**Exemplo de conteúdo de um log:**

```
2026-02-09 06:00:01 | INFO     | INÍCIO DA EXECUÇÃO: OrquestradorPai
2026-02-09 06:00:01 | INFO     | Total de orquestradores filhos: 3
2026-02-09 06:00:01 | INFO     | Iniciando orquestrador filho: Relatorios_Diarios
2026-02-09 06:00:15 | INFO     | Concluído: Relatorios_Diarios - Status: sucesso - Duração: 14.32s
```

### Relatórios TXT

Os relatórios ficam na pasta `relatorios/` e contêm uma tabela resumida:

```
relatorios/relatorio_20260209_060030.txt
```

**Exemplo de conteúdo de um relatório:**

```
================================================================
  RELATÓRIO DE EXECUÇÃO - ORQUESTRADOR PAI
  Data: 09/02/2026 06:00:30
  Duração Total: 45.67 segundos
================================================================

+----------------------+-----------------+---------------------+---------------------+----------------------------------------------------+
| ORQUESTRADOR         | STATUS          | INÍCIO              | CONCLUSÃO           | MOTIVO FALHA                                       |
+----------------------+-----------------+---------------------+---------------------+----------------------------------------------------+
| Relatorios_Diarios   | SUCESSO         | 09/02/2026 06:00:01 | 09/02/2026 06:00:15 |                                                    |
| Backup_Banco         | SUCESSO         | 09/02/2026 06:00:15 | 09/02/2026 06:00:28 |                                                    |
| Limpeza_Temporarios  | FALHA           | 09/02/2026 06:00:28 | 09/02/2026 06:00:30 | PermissionError: acesso negado ao arquivo X         |
+----------------------+-----------------+---------------------+---------------------+----------------------------------------------------+

RESUMO:
  - Total de orquestradores: 3
  - Sucessos: 2
  - Falhas: 1
  - Taxa de sucesso: 66.7%
```

### Resultados JSON (para uso técnico)

Cada orquestrador filho gera um arquivo JSON na pasta `logs/` com todos os dados estruturados. Esses arquivos são usados internamente pelo orquestrador pai.

---

## ❓ Perguntas Frequentes

### "Preciso saber programar para usar?"

**Não.** Você só precisa editar a lista de scripts na seção de configuração. Todas as instruções estão em português e o restante do código não precisa ser alterado.

### "O que acontece se um script falhar?"

Depende da configuração `COMPORTAMENTO_EM_ERRO`:

- `"parar"` → Para tudo
- `"continuar"` → Pula para o próximo
- `"reiniciar"` → Tenta novamente

O erro será registrado no log e no relatório.

### "Posso usar caminhos relativos?"

**Sim.** Se o script está na mesma pasta que o orquestrador filho, basta usar o nome: `"meu_script.py"`. Se está em uma subpasta: `"pasta/meu_script.py"`.

### "O sistema abre alguma janela na tela?"

**Não.** Toda a execução ocorre em segundo plano, sem abrir janelas visíveis.

### "O que é o `r` antes dos caminhos?"

É uma "raw string" do Python. Ela impede que a barra invertida `\` (usada nos caminhos do Windows) seja interpretada como caractere especial. **Sempre use `r"..."` em caminhos.**

### "Posso ter mais de um orquestrador filho por projeto?"

**Sim.** Basta copiar o `orquestrador_filho.py` com nomes diferentes (ex: `orquestrador_diario.py`, `orquestrador_semanal.py`) e referenciar cada um no orquestrador pai.

---

## 🔧 Solução de Problemas

### Erro: `python não é reconhecido como comando`

O Python não está no PATH do sistema. Reinstale o Python marcando a opção **"Add Python to PATH"**.

### Erro: `ModuleNotFoundError: No module named 'win32com'`

Instale a biblioteca:

```
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pywin32
```

### Erro: `FileNotFoundError` no relatório

O caminho de algum script está incorreto. Verifique:

- Se o caminho está escrito corretamente
- Se o arquivo realmente existe naquele local
- Se está usando `r"..."` para caminhos com barras invertidas

### Erro: `TimeoutError` no relatório

O script demorou mais que o tempo configurado. Aumente o valor de `TIMEOUT_SEGUNDOS`.

### E-mail não está sendo enviado

Verifique:

1. `ENVIAR_EMAIL` está como `True`?
2. `DESTINATARIOS_EMAIL` tem pelo menos um endereço?
3. O Outlook está instalado e configurado?
4. A biblioteca `pywin32` está instalada?

### O agendamento não foi criado

Execute o comando como **Administrador**:

1. Clique com botão direito no Prompt de Comando
2. Selecione "Executar como administrador"
3. Depois execute: `python orquestrador_pai.py --agendar`

---

## 📖 Resumo Rápido

| O que fazer | Onde fazer | O que editar |
|-------------|------------|--------------|
| Adicionar scripts de um projeto | `orquestrador_filho.py` | `SCRIPTS_A_EXECUTAR` |
| Conectar projetos ao orquestrador | `orquestrador_pai.py` | `ORQUESTRADORES_FILHOS` |
| Enviar e-mail com relatório | `orquestrador_pai.py` | `ENVIAR_EMAIL`, `DESTINATARIOS_EMAIL` |
| Agendar execução automática | Prompt de Comando | `python orquestrador_pai.py --agendar` |
| Testar um projeto específico | Prompt de Comando | `python orquestrador_pai.py --teste NomeDoFilho` |
| Ver relatório | Pasta `relatorios/` | Abrir o arquivo `.txt` mais recente |
| Ver log detalhado | Pasta `logs/` | Abrir o arquivo `.log` mais recente |
