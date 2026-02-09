# -*- coding: utf-8 -*-
"""
================================================================================
ORQUESTRADOR PAI - Gerenciador Central de Execução de Orquestradores Filhos
================================================================================

DESCRIÇÃO:
    Este arquivo é o ponto central de execução que gerencia e executa múltiplos
    orquestradores filhos. Ele coleta os resultados de cada filho, gera um
    relatório tabulado em formato TXT, envia por e-mail via Outlook, e pode
    ser agendado automaticamente no Task Scheduler do Windows.

DEPENDÊNCIAS:
    Para funcionalidade de e-mail via Outlook, instale o pywin32:
    
    pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pywin32
    
    IMPORTANTE: Sempre use os flags --trusted-host ao instalar pacotes.

COMO USAR:
    1. Edite a seção "CONFIGURAÇÃO DO USUÁRIO" abaixo
    2. Adicione os caminhos dos orquestradores filhos na lista ORQUESTRADORES_FILHOS
    3. Configure as opções de e-mail se desejar envio automático
    
    Modos de execução:
        python orquestrador_pai.py                    # Execução normal
        python orquestrador_pai.py --teste NomeFilho  # Executa apenas um filho específico
        python orquestrador_pai.py --agendar          # Configura agendamento no Windows

AUTOR: Gerado automaticamente
DATA: 2026-02-06
================================================================================
"""

# ==============================================================================
# IMPORTAÇÕES NECESSÁRIAS (bibliotecas padrão do Python, não precisa instalar nada)
# ==============================================================================
import os
import sys
import json
import subprocess
import logging
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

# ==============================================================================
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        CONFIGURAÇÃO DO USUÁRIO                            ║
# ║                                                                           ║
# ║  Edite apenas esta seção para configurar o orquestrador pai               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# ==============================================================================

# Lista de orquestradores filhos a serem executados
# Formato: Lista de dicionários com as seguintes chaves:
#   - "nome": Nome identificador do orquestrador filho (obrigatório)
#   - "caminho": Caminho do arquivo orquestrador_filho.py (obrigatório)
#   - "argumentos": Lista de argumentos extras (opcional)
#
# Exemplo:
#   ORQUESTRADORES_FILHOS = [
#       {"nome": "Projeto_A", "caminho": r"C:\Projetos\A\orquestrador_filho.py"},
#       {"nome": "Projeto_B", "caminho": r"C:\Projetos\B\orquestrador_filho.py"},
#       {"nome": "Backup", "caminho": "./backup/orquestrador_filho.py", "argumentos": ["--modo", "noturno"]},
#   ]

ORQUESTRADORES_FILHOS: list[dict] = [
    # Adicione seus orquestradores filhos aqui (remova o # do início da linha para ativar)
    # {"nome": "Projeto_Exemplo", "caminho": r"C:\MeuProjeto\orquestrador_filho.py"},
]

# Modo de execução dos orquestradores filhos
# Opções:
#   "sequencial" - Executa um filho por vez, aguardando a conclusão de cada um
#   "paralelo"   - Executa todos os filhos simultaneamente (mais rápido, mais recursos)
MODO_EXECUCAO: str = "sequencial"

# Tempo máximo de execução para cada orquestrador filho (em segundos)
TIMEOUT_SEGUNDOS: int = 1800  # 30 minutos por padrão

# Nome identificador deste orquestrador pai (usado nos logs e relatórios)
NOME_ORQUESTRADOR: str = "OrquestradorPai"

# Pasta onde os logs serão salvos (relativa ao diretório do script)
PASTA_LOGS: str = "logs"

# Pasta onde os relatórios TXT serão salvos
PASTA_RELATORIOS: str = "relatorios"

# ------------------------------------------------------------------------------
# CONFIGURAÇÃO DE E-MAIL (via Outlook/win32com)
# ------------------------------------------------------------------------------

# Habilita ou desabilita o envio de e-mail
ENVIAR_EMAIL: bool = False

# Destinatários do e-mail (lista de endereços)
DESTINATARIOS_EMAIL: list[str] = [
    # "usuario@empresa.com",
    # "equipe@empresa.com",
]

# Destinatários em cópia (CC)
DESTINATARIOS_CC: list[str] = []

# Assunto do e-mail (pode incluir {data} para data atual)
ASSUNTO_EMAIL: str = "Relatório de Execução - Orquestrador - {data}"

# Corpo do e-mail (pode incluir {resumo} para resumo da execução)
CORPO_EMAIL: str = """
Prezados,

Segue em anexo o relatório de execução do orquestrador.

{resumo}

Este é um e-mail automático. Não responda a esta mensagem.

Atenciosamente,
Sistema de Orquestração Automatizada
"""

# ------------------------------------------------------------------------------
# CONFIGURAÇÃO DE AGENDAMENTO (Task Scheduler do Windows)
# ------------------------------------------------------------------------------

# Nome da tarefa no Agendador de Tarefas
NOME_TAREFA_AGENDADA: str = "Orquestrador_Automatico"

# Descrição da tarefa
DESCRICAO_TAREFA: str = "Execução automática do sistema de orquestração Python"

# Horário de execução (formato HH:MM)
HORARIO_EXECUCAO: str = "06:00"

# Frequência de execução
# Opções: "diaria", "semanal", "mensal"
FREQUENCIA_EXECUCAO: str = "diaria"

# Dias da semana para execução semanal (1=Segunda, 7=Domingo)
# Exemplo: [1, 3, 5] para Segunda, Quarta e Sexta
DIAS_SEMANA: list[int] = [1, 2, 3, 4, 5]  # Segunda a Sexta

# Dia do mês para execução mensal (1-31)
DIA_MES: int = 1

# ==============================================================================
# FIM DA CONFIGURAÇÃO DO USUÁRIO - NÃO EDITE ABAIXO DESTA LINHA
# ==============================================================================


class StatusExecucao(Enum):
    """
    Lista dos possíveis resultados de execução de um orquestrador filho.
    (Enum = lista fixa de opções que não muda)
    """
    SUCESSO = "sucesso"
    FALHA = "falha"
    TIMEOUT = "timeout"
    NAO_ENCONTRADO = "nao_encontrado"
    ERRO_JSON = "erro_json"


class StatusGeral(Enum):
    """
    Lista dos possíveis resultados gerais do orquestrador pai.
    (Enum = lista fixa de opções que não muda)
    """
    SUCESSO_TOTAL = "sucesso_total"
    SUCESSO_PARCIAL = "sucesso_parcial"
    FALHA_TOTAL = "falha_total"
    NAO_EXECUTADO = "nao_executado"


@dataclass
class ResultadoFilho:
    """
    Estrutura que armazena o resultado da execução de um orquestrador filho.
    """
    nome: str
    caminho: str
    status: str
    inicio: str
    fim: str
    duracao_segundos: float
    caminho_json_resultado: str | None = None
    dados_resultado: dict | None = None
    erro_mensagem: str | None = None


def configurar_logging(pasta_logs: str, nome_orquestrador: str) -> str:
    """
    Configura o sistema de logging do orquestrador pai.
    
    Parâmetros:
        pasta_logs: Caminho da pasta onde os logs serão salvos
        nome_orquestrador: Nome do orquestrador para identificação
        
    Retorna:
        Caminho completo do arquivo de log criado
    """
    diretorio_script = Path(__file__).parent.absolute()
    caminho_pasta_logs = diretorio_script / pasta_logs
    caminho_pasta_logs.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo_log = f"{nome_orquestrador}_{timestamp}.log"
    caminho_log = caminho_pasta_logs / nome_arquivo_log
    
    formato_log = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    handler_arquivo = logging.FileHandler(
        filename=str(caminho_log),
        mode='w',
        encoding='utf-8'
    )
    handler_arquivo.setFormatter(formato_log)
    handler_arquivo.setLevel(logging.DEBUG)
    
    # Handler para console (apenas INFO e acima)
    handler_console = logging.StreamHandler()
    handler_console.setFormatter(formato_log)
    handler_console.setLevel(logging.INFO)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(handler_arquivo)
    logger.addHandler(handler_console)
    
    return str(caminho_log)


def _configurar_subprocesso_windows() -> tuple:
    """
    Prepara as configurações para executar subprocessos sem janela visível no Windows.
    
    Retorna:
        Tupla com (creation_flags, startupinfo) para uso no subprocess.run()
    """
    creation_flags = 0
    startupinfo = None
    
    if sys.platform == "win32":
        # CREATE_NO_WINDOW: impede que uma janela de console apareça
        creation_flags = subprocess.CREATE_NO_WINDOW
        # STARTUPINFO: configuração adicional para ocultar a janela
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
    
    return creation_flags, startupinfo


def executar_filho(
    nome: str,
    caminho: str,
    argumentos: list[str],
    timeout_segundos: int
) -> ResultadoFilho:
    """
    Executa um único orquestrador filho e captura seu resultado.
    
    Parâmetros:
        nome: Nome identificador do filho
        caminho: Caminho do arquivo orquestrador_filho.py
        argumentos: Lista de argumentos extras
        timeout_segundos: Tempo limite de execução
        
    Retorna:
        ResultadoFilho com todos os detalhes da execução
    """
    inicio = datetime.now()
    logging.info(f"Iniciando orquestrador filho: {nome}")
    logging.debug(f"Caminho: {caminho}")
    
    # Verifica se o arquivo existe
    if not os.path.isabs(caminho):
        diretorio_script = Path(__file__).parent.absolute()
        caminho_absoluto = str(diretorio_script / caminho)
    else:
        caminho_absoluto = caminho
    
    if not os.path.exists(caminho_absoluto):
        logging.error(f"Arquivo não encontrado: {caminho_absoluto}")
        fim = datetime.now()
        return ResultadoFilho(
            nome=nome,
            caminho=caminho,
            status=StatusExecucao.NAO_ENCONTRADO.value,
            inicio=inicio.isoformat(),
            fim=fim.isoformat(),
            duracao_segundos=(fim - inicio).total_seconds(),
            erro_mensagem=f"Arquivo não encontrado: {caminho_absoluto}"
        )
    
    # Prepara o comando
    comando = [sys.executable, caminho_absoluto] + argumentos
    
    try:
        # Configura execução em segundo plano (sem janela visível)
        creation_flags, startupinfo = _configurar_subprocesso_windows()
        
        # Executa o subprocesso
        processo = subprocess.run(
            comando,
            capture_output=True,
            text=True,
            timeout=timeout_segundos,
            creationflags=creation_flags,
            startupinfo=startupinfo,
            cwd=os.path.dirname(caminho_absoluto)
        )
        
        fim = datetime.now()
        duracao = (fim - inicio).total_seconds()
        
        # O orquestrador filho imprime o caminho do JSON na stdout (pode haver
        # outras mensagens antes, então pegamos apenas a última linha)
        linhas_stdout = processo.stdout.strip().splitlines()
        caminho_json = linhas_stdout[-1].strip() if linhas_stdout else ""
        dados_resultado = None
        status = StatusExecucao.SUCESSO.value if processo.returncode == 0 else StatusExecucao.FALHA.value
        
        # Tenta ler o arquivo JSON de resultado
        if caminho_json and os.path.exists(caminho_json):
            try:
                with open(caminho_json, 'r', encoding='utf-8') as f:
                    dados_resultado = json.load(f)
                logging.info(f"Resultado carregado: {caminho_json}")
            except json.JSONDecodeError as e:
                logging.warning(f"Erro ao ler JSON: {e}")
                status = StatusExecucao.ERRO_JSON.value
        
        logging.info(f"Concluído: {nome} - Status: {status} - Duração: {duracao:.2f}s")
        
        return ResultadoFilho(
            nome=nome,
            caminho=caminho,
            status=status,
            inicio=inicio.isoformat(),
            fim=fim.isoformat(),
            duracao_segundos=duracao,
            caminho_json_resultado=caminho_json if caminho_json else None,
            dados_resultado=dados_resultado,
            erro_mensagem=processo.stderr if processo.returncode != 0 else None
        )
        
    except subprocess.TimeoutExpired:
        fim = datetime.now()
        logging.error(f"Timeout expirado: {nome} (>{timeout_segundos}s)")
        return ResultadoFilho(
            nome=nome,
            caminho=caminho,
            status=StatusExecucao.TIMEOUT.value,
            inicio=inicio.isoformat(),
            fim=fim.isoformat(),
            duracao_segundos=timeout_segundos,
            erro_mensagem=f"Timeout: execução excedeu {timeout_segundos} segundos"
        )
        
    except Exception as e:
        fim = datetime.now()
        logging.error(f"Erro ao executar {nome}: {type(e).__name__}: {str(e)}")
        return ResultadoFilho(
            nome=nome,
            caminho=caminho,
            status=StatusExecucao.FALHA.value,
            inicio=inicio.isoformat(),
            fim=fim.isoformat(),
            duracao_segundos=(fim - inicio).total_seconds(),
            erro_mensagem=f"{type(e).__name__}: {str(e)}"
        )


def executar_filhos_sequencial(
    filhos: list[dict],
    timeout_segundos: int
) -> list[ResultadoFilho]:
    """
    Executa os orquestradores filhos de forma sequencial.
    
    Parâmetros:
        filhos: Lista de configurações dos filhos
        timeout_segundos: Tempo limite para cada filho
        
    Retorna:
        Lista de ResultadoFilho com resultados de cada execução
    """
    resultados = []
    
    for indice, filho in enumerate(filhos, start=1):
        logging.info(f"[{indice}/{len(filhos)}] Processando: {filho['nome']}")
        
        resultado = executar_filho(
            nome=filho["nome"],
            caminho=filho["caminho"],
            argumentos=filho.get("argumentos", []),
            timeout_segundos=timeout_segundos
        )
        
        resultados.append(resultado)
        logging.info("-" * 60)
    
    return resultados


def executar_filhos_paralelo(
    filhos: list[dict],
    timeout_segundos: int
) -> list[ResultadoFilho]:
    """
    Executa os orquestradores filhos de forma paralela.
    
    Parâmetros:
        filhos: Lista de configurações dos filhos
        timeout_segundos: Tempo limite para cada filho
        
    Retorna:
        Lista de ResultadoFilho com resultados de cada execução
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    resultados = []
    
    with ThreadPoolExecutor(max_workers=len(filhos)) as executor:
        # Submete todas as tarefas
        futuros = {
            executor.submit(
                executar_filho,
                filho["nome"],
                filho["caminho"],
                filho.get("argumentos", []),
                timeout_segundos
            ): filho["nome"]
            for filho in filhos
        }
        
        # Coleta os resultados conforme completam
        for futuro in as_completed(futuros):
            nome = futuros[futuro]
            try:
                resultado = futuro.result()
                resultados.append(resultado)
            except Exception as e:
                logging.error(f"Erro ao processar {nome}: {e}")
    
    return resultados


def formatar_data_hora(iso_string: str) -> str:
    """
    Formata uma string ISO para formato legível brasileiro.
    """
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        return iso_string


def gerar_relatorio_txt(
    resultados: list[ResultadoFilho],
    duracao_total: float,
    caminho_destino: str
) -> str:
    """
    Gera um relatório tabulado em formato TXT.
    
    Parâmetros:
        resultados: Lista de ResultadoFilho
        duracao_total: Duração total da execução
        caminho_destino: Pasta onde salvar o relatório
        
    Retorna:
        Caminho do arquivo de relatório gerado
    """
    # Cria pasta de destino
    Path(caminho_destino).mkdir(parents=True, exist_ok=True)
    
    # Define nome do arquivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"relatorio_{timestamp}.txt"
    caminho_relatorio = os.path.join(caminho_destino, nome_arquivo)
    
    # Calcula larguras das colunas
    largura_nome = max(20, max((len(r.nome) for r in resultados), default=20))
    largura_status = 15
    largura_data = 19
    largura_motivo = 50
    
    # Monta o cabeçalho
    separador = "+" + "-" * (largura_nome + 2)
    separador += "+" + "-" * (largura_status + 2)
    separador += "+" + "-" * (largura_data + 2)
    separador += "+" + "-" * (largura_data + 2)
    separador += "+" + "-" * (largura_motivo + 2) + "+"
    
    cabecalho = f"| {'ORQUESTRADOR':<{largura_nome}}"
    cabecalho += f" | {'STATUS':<{largura_status}}"
    cabecalho += f" | {'INÍCIO':<{largura_data}}"
    cabecalho += f" | {'CONCLUSÃO':<{largura_data}}"
    cabecalho += f" | {'MOTIVO FALHA':<{largura_motivo}} |"
    
    linhas = []
    linhas.append("=" * len(separador))
    linhas.append(f"  RELATÓRIO DE EXECUÇÃO - ORQUESTRADOR PAI")
    linhas.append(f"  Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    linhas.append(f"  Duração Total: {duracao_total:.2f} segundos")
    linhas.append("=" * len(separador))
    linhas.append("")
    linhas.append(separador)
    linhas.append(cabecalho)
    linhas.append(separador)
    
    # Adiciona linhas de dados
    for resultado in resultados:
        inicio_fmt = formatar_data_hora(resultado.inicio)
        fim_fmt = formatar_data_hora(resultado.fim)
        motivo = (resultado.erro_mensagem or "")[:largura_motivo]
        
        # Traduz status para português
        status_pt = {
            "sucesso": "SUCESSO",
            "falha": "FALHA",
            "timeout": "TIMEOUT",
            "nao_encontrado": "NÃO ENCONTRADO",
            "erro_json": "ERRO JSON"
        }.get(resultado.status, resultado.status.upper())
        
        linha = f"| {resultado.nome:<{largura_nome}}"
        linha += f" | {status_pt:<{largura_status}}"
        linha += f" | {inicio_fmt:<{largura_data}}"
        linha += f" | {fim_fmt:<{largura_data}}"
        linha += f" | {motivo:<{largura_motivo}} |"
        linhas.append(linha)
    
    linhas.append(separador)
    
    # Adiciona resumo
    total = len(resultados)
    sucessos = sum(1 for r in resultados if r.status == StatusExecucao.SUCESSO.value)
    falhas = total - sucessos
    
    linhas.append("")
    linhas.append("RESUMO:")
    linhas.append(f"  - Total de orquestradores: {total}")
    linhas.append(f"  - Sucessos: {sucessos}")
    linhas.append(f"  - Falhas: {falhas}")
    linhas.append(f"  - Taxa de sucesso: {(sucessos/total*100) if total > 0 else 0:.1f}%")
    linhas.append("")
    linhas.append("=" * len(separador))
    
    # Salva o arquivo
    with open(caminho_relatorio, 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas))
    
    logging.info(f"Relatório gerado: {caminho_relatorio}")
    return caminho_relatorio


def enviar_email_outlook(
    destinatarios: list[str],
    destinatarios_cc: list[str],
    assunto: str,
    corpo: str,
    anexo: str
) -> bool:
    """
    Envia e-mail via Outlook usando win32com.
    
    Parâmetros:
        destinatarios: Lista de e-mails dos destinatários
        destinatarios_cc: Lista de e-mails em cópia
        assunto: Assunto do e-mail
        corpo: Corpo do e-mail
        anexo: Caminho do arquivo a anexar
        
    Retorna:
        True se enviou com sucesso, False caso contrário
    """
    try:
        import win32com.client
        
        logging.info("Conectando ao Outlook...")
        outlook = win32com.client.Dispatch("Outlook.Application")
        mail = outlook.CreateItem(0)  # 0 = olMailItem
        
        mail.To = "; ".join(destinatarios)
        if destinatarios_cc:
            mail.CC = "; ".join(destinatarios_cc)
        mail.Subject = assunto
        mail.Body = corpo
        
        if anexo and os.path.exists(anexo):
            mail.Attachments.Add(anexo)
            logging.info(f"Anexo adicionado: {anexo}")
        
        mail.Send()
        logging.info(f"E-mail enviado para: {', '.join(destinatarios)}")
        return True
        
    except ImportError:
        logging.error(
            "Biblioteca win32com não encontrada. "
            "Instale com: pip install --trusted-host pypi.org "
            "--trusted-host files.pythonhosted.org pywin32"
        )
        return False
        
    except Exception as e:
        logging.error(f"Erro ao enviar e-mail: {type(e).__name__}: {str(e)}")
        return False


def criar_tarefa_agendada() -> bool:
    """
    Cria uma tarefa no Agendador de Tarefas do Windows.
    
    Retorna:
        True se criou com sucesso, False caso contrário
    """
    logging.info("Configurando tarefa agendada...")
    
    # Caminho absoluto deste script
    caminho_script = os.path.abspath(__file__)
    caminho_python = sys.executable
    
    # Monta o comando schtasks
    comando_base = [
        "schtasks", "/create", "/tn", NOME_TAREFA_AGENDADA,
        "/tr", f'"{caminho_python}" "{caminho_script}"',
        "/sc", "DAILY" if FREQUENCIA_EXECUCAO == "diaria" else "WEEKLY" if FREQUENCIA_EXECUCAO == "semanal" else "MONTHLY",
        "/st", HORARIO_EXECUCAO,
        "/f"  # Força substituição se já existir
    ]
    
    # Adiciona dias da semana para execução semanal
    if FREQUENCIA_EXECUCAO == "semanal" and DIAS_SEMANA:
        dias_map = {1: "MON", 2: "TUE", 3: "WED", 4: "THU", 5: "FRI", 6: "SAT", 7: "SUN"}
        dias_str = ",".join(dias_map[d] for d in DIAS_SEMANA if d in dias_map)
        comando_base.extend(["/d", dias_str])
    
    # Adiciona dia do mês para execução mensal
    if FREQUENCIA_EXECUCAO == "mensal":
        comando_base.extend(["/d", str(DIA_MES)])
    
    try:
        resultado = subprocess.run(
            comando_base,
            capture_output=True,
            text=True
        )
        
        if resultado.returncode == 0:
            logging.info(f"Tarefa '{NOME_TAREFA_AGENDADA}' criada com sucesso!")
            logging.info(f"Frequência: {FREQUENCIA_EXECUCAO}")
            logging.info(f"Horário: {HORARIO_EXECUCAO}")
            return True
        else:
            logging.error(f"Erro ao criar tarefa: {resultado.stderr}")
            return False
            
    except Exception as e:
        logging.error(f"Erro ao criar tarefa agendada: {e}")
        return False


def executar_modo_teste(nome_filho: str) -> None:
    """
    Executa apenas um orquestrador filho específico para depuração.
    
    Parâmetros:
        nome_filho: Nome do orquestrador filho a executar
    """
    # Busca o filho pelo nome
    filho_encontrado = None
    for filho in ORQUESTRADORES_FILHOS:
        if filho["nome"].lower() == nome_filho.lower():
            filho_encontrado = filho
            break
    
    if not filho_encontrado:
        print(f"ERRO: Orquestrador filho '{nome_filho}' não encontrado.")
        print("Orquestradores disponíveis:")
        for filho in ORQUESTRADORES_FILHOS:
            print(f"  - {filho['nome']}")
        sys.exit(1)
    
    # Configura logging
    configurar_logging(PASTA_LOGS, f"{NOME_ORQUESTRADOR}_teste")
    
    logging.info(f"=== MODO DE TESTE: {nome_filho} ===")
    
    # Executa apenas este filho
    resultado = executar_filho(
        nome=filho_encontrado["nome"],
        caminho=filho_encontrado["caminho"],
        argumentos=filho_encontrado.get("argumentos", []),
        timeout_segundos=TIMEOUT_SEGUNDOS
    )
    
    # Exibe resultado detalhado
    print("\n" + "=" * 60)
    print("RESULTADO DO TESTE")
    print("=" * 60)
    print(f"Nome: {resultado.nome}")
    print(f"Status: {resultado.status}")
    print(f"Duração: {resultado.duracao_segundos:.2f} segundos")
    if resultado.erro_mensagem:
        print(f"Erro: {resultado.erro_mensagem}")
    if resultado.dados_resultado:
        print("\nDados do resultado:")
        print(json.dumps(resultado.dados_resultado, indent=2, ensure_ascii=False))
    print("=" * 60)


def executar_orquestrador() -> int:
    """
    Função principal que executa o orquestrador pai.
    
    Retorna:
        Código de saída (0 = sucesso, 1 = falha parcial, 2 = falha total)
    """
    inicio_execucao = datetime.now()
    
    # Configura logging
    diretorio_script = Path(__file__).parent.absolute()
    caminho_log = configurar_logging(PASTA_LOGS, NOME_ORQUESTRADOR)
    
    logging.info("=" * 60)
    logging.info(f"INÍCIO DA EXECUÇÃO: {NOME_ORQUESTRADOR}")
    logging.info("=" * 60)
    logging.info(f"Total de orquestradores filhos: {len(ORQUESTRADORES_FILHOS)}")
    logging.info(f"Modo de execução: {MODO_EXECUCAO}")
    logging.info(f"Timeout por filho: {TIMEOUT_SEGUNDOS} segundos")
    logging.info("-" * 60)
    
    # Verifica se há filhos configurados
    if not ORQUESTRADORES_FILHOS:
        logging.warning("Nenhum orquestrador filho configurado!")
        return 2
    
    # Executa os filhos
    if MODO_EXECUCAO == "paralelo":
        resultados = executar_filhos_paralelo(ORQUESTRADORES_FILHOS, TIMEOUT_SEGUNDOS)
    else:
        resultados = executar_filhos_sequencial(ORQUESTRADORES_FILHOS, TIMEOUT_SEGUNDOS)
    
    # Calcula estatísticas
    fim_execucao = datetime.now()
    duracao_total = (fim_execucao - inicio_execucao).total_seconds()
    sucessos = sum(1 for r in resultados if r.status == StatusExecucao.SUCESSO.value)
    
    # Gera relatório TXT
    caminho_relatorios = diretorio_script / PASTA_RELATORIOS
    caminho_relatorio = gerar_relatorio_txt(resultados, duracao_total, str(caminho_relatorios))
    
    # Prepara resumo para e-mail
    resumo = f"""
Resumo da Execução:
- Total de orquestradores: {len(resultados)}
- Sucessos: {sucessos}
- Falhas: {len(resultados) - sucessos}
- Duração total: {duracao_total:.2f} segundos
"""
    
    # Envia e-mail se configurado
    email_enviado = False
    if ENVIAR_EMAIL and DESTINATARIOS_EMAIL:
        data_atual = datetime.now().strftime("%d/%m/%Y")
        assunto = ASSUNTO_EMAIL.replace("{data}", data_atual)
        corpo = CORPO_EMAIL.replace("{resumo}", resumo)
        
        email_enviado = enviar_email_outlook(
            destinatarios=DESTINATARIOS_EMAIL,
            destinatarios_cc=DESTINATARIOS_CC,
            assunto=assunto,
            corpo=corpo,
            anexo=caminho_relatorio
        )
        
        if not email_enviado:
            logging.warning(
                f"Falha no envio de e-mail. "
                f"Relatório salvo em: {caminho_relatorio}"
            )
    
    # Log final
    logging.info("=" * 60)
    logging.info("RESUMO DA EXECUÇÃO")
    logging.info("=" * 60)
    logging.info(f"Duração total: {duracao_total:.2f} segundos")
    logging.info(f"Orquestradores com sucesso: {sucessos}/{len(resultados)}")
    logging.info(f"Relatório: {caminho_relatorio}")
    if ENVIAR_EMAIL:
        logging.info(f"E-mail enviado: {'Sim' if email_enviado else 'Não'}")
    logging.info("=" * 60)
    
    # Retorna código de saída
    if sucessos == len(resultados):
        return 0
    elif sucessos == 0:
        return 2
    else:
        return 1


def processar_argumentos() -> argparse.Namespace:
    """
    Processa os argumentos de linha de comando.
    
    Retorna:
        Namespace com os argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description="Orquestrador Pai - Gerenciador Central de Execução",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python orquestrador_pai.py                    # Execução normal
  python orquestrador_pai.py --teste Projeto_A  # Testa apenas um filho
  python orquestrador_pai.py --agendar          # Configura agendamento
  python orquestrador_pai.py --listar           # Lista filhos configurados
        """
    )
    
    parser.add_argument(
        "--teste",
        metavar="NOME",
        help="Executa apenas o orquestrador filho especificado (modo de depuração)"
    )
    
    parser.add_argument(
        "--agendar",
        "--schedule",
        action="store_true",
        help="Configura agendamento automático no Task Scheduler do Windows"
    )
    
    parser.add_argument(
        "--listar",
        action="store_true",
        help="Lista todos os orquestradores filhos configurados"
    )
    
    return parser.parse_args()


# ==============================================================================
# PONTO DE ENTRADA PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    args = processar_argumentos()
    
    try:
        # Modo listar
        if args.listar:
            print("\nOrquestradores filhos configurados:")
            print("-" * 40)
            if ORQUESTRADORES_FILHOS:
                for i, filho in enumerate(ORQUESTRADORES_FILHOS, 1):
                    print(f"{i}. {filho['nome']}")
                    print(f"   Caminho: {filho['caminho']}")
                    if filho.get('argumentos'):
                        print(f"   Args: {filho['argumentos']}")
            else:
                print("(Nenhum orquestrador configurado)")
            sys.exit(0)
        
        # Modo agendar
        if args.agendar:
            configurar_logging(PASTA_LOGS, f"{NOME_ORQUESTRADOR}_agendamento")
            sucesso = criar_tarefa_agendada()
            sys.exit(0 if sucesso else 1)
        
        # Modo teste
        if args.teste:
            executar_modo_teste(args.teste)
            sys.exit(0)
        
        # Execução normal
        codigo_saida = executar_orquestrador()
        sys.exit(codigo_saida)
        
    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário.")
        sys.exit(130)
        
    except Exception as e:
        print(f"ERRO FATAL: {type(e).__name__}: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(99)
