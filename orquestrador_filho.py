# -*- coding: utf-8 -*-
"""
================================================================================
ORQUESTRADOR FILHO - Template Versátil para Execução de Scripts Python
================================================================================

DESCRIÇÃO:
    Este arquivo é um template que pode ser copiado para qualquer projeto que
    necessite de execução orquestrada de múltiplos scripts Python. Ele executa
    uma lista de scripts sequencialmente, captura erros detalhados, gera logs
    e retorna um relatório estruturado para o orquestrador pai.

DEPENDÊNCIAS:
    Este script utiliza apenas bibliotecas padrão do Python.
    Caso precise instalar pacotes adicionais, use sempre:
    
    pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <pacote>

COMO USAR:
    1. Copie este arquivo para a pasta do seu projeto
    2. Edite a seção "CONFIGURAÇÃO DO USUÁRIO" abaixo
    3. Adicione os caminhos dos seus scripts na lista SCRIPTS_A_EXECUTAR
    4. Execute: python orquestrador_filho.py

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
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

# ==============================================================================
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        CONFIGURAÇÃO DO USUÁRIO                            ║
# ║                                                                           ║
# ║  Edite apenas esta seção para configurar o orquestrador para seu projeto  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# ==============================================================================

# Lista de scripts a serem executados, na ordem desejada.
# Formato: Cada item pode ser:
#   - Uma string com o caminho do script (absoluto ou relativo à pasta deste arquivo)
#   - Um dicionário com 'caminho' e opcionalmente 'argumentos'
#
# Exemplos:
#   SCRIPTS_A_EXECUTAR = [
#       "script1.py",                              # script na mesma pasta
#       "pasta/script2.py",                        # script em subpasta
#       {"caminho": "script3.py", "argumentos": ["--verbose"]},  # com argumentos
#       r"C:\Projetos\MeuProjeto\script_especial.py",            # caminho absoluto
#   ]

SCRIPTS_A_EXECUTAR: list = [
    # Adicione seus scripts aqui (remova o # do início da linha para ativar)
    # "exemplo_script_1.py",
    # "pasta/exemplo_script_2.py",
    # {"caminho": "exemplo_script_3.py", "argumentos": ["--modo", "producao"]},
]

# Tempo máximo de execução para cada script (em segundos)
# Se um script demorar mais que isso, será forçadamente encerrado
TIMEOUT_SEGUNDOS: int = 300  # 5 minutos por padrão

# Comportamento quando um script falhar
# Opções:
#   "parar"     - Para a execução imediatamente ao encontrar erro
#   "continuar" - Registra o erro e continua para o próximo script
#   "reiniciar" - Tenta executar o script novamente (máximo 3 tentativas)
COMPORTAMENTO_EM_ERRO: str = "continuar"

# Número máximo de tentativas quando COMPORTAMENTO_EM_ERRO = "reiniciar"
MAX_TENTATIVAS: int = 3

# Nome identificador deste orquestrador (usado nos logs e relatórios)
NOME_ORQUESTRADOR: str = "OrquestradorFilho"

# Pasta onde os logs serão salvos (relativa ao diretório do script)
PASTA_LOGS: str = "logs"

# ==============================================================================
# FIM DA CONFIGURAÇÃO DO USUÁRIO - NÃO EDITE ABAIXO DESTA LINHA
# ==============================================================================

# Valores válidos para COMPORTAMENTO_EM_ERRO
_COMPORTAMENTOS_VALIDOS = {"parar", "continuar", "reiniciar"}


class StatusExecucao(Enum):
    """
    Lista dos possíveis resultados de execução de um script.
    (Enum = lista fixa de opções que não muda)
    """
    SUCESSO = "sucesso"
    FALHA = "falha"
    TIMEOUT = "timeout"
    NAO_ENCONTRADO = "nao_encontrado"
    ERRO_PERMISSAO = "erro_permissao"


class StatusGeral(Enum):
    """
    Lista dos possíveis resultados gerais do orquestrador.
    (Enum = lista fixa de opções que não muda)
    """
    SUCESSO_TOTAL = "sucesso_total"         # Todos os scripts executaram com sucesso
    SUCESSO_PARCIAL = "sucesso_parcial"     # Alguns scripts falharam, outros tiveram sucesso
    FALHA_TOTAL = "falha_total"             # Todos os scripts falharam
    NAO_EXECUTADO = "nao_executado"         # Nenhum script foi executado


@dataclass
class ResultadoScript:
    """
    Estrutura que armazena o resultado da execução de um único script.
    
    Atributos:
        nome_script: Nome ou caminho do script executado
        status: Status da execução (sucesso, falha, timeout, etc.)
        inicio: Data/hora de início da execução
        fim: Data/hora de término da execução
        duracao_segundos: Tempo total de execução em segundos
        codigo_saida: Código de retorno do processo (0 = sucesso)
        saida_padrao: Texto capturado da saída padrão (stdout)
        saida_erro: Texto capturado da saída de erro (stderr)
        erro_tipo: Tipo da exceção (se houver)
        erro_mensagem: Mensagem de erro detalhada (se houver)
        erro_traceback: Traceback completo formatado (se houver)
        tentativas: Número de tentativas realizadas
    """
    nome_script: str
    status: str
    inicio: str
    fim: str
    duracao_segundos: float
    codigo_saida: int | None = None
    saida_padrao: str = ""
    saida_erro: str = ""
    erro_tipo: str | None = None
    erro_mensagem: str | None = None
    erro_traceback: str | None = None
    tentativas: int = 1


@dataclass
class ResultadoOrquestrador:
    """
    Estrutura que armazena o resultado geral da execução do orquestrador.
    
    Atributos:
        nome_orquestrador: Nome identificador do orquestrador
        status_geral: Status consolidado de toda a execução
        inicio_execucao: Data/hora de início do orquestrador
        fim_execucao: Data/hora de término do orquestrador
        duracao_total_segundos: Tempo total de execução
        total_scripts: Quantidade total de scripts configurados
        scripts_sucesso: Quantidade de scripts que executaram com sucesso
        scripts_falha: Quantidade de scripts que falharam
        resultados_scripts: Lista detalhada com resultado de cada script
        caminho_log: Caminho do arquivo de log gerado
    """
    nome_orquestrador: str
    status_geral: str
    inicio_execucao: str
    fim_execucao: str
    duracao_total_segundos: float
    total_scripts: int
    scripts_sucesso: int
    scripts_falha: int
    resultados_scripts: list[dict]
    caminho_log: str


def configurar_logging(pasta_logs: str, nome_orquestrador: str) -> str:
    """
    Configura o sistema de logging do orquestrador.
    
    Esta função cria a pasta de logs (se não existir) e configura o logger
    para registrar tanto em arquivo quanto no console (para depuração).
    
    Parâmetros:
        pasta_logs: Caminho da pasta onde os logs serão salvos
        nome_orquestrador: Nome do orquestrador para identificação
        
    Retorna:
        Caminho completo do arquivo de log criado
    """
    # Obtém o diretório onde este script está localizado
    diretorio_script = Path(__file__).parent.absolute()
    caminho_pasta_logs = diretorio_script / pasta_logs
    
    # Cria a pasta de logs se não existir
    caminho_pasta_logs.mkdir(parents=True, exist_ok=True)
    
    # Gera nome do arquivo de log com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo_log = f"{nome_orquestrador}_{timestamp}.log"
    caminho_log = caminho_pasta_logs / nome_arquivo_log
    
    # Configura o formato das mensagens de log
    formato_log = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configura handler para arquivo
    handler_arquivo = logging.FileHandler(
        filename=str(caminho_log),
        mode='w',
        encoding='utf-8'
    )
    handler_arquivo.setFormatter(formato_log)
    handler_arquivo.setLevel(logging.DEBUG)
    
    # Configura o logger raiz
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Remove handlers existentes para evitar duplicação
    logger.handlers.clear()
    logger.addHandler(handler_arquivo)
    
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


def normalizar_configuracao_script(script) -> dict:
    """
    Normaliza a configuração de um script para um formato padrão.
    
    Esta função aceita tanto strings simples quanto dicionários e retorna
    sempre um dicionário com as chaves 'caminho' e 'argumentos'.
    
    Parâmetros:
        script: Pode ser uma string com o caminho ou um dicionário
        
    Retorna:
        Dicionário com 'caminho' (str) e 'argumentos' (list)
        
    Exemplos:
        >>> normalizar_configuracao_script("meu_script.py")
        {'caminho': 'meu_script.py', 'argumentos': []}
        
        >>> normalizar_configuracao_script({"caminho": "script.py", "argumentos": ["--verbose"]})
        {'caminho': 'script.py', 'argumentos': ['--verbose']}
    """
    if isinstance(script, str):
        return {"caminho": script, "argumentos": []}
    elif isinstance(script, dict):
        if "caminho" not in script:
            raise ValueError(
                f"Configuração de script inválida: falta a chave 'caminho'. "
                f"Use o formato: {{\"caminho\": \"script.py\"}}"
            )
        return {
            "caminho": script["caminho"],
            "argumentos": script.get("argumentos", [])
        }
    else:
        raise ValueError(
            f"Configuração de script inválida: {script}. "
            f"Use uma string (\"script.py\") ou dicionário ({{\"caminho\": \"script.py\"}})"
        )


def verificar_script_existe(caminho_script: str) -> tuple[bool, str]:
    """
    Verifica se o arquivo de script existe e é acessível.
    
    Parâmetros:
        caminho_script: Caminho do script (absoluto ou relativo)
        
    Retorna:
        Tupla com (existe: bool, caminho_absoluto: str)
    """
    # Se o caminho for relativo, resolve a partir do diretório deste script
    if not os.path.isabs(caminho_script):
        diretorio_script = Path(__file__).parent.absolute()
        caminho_absoluto = diretorio_script / caminho_script
    else:
        caminho_absoluto = Path(caminho_script)
    
    return caminho_absoluto.exists(), str(caminho_absoluto)


def executar_script(
    caminho_script: str,
    argumentos: list[str],
    timeout_segundos: int
) -> ResultadoScript:
    """
    Executa um único script Python e captura seu resultado.
    
    Esta função executa o script em um subprocesso, capturando toda a saída
    e tratando possíveis erros como timeout, arquivo não encontrado, etc.
    A execução ocorre em segundo plano, sem abrir janelas visíveis no Windows.
    
    Parâmetros:
        caminho_script: Caminho absoluto do script a executar
        argumentos: Lista de argumentos de linha de comando
        timeout_segundos: Tempo máximo de execução
        
    Retorna:
        ResultadoScript com todos os detalhes da execução
    """
    inicio = datetime.now()
    nome_script = os.path.basename(caminho_script)
    
    logging.info(f"Iniciando execução: {nome_script}")
    logging.debug(f"Caminho completo: {caminho_script}")
    if argumentos:
        logging.debug(f"Argumentos: {argumentos}")
    
    # Verifica se o script existe
    existe, caminho_absoluto = verificar_script_existe(caminho_script)
    if not existe:
        logging.error(f"Script não encontrado: {caminho_script}")
        fim = datetime.now()
        return ResultadoScript(
            nome_script=nome_script,
            status=StatusExecucao.NAO_ENCONTRADO.value,
            inicio=inicio.isoformat(),
            fim=fim.isoformat(),
            duracao_segundos=(fim - inicio).total_seconds(),
            erro_tipo="FileNotFoundError",
            erro_mensagem=f"O arquivo não foi encontrado: {caminho_script}"
        )
    
    # Prepara o comando para execução
    # Usa sys.executable para garantir que usa o mesmo interpretador Python
    comando = [sys.executable, caminho_absoluto] + argumentos
    
    try:
        # Configura execução em segundo plano (sem janela visível)
        creation_flags, startupinfo = _configurar_subprocesso_windows()
        
        # Executa o subprocesso
        processo = subprocess.run(
            comando,
            capture_output=True,          # Captura stdout e stderr
            text=True,                    # Retorna strings ao invés de bytes
            timeout=timeout_segundos,     # Limite de tempo
            creationflags=creation_flags,
            startupinfo=startupinfo,
            cwd=os.path.dirname(caminho_absoluto)  # Executa no diretório do script
        )
        
        fim = datetime.now()
        duracao = (fim - inicio).total_seconds()
        
        # Determina o status baseado no código de saída
        if processo.returncode == 0:
            status = StatusExecucao.SUCESSO.value
            logging.info(f"Concluído com sucesso: {nome_script} ({duracao:.2f}s)")
        else:
            status = StatusExecucao.FALHA.value
            logging.warning(
                f"Falha na execução: {nome_script} "
                f"(código de saída: {processo.returncode})"
            )
            if processo.stderr:
                logging.error(f"Erro: {processo.stderr[:500]}")  # Limita tamanho do log
        
        return ResultadoScript(
            nome_script=nome_script,
            status=status,
            inicio=inicio.isoformat(),
            fim=fim.isoformat(),
            duracao_segundos=duracao,
            codigo_saida=processo.returncode,
            saida_padrao=processo.stdout,
            saida_erro=processo.stderr
        )
        
    except subprocess.TimeoutExpired:
        fim = datetime.now()
        logging.error(f"Timeout expirado: {nome_script} (>{timeout_segundos}s)")
        return ResultadoScript(
            nome_script=nome_script,
            status=StatusExecucao.TIMEOUT.value,
            inicio=inicio.isoformat(),
            fim=fim.isoformat(),
            duracao_segundos=timeout_segundos,
            erro_tipo="TimeoutError",
            erro_mensagem=f"Script excedeu o tempo limite de {timeout_segundos} segundos"
        )
        
    except PermissionError as e:
        fim = datetime.now()
        logging.error(f"Erro de permissão: {nome_script} - {str(e)}")
        return ResultadoScript(
            nome_script=nome_script,
            status=StatusExecucao.ERRO_PERMISSAO.value,
            inicio=inicio.isoformat(),
            fim=fim.isoformat(),
            duracao_segundos=(fim - inicio).total_seconds(),
            erro_tipo="PermissionError",
            erro_mensagem=str(e),
            erro_traceback=traceback.format_exc()
        )
        
    except Exception as e:
        fim = datetime.now()
        logging.error(f"Erro inesperado: {nome_script} - {type(e).__name__}: {str(e)}")
        return ResultadoScript(
            nome_script=nome_script,
            status=StatusExecucao.FALHA.value,
            inicio=inicio.isoformat(),
            fim=fim.isoformat(),
            duracao_segundos=(fim - inicio).total_seconds(),
            erro_tipo=type(e).__name__,
            erro_mensagem=str(e),
            erro_traceback=traceback.format_exc()
        )


def executar_com_tentativas(
    config_script: dict,
    timeout_segundos: int,
    max_tentativas: int
) -> ResultadoScript:
    """
    Executa um script com possibilidade de múltiplas tentativas.
    
    Esta função tenta executar o script múltiplas vezes em caso de falha,
    quando o comportamento em erro está configurado como "reiniciar".
    
    Parâmetros:
        config_script: Dicionário com 'caminho' e 'argumentos'
        timeout_segundos: Tempo limite para cada execução
        max_tentativas: Número máximo de tentativas
        
    Retorna:
        ResultadoScript da última tentativa (ou primeira bem-sucedida)
    """
    resultado = None
    
    for tentativa in range(1, max_tentativas + 1):
        if tentativa > 1:
            logging.info(f"Tentativa {tentativa}/{max_tentativas}: {config_script['caminho']}")
        
        resultado = executar_script(
            caminho_script=config_script["caminho"],
            argumentos=config_script["argumentos"],
            timeout_segundos=timeout_segundos
        )
        resultado.tentativas = tentativa
        
        # Se teve sucesso, retorna imediatamente
        if resultado.status == StatusExecucao.SUCESSO.value:
            return resultado
    
    # Se chegou aqui, todas as tentativas falharam
    if max_tentativas > 1:
        logging.warning(
            f"Todas as {max_tentativas} tentativas falharam: "
            f"{config_script['caminho']}"
        )
    
    return resultado


def calcular_status_geral(resultados: list[ResultadoScript]) -> str:
    """
    Calcula o status geral da execução baseado nos resultados individuais.
    
    Parâmetros:
        resultados: Lista de ResultadoScript de cada script executado
        
    Retorna:
        String com o status geral (sucesso_total, sucesso_parcial, falha_total)
    """
    if not resultados:
        return StatusGeral.NAO_EXECUTADO.value
    
    sucessos = sum(1 for r in resultados if r.status == StatusExecucao.SUCESSO.value)
    total = len(resultados)
    
    if sucessos == total:
        return StatusGeral.SUCESSO_TOTAL.value
    elif sucessos == 0:
        return StatusGeral.FALHA_TOTAL.value
    else:
        return StatusGeral.SUCESSO_PARCIAL.value


def salvar_resultado_json(resultado: ResultadoOrquestrador) -> str:
    """
    Salva o resultado da execução em um arquivo JSON.
    
    O arquivo JSON é salvo no mesmo diretório dos logs e pode ser lido
    pelo orquestrador pai para processar os resultados.
    
    Parâmetros:
        resultado: Objeto ResultadoOrquestrador com todos os dados
        
    Retorna:
        Caminho do arquivo JSON gerado
    """
    # Converte o resultado para dicionário
    dados = asdict(resultado)
    
    # Define o caminho do arquivo JSON
    diretorio_log = os.path.dirname(resultado.caminho_log)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"{resultado.nome_orquestrador}_resultado_{timestamp}.json"
    caminho_json = os.path.join(diretorio_log, nome_arquivo)
    
    # Salva o arquivo
    with open(caminho_json, 'w', encoding='utf-8') as arquivo:
        json.dump(dados, arquivo, ensure_ascii=False, indent=2)
    
    logging.info(f"Resultado salvo em: {caminho_json}")
    return caminho_json


def executar_orquestrador() -> ResultadoOrquestrador:
    """
    Função principal que executa o orquestrador filho.
    
    Esta função coordena toda a execução:
    1. Configura o sistema de logging
    2. Processa a lista de scripts configurados
    3. Executa cada script sequencialmente
    4. Trata erros conforme comportamento configurado
    5. Gera o resultado final
    
    Retorna:
        ResultadoOrquestrador com todos os dados da execução
    """
    # Marca o início da execução
    inicio_execucao = datetime.now()
    
    # Configura o logging
    caminho_log = configurar_logging(PASTA_LOGS, NOME_ORQUESTRADOR)
    
    # Valida a configuração do comportamento em erro
    comportamento = COMPORTAMENTO_EM_ERRO.lower().strip()
    if comportamento not in _COMPORTAMENTOS_VALIDOS:
        logging.error(
            f"COMPORTAMENTO_EM_ERRO inválido: '{COMPORTAMENTO_EM_ERRO}'. "
            f"Opções válidas: {', '.join(_COMPORTAMENTOS_VALIDOS)}. "
            f"Usando 'continuar' como padrão."
        )
        comportamento = "continuar"
    
    logging.info("=" * 60)
    logging.info(f"INÍCIO DA EXECUÇÃO: {NOME_ORQUESTRADOR}")
    logging.info("=" * 60)
    logging.info(f"Total de scripts configurados: {len(SCRIPTS_A_EXECUTAR)}")
    logging.info(f"Timeout por script: {TIMEOUT_SEGUNDOS} segundos")
    logging.info(f"Comportamento em erro: {comportamento}")
    logging.info("-" * 60)
    
    # Verifica se há scripts configurados
    if not SCRIPTS_A_EXECUTAR:
        logging.warning(
            "Nenhum script configurado em SCRIPTS_A_EXECUTAR! "
            "Edite a seção CONFIGURAÇÃO DO USUÁRIO para adicionar scripts."
        )
    
    # Lista para armazenar os resultados de cada script
    resultados: list[ResultadoScript] = []
    
    # Processa cada script configurado
    for indice, script in enumerate(SCRIPTS_A_EXECUTAR, start=1):
        logging.info(f"Script {indice}/{len(SCRIPTS_A_EXECUTAR)}")
        
        try:
            # Normaliza a configuração do script
            config_script = normalizar_configuracao_script(script)
        except ValueError as e:
            logging.error(f"Configuração inválida: {e}")
            continue
        
        # Executa o script (com ou sem tentativas)
        if comportamento == "reiniciar":
            resultado = executar_com_tentativas(
                config_script=config_script,
                timeout_segundos=TIMEOUT_SEGUNDOS,
                max_tentativas=MAX_TENTATIVAS
            )
        else:
            resultado = executar_script(
                caminho_script=config_script["caminho"],
                argumentos=config_script["argumentos"],
                timeout_segundos=TIMEOUT_SEGUNDOS
            )
        
        resultados.append(resultado)
        
        # Verifica comportamento em erro
        if resultado.status != StatusExecucao.SUCESSO.value:
            if comportamento == "parar":
                logging.warning("Execução interrompida devido a erro (modo: parar)")
                break
            elif comportamento == "continuar":
                logging.info("Continuando para o próximo script (modo: continuar)")
        
        logging.info("-" * 60)
    
    # Calcula estatísticas finais
    fim_execucao = datetime.now()
    duracao_total = (fim_execucao - inicio_execucao).total_seconds()
    scripts_sucesso = sum(1 for r in resultados if r.status == StatusExecucao.SUCESSO.value)
    scripts_falha = len(resultados) - scripts_sucesso
    status_geral = calcular_status_geral(resultados)
    
    # Cria o resultado final
    resultado_final = ResultadoOrquestrador(
        nome_orquestrador=NOME_ORQUESTRADOR,
        status_geral=status_geral,
        inicio_execucao=inicio_execucao.isoformat(),
        fim_execucao=fim_execucao.isoformat(),
        duracao_total_segundos=duracao_total,
        total_scripts=len(SCRIPTS_A_EXECUTAR),
        scripts_sucesso=scripts_sucesso,
        scripts_falha=scripts_falha,
        resultados_scripts=[asdict(r) for r in resultados],
        caminho_log=caminho_log
    )
    
    # Loga o resumo final
    logging.info("=" * 60)
    logging.info("RESUMO DA EXECUÇÃO")
    logging.info("=" * 60)
    logging.info(f"Status geral: {status_geral}")
    logging.info(f"Duração total: {duracao_total:.2f} segundos")
    logging.info(f"Scripts com sucesso: {scripts_sucesso}/{len(SCRIPTS_A_EXECUTAR)}")
    logging.info(f"Scripts com falha: {scripts_falha}/{len(SCRIPTS_A_EXECUTAR)}")
    logging.info("=" * 60)
    
    # Salva o resultado em JSON
    caminho_json = salvar_resultado_json(resultado_final)
    
    # Imprime o caminho do JSON para stdout (para o orquestrador pai capturar)
    print(caminho_json)
    
    return resultado_final


# ==============================================================================
# PONTO DE ENTRADA PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    """
    Ponto de entrada quando o script é executado diretamente.
    
    Executa o orquestrador e retorna o código de saída apropriado:
    - 0: Todos os scripts executaram com sucesso
    - 1: Pelo menos um script falhou
    - 2: Nenhum script foi executado
    """
    try:
        resultado = executar_orquestrador()
        
        # Define código de saída baseado no status
        if resultado.status_geral == StatusGeral.SUCESSO_TOTAL.value:
            sys.exit(0)
        elif resultado.status_geral == StatusGeral.NAO_EXECUTADO.value:
            sys.exit(2)
        else:
            sys.exit(1)
            
    except Exception as e:
        # Captura qualquer erro não tratado
        print(f"ERRO FATAL: {type(e).__name__}: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(99)
