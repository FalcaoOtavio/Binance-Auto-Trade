import subprocess
import os
import sys

# Mapeamento entre escolha do usuário e caminho do arquivo
ativos = {
    1: "./Binance-Trade-Auto-BTC/IA.py",
    2: "./Binance-Trade-Auto-BEAMX/IA.py",  # Certifique-se de que a extensão está correta
    3: "./Binance-Trade-Auto-DOGE/IA.py",
    4: "./Binance-Trade-Auto-XMR/IA.py"
}

# Solicita ao usuário a escolha do ativo
escolha = int(input("""
        Escolha qual ativo será analisado:
        1- BTC/USDT
        2- BEAMX/USDT
        3- DOGE/USDT
        4- XMR/USDT
"""))

# Verifica se a escolha é válida
if escolha in ativos:
    # Obtém o caminho do arquivo correspondente à escolha do usuário
    arquivo_ia = ativos[escolha]
    
    # Verifica o sistema operacional
    if os.name == 'nt':  # Sistema operacional Windows
        subprocess.Popen(["python3", arquivo_ia], creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        # Para sistemas operacionais Unix-like (Linux, MacOS), não use creationflags
        subprocess.Popen(["python3", arquivo_ia])
else:
    print("Escolha inválida. Por favor, escolha uma opção válida.")
