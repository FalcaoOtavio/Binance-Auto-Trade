import subprocess
import os
import sys

ativos = {
    1: "./Binance-Trade-Auto-BTC/IA.py",
    2: "./Binance-Trade-Auto-BEAMX/IA.py",
    3: "./Binance-Trade-Auto-DOGE/IA.py",
    4: "./Binance-Trade-Auto-XMR/IA.py"
}

escolha = int(input("""
        Escolha qual ativo será analisado:
        1- BTC/USDT
        2- BEAMX/USDT
        3- DOGE/USDT
        4- XMR/USDT
"""))

if escolha in ativos:
    arquivo_ia = ativos[escolha]
    
    if os.name == 'nt':  # Sistema operacional Windows
        subprocess.Popen(["python3", arquivo_ia], creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        # Sistema operacional Linux or MacOS
        subprocess.Popen(["python3", arquivo_ia])
else:
    print("Invalid! Retry!")
