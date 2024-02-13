from binance.client import Client
import pandas as pd
import numpy as np
import time
from datetime import datetime
from API import api_key0, api_secret0

# Insira suas chaves de API
api_key = api_key0
api_secret = api_secret0

# Conexão com a API da Binance
client = Client(api_key, api_secret)

def calcular_macd(df, short_period=12, long_period=26, signal=9):
    """Calcula o MACD e a linha de sinal."""
    df['EMA_curto'] = df['close'].ewm(span=short_period, adjust=False).mean()
    df['EMA_longo'] = df['close'].ewm(span=long_period, adjust=False).mean()
    df['MACD'] = df['EMA_curto'] - df['EMA_longo']
    df['Signal_line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    return df

def verificar_condicoes_entrada(df):
    """Verifica condições de entrada baseadas no MACD."""
    ultimo_macd = df.iloc[-1]['MACD']
    ultimo_signal = df.iloc[-1]['Signal_line']
    if ultimo_macd > ultimo_signal:
        return 'LONG'
    elif ultimo_macd < ultimo_signal:
        return 'SHORT'
    return None

def executar_ordem_com_stop_loss_e_take_profit(client, side, symbol="BTCUSDT", quantity=0.00003, leverage=5):
    """Executa uma ordem de mercado, então define stop loss e take profit."""
    client.futures_change_leverage(symbol=symbol, leverage=leverage)
    market_order = client.futures_create_order(symbol=symbol, side=side, type='MARKET', quantity=quantity)
    execution_price = float(market_order['fills'][0]['price'])

    if side == 'BUY':
        stop_loss_price = execution_price * 0.98
        take_profit_price = execution_price * 1.02
    else:
        stop_loss_price = execution_price * 1.02
        take_profit_price = execution_price * 0.98
    stop_side = 'SELL' if side == 'BUY' else 'BUY'

    print(f"Ordem de {side} executada. Preço de execução: {execution_price}")
    print(f"Stop Loss: {stop_loss_price}, Take Profit: {take_profit_price}")

    stop_loss_order = client.futures_create_order(
        symbol=symbol,
        side=stop_side,
        type='STOP_MARKET',
        stopPrice=round(stop_loss_price, 2),
        closePosition='true'
    )

    take_profit_order = client.futures_create_order(
        symbol=symbol,
        side=stop_side,
        type='TAKE_PROFIT_MARKET',
        stopPrice=round(take_profit_price, 2),
        closePosition='true'
    )

    return market_order, stop_loss_order, take_profit_order

print("Conectado à API da Binance.")

while True:
    try:
        klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "1 day ago UTC")
        df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['close'] = pd.to_numeric(df['close'])
        df = calcular_macd(df)

        condicao_entrada = verificar_condicoes_entrada(df)
        if condicao_entrada:
            # Aqui você pode ajustar 'quantity' e 'leverage' conforme necessário
            executar_ordem_com_stop_loss_e_take_profit(client, 'BUY' if condicao_entrada == 'LONG' else 'SELL', symbol="BTCUSDT", quantity=0.00003, leverage=5)
            time.sleep(90)  # Aguarda 90 sec antes da próxima checagem para evitar excesso de operações
    except Exception as e:
        print(f"Erro na execução: {e}")
        break

    time.sleep(60)  # Intervalo antes de verificar novamente
