from API import api_key0, api_secret0
import ccxt
import pandas as pd

# Configuração da conexão com a API da Binance
binance = ccxt.binance()

# Define o par de moedas e o intervalo de tempo para os dados
symbol = 'BTC/USDT'  # Exemplo com Bitcoin vs USDT
timeframe = '1h'  # Dados a cada 1h

# Baixa os dados históricos
candles = binance.fetch_ohlcv(symbol, timeframe)
df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Calcula o MACD
# MACD = EMA de 12 períodos do preço de fechamento - EMA de 26 períodos do preço de fechamento
# Sinal = EMA de 9 períodos do MACD
df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA12'] - df['EMA26']
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Selecione as colunas relevantes para salvar
df_final = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'MACD', 'Signal']]

# Salva os dados em um arquivo CSV
csv_file = './Binance-Trade-Auto-BTC/files/binance_data_with_macd.csv'
df_final.to_csv(csv_file, index=False)

print(f'Dados salvos em {csv_file}.')
