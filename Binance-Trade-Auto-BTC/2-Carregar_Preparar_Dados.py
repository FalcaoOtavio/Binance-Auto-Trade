import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Carrega os dados
df = pd.read_csv('./Binance-Trade-Auto-BTC/files/binance_data_with_macd.csv')

# Define a ação baseada nas condições do MACD e Signal
def define_action(row):
    if row['MACD'] > row['Signal']:  # Condição de sobrecompra
        return 2  # vender
    elif row['MACD'] < row['Signal']:  # Condição de sobrevenda
        return 1  # comprar
    else:
        return 0  # esperar

df['action'] = df.apply(define_action, axis=1)

# Separa os dados em características (X) e rótulo (y)
X = df[['MACD', 'Signal']].values
y = df['action'].values

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aqui está a correção: Converte os arrays NumPy em DataFrames antes de salvar
df_X_train = pd.DataFrame(X_train)
df_X_test = pd.DataFrame(X_test)
df_y_train = pd.DataFrame(y_train, columns=['action'])
df_y_test = pd.DataFrame(y_test, columns=['action'])

# Salva os DataFrames em arquivos CSV
df_X_train.to_csv('./Binance-Trade-Auto-BTC/X_train.csv', index=False)
df_X_test.to_csv('./Binance-Trade-Auto-BTC/X_test.csv', index=False)
df_y_train.to_csv('./Binance-Trade-Auto-BTC/y_train.csv', index=False)
df_y_test.to_csv('./Binance-Trade-Auto-BTC/y_test.csv', index=False)

print("Dados Carregados e Preparados com SUCESSO!!")