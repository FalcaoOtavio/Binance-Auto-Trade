import ccxt
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def calcular_macd(df):
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def coletar_dados():
    binance = ccxt.binance()
    candles = binance.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=100)
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = calcular_macd(df)
    df_final = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'MACD', 'Signal']]
    csv_file = './Binance-Trade-Auto-BTC/files/binance_data_with_macd.csv'
    df_final.to_csv(csv_file, index=False)
    return df

def preparar_dados(df):
    df = df.dropna(subset=['MACD', 'Signal'])
    X = df[['MACD', 'Signal']].values
    return X

def define_action(row):
    if row['MACD'] > row['Signal']:  # Indica potencial sobrecompra
        return 2  # Ação: vender (short)
    elif row['MACD'] < row['Signal']:  # Indica potencial sobrevenda
        return 1  # Ação: comprar (long)
    else:
        return 0  # Ação: esperar

def obter_feedback(model, X):
    latest_data = X[-1].reshape(1, -1)
    prediction = model.predict(latest_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    decisions = ['esperar', 'comprar (long)', 'vender (short)']
    print(f"Decisão recomendada: {decisions[predicted_class]}")
    
    feedback = int(input("O modelo acertou? 1 para Sim, 2 para Não: "))
    
    if feedback == 1:
        # O modelo acertou, então retornamos a classe prevista pelo modelo.
        return predicted_class
    else:
        # O modelo errou, então pedimos ao usuário para especificar a ação correta.
        print("Qual é a ação correta?")
        for i, action in enumerate(decisions):
            print(f"{i}: {action}")
        correct_action = int(input("Escolha a ação correta (0: esperar, 1: comprar (long), 2: vender (short)): "))
        # Retornamos a ação correta especificada pelo usuário.
        return correct_action

def re_treinar_modelo(X, y):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(2,), kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    if len(X) > 1:  # Checa se há mais de uma amostra
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    else:
        model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    
    return model

def main():
    df_novos_dados = coletar_dados()
    X = preparar_dados(df_novos_dados)
    model = load_model('./Binance-Trade-Auto-BTC-15Min/files/ia.h5')
    
    # A função obter_feedback foi ajustada para retornar diretamente a ação correta
    correct_label = obter_feedback(model, X)

    y = np.array([correct_label])
    y = to_categorical(y, num_classes=3)
    X = X[-1].reshape(1, -1)  # Usando apenas o último ponto de dados para simplificar
    
    # Re-treinar o modelo com a entrada mais recente e a etiqueta corrigida baseada na ação correta
    model_retreinado = re_treinar_modelo(X, y)
    model_retreinado.save('./Binance-Trade-Auto-BTC-15Min/files/ia.h5')

if __name__ == '__main__':
    main()