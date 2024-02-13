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
    candles = binance.fetch_ohlcv('XMR/USDT', timeframe='1h', limit=100)
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = calcular_macd(df)
    return df

def preparar_dados(df):
    df = df.dropna(subset=['MACD', 'Signal'])
    X = df[['MACD', 'Signal']].values
    return X

def obter_feedback(model, X):
    latest_data = X[-1].reshape(1, -1)
    prediction = model.predict(latest_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    decisions = ['esperar', 'comprar (long)', 'vender (short)']
    print(f"Decisão recomendada: {decisions[predicted_class]}")
    feedback = int(input("O modelo acertou? 1 para Sim, 2 para Não: "))
    return feedback, predicted_class

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
    model = load_model('./files/ia.h5')
    feedback, predicted_class = obter_feedback(model, X)

    if feedback == 1:
        correct_label = predicted_class
    else:
        correct_label = int(input("Qual era a ação correta? 0 para Esperar, 1 para Comprar, 2 para Vender: "))
    
    y = np.array([correct_label])
    y = to_categorical(y, num_classes=3)
    X = X[-1].reshape(1, -1)  # Usando apenas o último ponto de dados para simplificar
    model_retreinado = re_treinar_modelo(X, y)
    model_retreinado.save('./files/ia.h5')

if __name__ == '__main__':
    main()