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
    df_final = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'MACD', 'Signal']]
    csv_file = './Binance-Trade-Auto-XMR/files/binance_data_with_macd.csv'
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
    model = load_model('./Binance-Trade-Auto-XMR/files/ia.h5')
    feedback, predicted_class = obter_feedback(model, X)

    # Aqui, você usa a última entrada de dados para aplicar a função define_action diretamente
    # Isto é mais uma validação da ação sugerida pelo modelo com a lógica do MACD
    row = df_novos_dados.iloc[-1]  # Pega a última linha do DataFrame
    correct_action = define_action(row)  # Define a ação correta baseada na lógica do MACD

    # Comparar a ação correta com a previsão do modelo e o feedback do usuário
    if feedback == 1:
        correct_label = predicted_class
    else:
        # Se o feedback for que o modelo errou, usa a ação correta definida pela lógica do MACD
        correct_label = correct_action
        print(f"A ação correta baseada na lógica do MACD seria: {['esperar', 'comprar (long)', 'vender (short)'][correct_label]}")

    y = np.array([correct_label])
    y = to_categorical(y, num_classes=3)
    X = X[-1].reshape(1, -1)  # Usando apenas o último ponto de dados para simplificar
    
    # Re-treinar o modelo com a entrada mais recente e a etiqueta corrigida baseada no feedback e na lógica do MACD
    model_retreinado = re_treinar_modelo(X, y)
    model_retreinado.save('./Binance-Trade-Auto-XMR/files/ia.h5')

if __name__ == '__main__':
    main()