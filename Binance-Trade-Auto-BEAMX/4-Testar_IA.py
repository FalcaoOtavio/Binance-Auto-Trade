from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

# Carregar o modelo treinado
model = load_model('./files/ia.h5')

# Carregar os dados existentes
df = pd.read_csv('./files/binance_data_with_macd.csv')

# Supondo que o pré-processamento necessário seja apenas a seleção das características
# Certifique-se de ajustar este código conforme necessário para corresponder ao seu pré-processamento exato
X = df[['MACD', 'Signal']].values

# Fazendo previsão para a última entrada do conjunto de dados
latest_data = X[-1].reshape(1, -1)  # Redimensiona para garantir que tenha a forma correta
prediction = model.predict(latest_data)
predicted_class = np.argmax(prediction, axis=1)[0]

# Mapeia a classe predita para a decisão
decisions = ['esperar', 'comprar (long)', 'vender (short)']
predicted_decision = decisions[predicted_class]
print(f"Decisão recomendada para a última entrada: {predicted_decision}")

# Coletar feedback do usuário
feedback = int(input("O modelo acertou? 1 para Sim, 2 para Não: "))
# Após coletar o feedback do usuário
if feedback == 1:
    feedback_label = 'Acertou'
else:
    feedback_label = 'Errou'

# Salvar a entrada, a previsão e o feedback
feedback_data = pd.DataFrame({
    'MACD': [X[-1][0]],
    'Signal': [X[-1][1]],
    'Previsão': [predicted_decision],
    'Feedback': [feedback_label]
})

# Supondo que você tenha um arquivo chamado feedback.csv para armazenar esses dados
# Se o arquivo não existir, ele será criado. Se existir, os dados serão adicionados.
feedback_file = 'feedback.csv'
with open(feedback_file, 'a') as f:
    feedback_data.to_csv(f, header=f.tell()==0, index=False)

print("Feedback salvo com sucesso.")