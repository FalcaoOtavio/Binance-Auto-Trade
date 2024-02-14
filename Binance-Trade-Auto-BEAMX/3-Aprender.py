from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Carrega os dados (assumindo que você já os preparou)
X_train = pd.read_csv('./files/X_train.csv').values
X_test = pd.read_csv('./files/X_test.csv').values
y_train = pd.read_csv('./files/y_train.csv').values
y_test = pd.read_csv('./files/y_test.csv').values

# Converte os rótulos para one-hot encoding
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# Configuração do modelo
model = Sequential([
    Dense(128, activation='relu', input_shape=(2,), kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Compila o modelo
optimizer = Adam(learning_rate=0.0001)  # Taxa de aprendizado menor
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Treina o modelo
history = model.fit(X_train, y_train, epochs=15000, batch_size=32, validation_split=0.2, verbose=1)  # Mais épocas

# Avalia o modelo
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Salva o modelo
model.save('./files/ia.h5')  # O modelo é salvo no arquivo meu_modelo.h5

print("Modelo salvo com sucesso.")