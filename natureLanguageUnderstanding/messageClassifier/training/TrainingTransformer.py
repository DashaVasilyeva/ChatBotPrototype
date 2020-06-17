from sklearn.preprocessing import LabelEncoder
import tensorflow.keras as keras
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Dropout
from tensorflow.python.keras.utils import np_utils
import numpy as np

from natureLanguageUnderstanding.dataHandlers.DataHandler import DataHandler

import matplotlib.pyplot as plt

from natureLanguageUnderstanding.messageClassifier.model.Transformer.Transformer import  \
    TransformerBlock, PreProcessingLayer

# random seed
seed = 10
np.random.seed(seed)

# Загружаем данные для обучения
messages, intents, unique_intents = DataHandler.load_data('../../../data/intents_dataset.csv')
uniq = list(set(intents))
print(uniq)
print(len(uniq))


# Преобразовываем класс в виде слова в число
encoder = LabelEncoder()
encoder.fit(intents)
encoded_intents = encoder.transform(intents)
print(encoded_intents)


# Используем one-hot для категорий
Y_train = np_utils.to_categorical(encoded_intents)
X = DataHandler.get_preprocessed_messages(messages)

# Размер словаря (количество возможных слов)
vocabulary_size = 5000


# Максимальная длина входной последовательности
max_input_length = 1000

# Количество классов
classes_quantity = len(uniq)

tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(X)
X_temp = tokenizer.texts_to_sequences(X)
X_train = pad_sequences(X_temp, padding='post', maxlen=max_input_length)

embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = Input(shape=(max_input_length,))
embedding_layer = PreProcessingLayer(embed_dim, vocabulary_size)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = Dense(20, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(classes_quantity, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Обучение
# epoch - максимальное количество эпох до остановки
# batch_size - сколько объектов будет загружаться за итерацию

epochs = 50
batch_size = 1
print('here')
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.000001)], verbose=2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Точность модели')
plt.ylabel('Точность')
plt.xlabel('Эпоха')
plt.legend(['Обучение', 'Тестирование'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Ошибка модели')
plt.ylabel('Ошибка')
plt.xlabel('Эпоха')
plt.legend(['Обучение', 'Тестирование'], loc='upper left')
plt.show()

# serialize model to JSON
model_json = model.to_json()
with open("../../../resources/modelTransformer.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../../../resources/modelTransformer.h5")
print("Saved model to disk")