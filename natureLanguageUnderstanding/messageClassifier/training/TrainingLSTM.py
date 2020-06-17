from sklearn.preprocessing import LabelEncoder
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import np_utils

from natureLanguageUnderstanding.messageClassifier.model.LSTM.LongShortTermMemory import LongShortTermMemory
from natureLanguageUnderstanding.dataHandlers.DataHandler import DataHandler

import matplotlib.pyplot as plt
import numpy as np

# random seed
seed = 10
np.random.seed(seed)
print('Загрузка данных')

# Загружаем данные для обучения
messages, intents, unique_intents = DataHandler.load_data('../../../data/data1.data')
uniq = list(set(intents))

# Преобразовываем класс в виде слова в число
encoder = LabelEncoder()
encoder.fit(intents)
encoded_intents = encoder.transform(intents)

# Используем one-hot для категорий
Y_train = np_utils.to_categorical(encoded_intents)
X = DataHandler.get_preprocessed_messages(messages)

# Размер словаря
vocabulary_size = 5000

# Количество блоков LSTM
output_dimension = 80

# Максимальная длина входной последовательности
max_input_length = 1000

# Количество классов
classes_quantity = len(uniq)
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(X)
X_temp = tokenizer.texts_to_sequences(X)
X_train = pad_sequences(X_temp, padding='post', maxlen=max_input_length)

model = LongShortTermMemory.build(
    vocabulary_size,
    output_dimension,
    max_input_length,
    classes_quantity
)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение
# epoch - максимальное количество эпох до остановки
# batch_size - сколько объектов будет загружаться за итерацию
epochs = 9
batch_size = 10

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)], verbose=2)


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

model_json = model.to_json()
with open("../../../resources/model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("../../../resources/model.h5")
print("Saved model to disk")