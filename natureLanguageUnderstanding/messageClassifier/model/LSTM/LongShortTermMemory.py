from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Activation, Dense, \
    SpatialDropout1D, Bidirectional

DROPOUT_25 = 0.25
DROPOUT_45 = 0.45


class LongShortTermMemory:

    # vocabulary_size - размер словаря
    # output_dimension - количество блоков LSTM
    # max_input_length - максимальная длина входной последовательности
    # classes_quantity - количество классов

    @staticmethod
    def build(vocabulary_size, output_dimension, max_input_length, classes_quantity):
        model = Sequential()

        # vocabulary_size - максимальное количество уникальных слов в датасете
        # hidden_size - размерность векторов
        # input_length - длина максимального сообщения

        model.add(Embedding(input_dim=vocabulary_size, output_dim=output_dimension, input_length=max_input_length))
        model.add(SpatialDropout1D(DROPOUT_45))

        model.add(
            Bidirectional(LSTM(output_dimension, activation='tanh', recurrent_activation='sigmoid',
                               dropout=DROPOUT_25, recurrent_dropout=DROPOUT_25)))

        model.add(Dense(classes_quantity))
        model.add(Activation('softmax'))
        return model
