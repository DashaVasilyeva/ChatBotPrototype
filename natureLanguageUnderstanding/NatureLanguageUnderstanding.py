from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from tensorflow.python.keras.models import model_from_json
from natureLanguageUnderstanding.dataHandlers.DataHandler import DataHandler

import itertools
import operator


max_input_length = 1000
vocabulary_size = 5000


def most_common(L):
  SL = sorted((x, i) for i, x in enumerate(L))
  groups = itertools.groupby(SL, key=operator.itemgetter(0))

  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    return count, -min_index

  return max(groups, key=_auxfun)[0]


class NLU:
    def __init__(self):
        json_file = open('../../resources/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.intent_classifier = model_from_json(loaded_model_json)
        self.intent_classifier.load_weights('../../resources/model.h5')
        print("Loaded model from disk")


    def predict_intent(self, text):
        prepared_text = DataHandler.get_preprocessed_message(text)
        tokenizer = Tokenizer(num_words=vocabulary_size)
        tokenizer.fit_on_texts(prepared_text)
        X_temp = tokenizer.texts_to_sequences(prepared_text)
        X = pad_sequences(X_temp, padding='post', maxlen=max_input_length)
        print(X)
        result = self.intent_classifier.predict_classes(X)
        print("Res = " + str(result))
        print(most_common(result).item())
        return most_common(result).item()
