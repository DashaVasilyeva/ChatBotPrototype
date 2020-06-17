import re

import pandas
import pymorphy2
import spacy
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

ENGLISH_LANGUAGE = 'en'
RUSSIAN_LANGUAGE = 'ru'

ENGLISH_STOP_WORDS = set(stopwords.words('english'))
RUSSIAN_STOP_WORDS = set(stopwords.words('russian'))


class DataHandler:

    # Загрузка датасета из файла
    @staticmethod
    def load_data(filename):
        data_frame = pandas.read_csv(filename, encoding="utf-8",
                                     names=['Intent', 'Message'])
        intents = data_frame['Intent']
        unique_intents = list(set(intents))
        messages = list(data_frame['Message'])

        return messages, intents, unique_intents

    # удаляем знаки пунктуации и специальные символы
    @staticmethod
    def get_cleaned_message(message):
        cleaned_message = re.sub(r'[^ a-я А-Я a-z A-Z 0-9]', " ", message)

        return cleaned_message

    # токенизация
    @staticmethod
    def get_tokenized_words(message):
        tokenized_words = word_tokenize(message)

        return tokenized_words

    @staticmethod
    def get_filtered_tokens(tokens, language):

        filtered_tokens = []
        for token in tokens:
            if (language == RUSSIAN_LANGUAGE) and (not str(token).isspace()) and \
                    (str(token).lower() not in RUSSIAN_STOP_WORDS):
                filtered_tokens.append(token)

            if (language == ENGLISH_LANGUAGE) and (not str(token).isspace()) and \
                    (str(token).lower() not in ENGLISH_STOP_WORDS):
                if (not str(token).isdigit()):
                    filtered_tokens.append(token)
        return filtered_tokens

    @staticmethod
    def get_filtered_tokens2(tokens, language):

        filtered_tokens = []
        for token in tokens:

            if not str(token).isspace():
                filtered_tokens.append(token)

        return filtered_tokens


    @staticmethod
    def get_lemmatized_tokens(tokens, language):

        russian_analyzer = pymorphy2.MorphAnalyzer()

        lemmatized_tokens = []
        for token in tokens:
            if language == RUSSIAN_LANGUAGE:
                lemmatized_word_information = russian_analyzer.parse(token.lower())
                lemmatized_tokens.append(lemmatized_word_information.normal_form)

            if language == ENGLISH_LANGUAGE:
                lemmatized_tokens.append(token.lemma_)

        return lemmatized_tokens

    @staticmethod
    def get_preprocessed_message(message):
        english_analyzer = spacy.load('en_core_web_sm')

        lemmatized_tokens = []

        language = detect(message)

        cleaned_message = DataHandler.get_cleaned_message(message)

        if language == RUSSIAN_LANGUAGE:
            tokenized_words = DataHandler.get_tokenized_words(cleaned_message)
            filtered_tokens = DataHandler.get_filtered_tokens(tokenized_words, language)
            lemmatized_tokens = DataHandler.get_lemmatized_tokens(filtered_tokens, language)

        elif language == ENGLISH_LANGUAGE:
            tokenized_words = english_analyzer(cleaned_message)
            filtered_tokens = DataHandler.get_filtered_tokens2(tokenized_words, language)
            lemmatized_tokens = DataHandler.get_lemmatized_tokens(filtered_tokens, language)

        return lemmatized_tokens

    @staticmethod
    def get_preprocessed_messages(messages):
        result = []
        for message in messages:
            result.append(DataHandler.get_preprocessed_message(message))
        return result


