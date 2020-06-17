import pymongo

OK_MESSAGE = 'ОК'
ERROR_MESSAGE = 'ERROR'


class DatabaseConnector:
    def __init__(self):
        self.connection = None
        self.database = None
        self.intent_collection = None
        self.message_collection = None

    def open_connection(self):
        self.connection = pymongo.MongoClient("localhost", 27017)
        self.database = self.connection['chatBotDB']
        self.intent_collection = self.database['Intent']
        self.message_collection = self.database['Message']

    def close_connection(self):
        self.connection.close()
        self.database = None
        self.intent_collection = None
        self.message_collection = None

    def load_intent_by_code(self, code, message_id):
        try:
            self.open_connection()
            intent = self.intent_collection.find_one({'code': code})
            found_message = self.message_collection.find_one({'_id': message_id})
            if (intent is not None) and (found_message is not None):
                self.message_collection.update({'_id': message_id}, {'$set': {'detected_intent_id': intent['_id']}})
            self.close_connection()
            return intent
        except:
           self.close_connection()
           return ERROR_MESSAGE

    def save_correct_intent_for_message(self, intent, message_id):
        try:
            self.open_connection()
            found_intent = self.intent_collection.find_one({'intent': intent})
            found_message = self.message_collection.find_one({'_id': message_id})
            if (found_intent is not None) and (found_message is not None):
                self.message_collection.update({'_id': message_id}, {'$set': {'correct_intent_id': found_intent['_id']}})
            self.close_connection()
            return intent
        except:
           self.close_connection()
           return ERROR_MESSAGE

    def save_message(self, message):
        try:
            self.open_connection()
            _id = self.message_collection.insert(message)
            self.close_connection()
            return _id
        except:
           self.close_connection()
           return ERROR_MESSAGE

