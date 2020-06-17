import configparser
import telebot
from dialogManagement.databaseConnector.DatabaseConnector import DatabaseConnector

config = configparser.ConfigParser()
config.read('../../resources/configfile.txt')

bot = telebot.TeleBot(config['Telegram']['TelegramBotToken'])

INFORM_MESSAGE = 'Test mode. Please send a message to determine your intent.'
database_connector = DatabaseConnector()
ERROR_MESSAGE = 'ERROR'

message_id = None

@bot.message_handler(commands=['start'])
def send_welcome_and_inform(message):
    bot.send_message(message.from_user.id, INFORM_MESSAGE)


@bot.message_handler(content_types=['text'])
def get_text_message(message):
    result = database_connector.save_message(message.json)
    if result != ERROR_MESSAGE:
        bot.send_message(message.from_user.id,  'Bot received your message')
        global message_id
        message_id = result
    else:
        bot.send_message(message.from_user.id, ERROR_MESSAGE)
    

bot.polling(none_stop=True, interval=0)