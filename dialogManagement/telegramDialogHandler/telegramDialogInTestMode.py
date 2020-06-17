import configparser
import telebot

config = configparser.ConfigParser()
config.read('../../resources/main_config.txt')

bot = telebot.TeleBot(config['Telegram']['TelegramBotToken'])

INFORM_MESSAGE = 'Test mode. Please send a message to determine your intent.'

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.from_user.id, INFORM_MESSAGE)


@bot.message_handler(content_types=['text'])
def get_text_message(message):
    print(message)
    bot.send_message(message.from_user.id,  'Bot received your message')


bot.polling(none_stop=True, interval=0)