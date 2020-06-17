import configparser
import telebot
from natureLanguageUnderstanding.NatureLanguageUnderstanding import NLU
from dialogManagement.databaseConnector.DatabaseConnector import DatabaseConnector

config = configparser.ConfigParser()
config.read('../../resources/config.txt')

bot = telebot.TeleBot(config['Telegram']['TelegramBotToken'])

INFORM_MESSAGE = 'Test mode. Please send a message to determine your intent.'

nlu_module = NLU()
database_connector = DatabaseConnector()

btn_1 = 'Correctly'
btn_2 = 'Wrong, I will send the correct intent'
btn_3 = 'Pass'

STATE_ONE = 1
STATE_TWO = 2
ERROR_MESSAGE = 'ERROR'
OK_MESSAGE = 'ОК'

state = STATE_ONE
message_id = None

@bot.message_handler(commands=['start'])
def send_inform_message(message):
    bot.send_message(message.from_user.id, INFORM_MESSAGE)


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    global state
    if (state == STATE_ONE):
        result_id = database_connector.save_message(message.json)
        if result_id != ERROR_MESSAGE:
            bot.send_message(message.from_user.id, 'Bot received your message')
            global message_id
            message_id = result_id
            result = get_intent(message)
            if result != ERROR_MESSAGE:
                keyboard = types.InlineKeyboardMarkup()
                b1 = types.InlineKeyboardButton(text=btn_1, callback_data='1')
                keyboard.add(b1)
                b2 = types.InlineKeyboardButton(text=btn_2, callback_data='2')
                keyboard.add(b2)
                b3 = types.InlineKeyboardButton(text=btn_3, callback_data='3')
                keyboard.add(b3)
                msg = bot.reply_to(message, 'Your intent: ' + str(result), reply_markup=keyboard)
                bot.register_next_step_handler(msg, process_step)
            else:
                bot.send_message(message.from_user.id, ERROR_MESSAGE)
        else:
            bot.send_message(message.from_user.id, ERROR_MESSAGE)

    if (state == STATE_TWO):
        global message_id
        result = database_connector.save_correct_intent_for_message(message.text, message_id)
        if result != ERROR_MESSAGE:
            bot.send_message(message.from_user.id, 'Thanks, data saved. Please send another message')
        else:
            bot.send_message(message.from_user.id, ERROR_MESSAGE)
        message_id = None


def get_intent(message):
    intent_code = nlu_module.predict_intent(message.text)
    global message_id
    return database_connector.load_intent_by_code(intent_code, message_id)


def process_step(message):
    if message.text == btn_1:
        bot.send_message(message.from_user.id, 'Good! Please send another message.')
    elif message.text == btn_2:
        global state
        state = STATE_TWO
        bot.send_message(message.from_user.id, 'Waiting for correct intent...')
    else:
        bot.send_message(message.from_user.id, 'Passed. Please send another message.')


bot.polling(none_stop=True, interval=0)