import configparser
import telebot
from natureLanguageUnderstanding.NatureLanguageUnderstanding import NLU

config = configparser.ConfigParser()
config.read('../../resources/config.txt')

bot = telebot.TeleBot(config['Telegram']['TelegramBotToken'])

INFORM_MESSAGE = 'Test mode. Please send a message to determine your intent.'
nlu_module = NLU()

btn_1 = 'Correctly'
btn_2 = 'Wrong, I will send the correct intent'
btn_3 = 'Pass'

STATE_ONE = 1
STATE_TWO = 2

state = STATE_ONE

@bot.message_handler(commands=['start'])
def send_inform_message(message):
    bot.send_message(message.from_user.id, INFORM_MESSAGE)


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    global state
    if (state == STATE_ONE):
        result = get_intent(message)
        keyboard = types.InlineKeyboardMarkup()
        b1 = types.InlineKeyboardButton(text=btn_1, callback_data='1')
        keyboard.add(b1)
        b2 = types.InlineKeyboardButton(text=btn_2, callback_data='2')
        keyboard.add(b2)
        b3 = types.InlineKeyboardButton(text=btn_3, callback_data='3')
        keyboard.add(b3)
        msg = bot.reply_to(message, 'Your intent code: ' + str(result), reply_markup=keyboard)
        bot.register_next_step_handler(msg, process_step)
        
    if (state == STATE_TWO):
        # Here will be data saving
        bot.send_message(message.from_user.id, 'Thanks, data saved. Please send another message')


def get_intent(message):
    intent_code = nlu_module.predict_intent(message.text)
    # Here will be intent getting from database
    return intent_code

def process_step(message):
    if message.text==btn_1:
        bot.send_message(message.from_user.id, 'Good! Please send another message.')
    elif message.text == btn_2:
        global state
        state = STATE_TWO
        bot.send_message(message.from_user.id, 'Waiting for correct intent...')
    else:
        bot.send_message(message.from_user.id, 'Passed. Please send another message.')

bot.polling(none_stop=True, interval=0)