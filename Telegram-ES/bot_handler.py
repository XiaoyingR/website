# install python-telegram client using: pip install python-telegram-bot --upgrade

from telegram.ext import Updater
updater = Updater(token='445545317:AAH3ML4diFA7LWGXliV0BIMCXiNvovyOVOg') # Token generated at BotFather
dispatcher = updater.dispatcher

def start(bot, update):
    opening_msg = "Welcome to MSO search bot.\nType a keyword to receive three relevant records."
    bot.send_message(chat_id=update.message.chat_id, text=opening_msg)
# COMMAND handler
from telegram.ext import CommandHandler
start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)

from telegram.ext import MessageHandler, Filters

# MESSAGE handler
from ElasticSearch_Telegram import keyword_search
from telegram.ext import MessageHandler, Filters
def key_search(bot, update):
    t = update.message.text
    text_list, user_list, timeStamp_list = keyword_search('* ' + t + ' *')

    if len(user_list) < 3:  # search lowercase if not enough
        a, b, c = keyword_search('* ' + t.lower() + ' *')
        text_list += a[:(3 - len(text_list))]
        user_list += b[:(3 - len(user_list))]
        timeStamp_list += c[:(3 - len(timeStamp_list))]

    # still no match, return empty search
    if user_list == []:
        bot.send_message(chat_id=update.message.chat_id, text="No records match your keyword.")
    else:
        for num in range(len(timeStamp_list)):
            #             print(num)
            bot.send_message(chat_id=update.message.chat_id,
                             text="%s posted at %s:\n%s" % (user_list[num], timeStamp_list[num], text_list[num]))


key_search_handler = MessageHandler(Filters.text, key_search)
dispatcher.add_handler(key_search_handler)

updater.start_polling()

"""
Warning：
This script has to be stopped from the terminal using Ctrl+C.
"""