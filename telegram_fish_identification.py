import sys
import sqlite3

import numpy as np
from PIL import Image
import tensorflow as tf

import os
import logging
import telegram
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Updater, CommandHandler, MessageHandler, ConversationHandler, Filters

CLASSES = ['Grouper', 'Pomfret', 'Snapper']

LOCATION = 0
CANCEL = 1
PHOTO = 2
SPECIES = 3
LENGTH = 4
TIME = 5
CONFIRMATION = 6

# create SQlite database
database = "entries.db"


def loadDB():
    conn = sqlite3.connect(database)
    cur = conn.cursor()
    conn.text_factory = str
    cur.executescript('''CREATE TABLE IF NOT EXISTS fishdata
    (
    id INTEGER NOT NULL PRIMARY KEY UNIQUE, 
    Species TEXT,
    Length TEXT,
    Time TEXT,
    Photo BLOB);'''
                      )
    conn.commit()
    conn.close()


def create_entry(entry):
    sql = ''' INSERT INTO fishdata(Species,Length,Time, Photo) VALUES(?,?,?,?) '''
    conn = sqlite3.connect(database)
    cur = conn.cursor()
    cur.execute(sql, entry)
    conn.commit()
    return cur.lastrowid


# load database
loadDB()
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

reply_keyboard = [['Confirm', 'Restart']]
markup = ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, one_time_keyboard=True)
TOKEN = '????'
bot = telegram.Bot(token=TOKEN)
chat_id = '@sgwildlife'
updater = Updater(TOKEN, use_context=True)
dispatcher = updater.dispatcher


# helper function to convert digital data to binary format
def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData


def facts_to_str(user_data):
    facts = list()

    for key, value in user_data.items():
        facts.append('{} - {}'.format(key, value))

    return "\n".join(facts).join(['\n', '\n'])


def start(update, context):
    update.message.reply_text("Hi! I am your fish identification app. Pls enter your location.")
    return LOCATION


def location(update, context):
    user = update.message.from_user
    user_data = context.user_data
    category = 'Location'
    text = update.message.location
    user_data[category] = text
    logger.info("Location of %s: %f / %f", user.first_name, text.latitude, text.longitude)
    update.message.reply_text('Please send a photo of the fish')
    return PHOTO


def photo(update, context):
    user = update.message.from_user
    user_data = context.user_data
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('user_photo.jpg')
    global photo
    photo = convertToBinaryData('user_photo.jpg')
    logger.info("Photo of %s: %s", user.first_name, 'user_photo.jpg')
    image = tf.keras.preprocessing.image.load_img('user_photo.jpg', target_size=(299, 299))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    classifier = tf.keras.models.load_model("fish_ident_model.h5", compile=False)
    prediction = classifier.predict([image])[0]
    prediction = np.argmax(prediction)
    class_name = CLASSES[prediction]
    category = 'Species'
    user_data[category] = class_name
    logger.info("Name of the species: %s", class_name)
    update.message.reply_text('This fish is a {}'.format(class_name))
    update.message.reply_text('What is the length of the fish?')
    return LENGTH


def length(update, context):
    user = update.message.from_user
    user_data = context.user_data
    category = 'Length'
    text = update.message.text
    user_data[category] = text
    logger.info("Length of fish: %s", update.message.text)
    update.message.reply_text('What time of the fish caught or sighted?')
    return TIME


def time(update, context):
    user = update.message.from_user
    user_data = context.user_data
    category = 'Time of Sight'
    text = update.message.text
    user_data[category] = text
    logger.info("Time of Sight: %s", update.message.text)
    update.message.reply_text("Thank you for providing the information! Please check the information is correct:"
                              "{}".format(facts_to_str(user_data)), reply_markup=markup)
    return CONFIRMATION


def confirmation(update, context):
    user = update.message.from_user
    user_data = context.user_data
    update.message.reply_text("Thank you! I will post the information on the channel @" + chat_id + "  now.",
                              reply_markup=ReplyKeyboardRemove())
    bot.send_photo(chat_id=chat_id, photo=open('user_photo.jpg', 'rb'),
                   caption="Check the details below: \n {}".format(facts_to_str(user_data)),
                   parse_mode=telegram.ParseMode.HTML)
    entry = (user_data['Species'], user_data['Length'], user_data['Time of Sight'], photo)
    entry_id = create_entry(entry)
    return ConversationHandler.END


def cancel(update, context):
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text('You have exited the app.',
                              reply_markup=ReplyKeyboardRemove())

    return ConversationHandler.END


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


conv_handler = ConversationHandler(
    entry_points=[CommandHandler('start', start)],

    states={

        LOCATION: [CommandHandler('start', start), MessageHandler(Filters.location, location)],

        PHOTO: [CommandHandler('start', start), MessageHandler(Filters.photo, photo)],

        LENGTH: [CommandHandler('start', start), MessageHandler(Filters.text, length)],

        TIME: [CommandHandler('start', start), MessageHandler(Filters.text, time)],

        CONFIRMATION: [MessageHandler(Filters.regex('^Confirm$'), confirmation),
                       MessageHandler(Filters.regex('^Restart$'), start)]

    },

    fallbacks=[CommandHandler('cancel', cancel)])

dispatcher.add_handler(conv_handler)
# Add error handlers
updater.start_polling()
updater.idle()
