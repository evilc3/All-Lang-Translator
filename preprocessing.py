# This file reads in the dataset which consist of tab seperated sentences 
# where sentence A is of forign and sentence B is of english 
# We first seperate this sentences 
# Format the data 
# Use the tokenizer to tokenzer the data and crate a word - to - id mapping.

import tensorflow as tf
import unicodedata
import re 
import numpy as np
import io 
import time 
from sklearn.model_selection import train_test_split
import constants

"""
Things to do:

1. Remove the . from sent. 
2. covert text to lower 

"""

random_samples = None 

def save_random(word_pairs):

    global random_samples 

    indices = np.random.choice(np.arange(len(word_pairs)),10)
    
    random_samples = np.array(word_pairs)[indices]


def get_random_samples():

    global random_samples
    # random_smaple structure : [[target_sent,input_sent]]
    return random_samples


# Converts the unicode file to ascii
def unicode_to_ascii(s):

  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w,regex):

  w = unicode_to_ascii(w.lower().strip())

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  
  if regex:
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        # w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
        # removing everything except letters

        w = re.sub(r"[^a-zA-Z]+", " ", w)


  w = w.strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w



# read the data from the specified file and return sentence_paris
#output : [english,spanish]
def get_data(path,number_sent):

    file = io.open(path,encoding = "UTF-8").read()

    lines = file.split("\n")

    if number_sent == None:
        number_sent = len(lines)

    save_random([[w for w in l.split("\t")[:2]] for l in lines[:number_sent]])


    #implement a  better method with this.

    #
    target_words  = [preprocess_sentence(l.split("\t")[0],regex=constants.PREPROCESS_TARGET)  for l in lines[:number_sent]]
    
    input_words = [preprocess_sentence(l.split("\t")[1],regex=constants.PREPROCESS_INPUT) for l in lines[:number_sent]]

    return target_words,input_words

    



#returns the padded sequence of integers 
def tokenize(lang):
  
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                            padding='post')

    return tensor, lang_tokenizer    


def load_dataset(path, num_examples=None):
   
    # creating cleaned input, output pairs
    targ_lang, inp_lang = get_data(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer    


def get_train_test_data(path,num_examples = None,test_size = 0.2):


    input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer  = load_dataset(path,num_examples)

    # max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
   
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=test_size)

    return input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val,inp_lang_tokenizer, targ_lang_tokenizer


def get_tf_dataset(input_tensor_train, target_tensor_train,batch_size):


    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(batch_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


