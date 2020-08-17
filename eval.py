import tensorflow as tf
from encoder import Encoder 
from decoder import Decoder 
from preprocessing  import preprocess_sentence
import numpy as np
import time as time 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import constants


def evaluate(sentence,max_length_targ, max_length_inp,units,inp_lang,targ_lang,encoder,decoder):
  
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  sentence = preprocess_sentence(sentence,regex = constants.PREPROCESS_INPUT)

  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]

  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')

  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))

    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targ_lang.index_word[predicted_id] + ' '

    if targ_lang.index_word[predicted_id] == '<end>':
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot

# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
      
  # fig = plt.figure(figsize=(10,10))
  # plt = fig.add_subplot(1, 1, 1)
  return plt.matshow(attention, cmap='vidis')

  
  # fontdict = {'fontsize': 14}

  # ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  # ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  # plt.show()  


def translate(sentence,max_length_targ, max_length_inp,units,inp_lang,targ_lang,encoder,decoder):
     
    result, sentence, attention_plot = evaluate(sentence,max_length_targ, max_length_inp,units,
                                                inp_lang,targ_lang,encoder,decoder)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    # plot_attention(attention_plot, sentence.split(' '), result.split(' '))  

    return result,sentence,attention_plot