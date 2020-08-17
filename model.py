from encoder import Encoder 
from decoder import Decoder 
import tensorflow as tf
from tensorflow import keras
import constants 


def load_model(name,vocab_size,embedding_dims,units,batch_size):

    if name == "encoder":
        encoder = Encoder(vocab_size,embedding_dims,units,batch_size)

        # print(encoder)


        encoder.load_weights(constants.ENCODER_SAVE_PATH)

        return encoder
    else:
        decoder = Decoder(vocab_size,embedding_dims,units,batch_size)
        decoder.load_weights(constants.DECODER_SAVE_PATH)

        return decoder


def create_model(name,vocab_size,embedding_dims,units,batch_size):
    
    if name == "encoder":
        encoder = Encoder(vocab_size,embedding_dims,units,batch_size)

   
        return encoder
    else:
        decoder = Decoder(vocab_size,embedding_dims,units,batch_size)
  
        return decoder