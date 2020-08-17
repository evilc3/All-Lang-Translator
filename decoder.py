import tensorflow as tf
from attention import AdditiveAttention
import numpy as np

class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.gru = tf.keras.layers.GRU(self.dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = AdditiveAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights



if __name__ == "__main__":

    vocab_inp_size = 100
    embedding_dim = 32
    units = 100
    BATCH_SIZE = 32
   
    tf.keras.backend.set_floatx('float64')


    decoder = Decoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

    sample_hidden = np.zeros((32,units))
    example_input_batch = np.ones((32,1))
    encoder_op = np.ones((32,11,1))

    op,state,attention_weights = decoder(example_input_batch, sample_hidden,encoder_op)

    print ('deoder output shape (remember the decoder ouputs one word at a time) : (batch size, units) {}'.format(op.shape))
    print ('Encoder Hidden state shape: (batch size, units) {}'.format(state.shape))

    print(attention_weights.shape)