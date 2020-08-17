import tensorflow as tf
from encoder import Encoder 
from decoder import Decoder 
import time as time 
import pickle as pk
import constants
import eval
import preprocessing as p


EPOCHS = constants.EPOCHS
encoder_save_path = constants.ENCODER_SAVE_PATH
decoder_save_path = constants.DECODER_SAVE_PATH




optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)





# @tf.function
def train(dataset,BATCH_SIZE,steps_per_epoch,inp_lang,targ_lang,encoder,decoder):
    

    # encoder = Encoder(vocab_inp_size,embedding_dim,units,BATCH_SIZE)
    # decoder = Decoder(vocab_tar_size,embedding_dim,units,BATCH_SIZE)    
    # checkpoint = tf.train.Checkpoint(optimizer=optimizer,
    #                                 encoder=encoder,
    #                                 decoder=decoder)



    for epoch in range(EPOCHS):
        
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0.0


        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):

            loss = 0

            with tf.GradientTape() as tape:

                enc_output, enc_hidden = encoder(inp, enc_hidden)
                dec_hidden = enc_hidden

                #initial input 
                dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):

                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                    loss += loss_function(targ[:, t], predictions)

                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (loss / int(targ.shape[1]))

            variables = encoder.trainable_variables + decoder.trainable_variables

            gradients = tape.gradient(loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))



            total_loss +=  batch_loss

            if batch % 10 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,batch,batch_loss))
        
        
        
        
        # saving (checkpoint) the model every 2 epochs
        # if (epoch + 1) % 2 == 0:
        #     checkpoint.save(file_prefix = checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

     
    print("saving encoder and decoder...")

    # tf.saved_model.save(encoder,"saved_model/encoder/")
    # tf.saved_model.save(decoder,"saved_model/decoder/")
    encoder.save_weights(encoder_save_path)
    decoder.save_weights(decoder_save_path)
    
    print("encoder and decoder saved in saved_models folder")
   
    print("printing few results")


    random_samples = p.get_random_samples()

    for target_sent,input_sent in random_samples:

        # print("************************************\n")


        # print(f'input sent : {input_sent}')
        # print(f'target sent : {target_sent}')

        eval.translate(input_sent,targ.shape[1], 
                                                inp.shape[1],
                                                constants.UNITS,
                                                inp_lang,
                                                targ_lang,
                                                encoder,
                                                decoder)

        print(f'actual value : {target_sent}')
        # print(f'predicted sent : {predicted_value}')

        print('**************************************\n')

    # print(eval.translate("ve.",targ.shape[1],inp.shape[1],constants.UNITS,inp_lang,targ_lang,encoder,decoder))
    
    