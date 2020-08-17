import tensorflow as tf
import os 
import warnings
import pickle as pk
import numpy as np
# import argparse

import preprocessing as p
from train import train
import model
import eval
import constants 


# initializing all the variables.

path = constants.PATH
sent_limit = constants.SENT_LIMIT
BATCH_SIZE = constants.BATCH_SIZE
embedding_dim = constants.EMBEDDING_DIM
units = constants.UNITS





def trainModel(path):

    global sent_limit,BATCH_SIZE,embedding_dim,units

    #our input is spanish and output is english
    #returning train test split and input , target tokenizer 

    print('preproessing data ..\ncreating train test split..')

    input_tensor_train, input_tensor_val, target_tensor_train,\
         target_tensor_val,inp_lang,targ_lang = p.get_train_test_data(path,sent_limit)

    print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))


    # params  
    max_length_targ = target_tensor_train.shape[1]
    max_length_inp = input_tensor_train.shape[1]
    # BUFFER_SIZE = len(input_tensor_train)
    steps_per_epoch = len(input_tensor_train)//BATCH_SIZE  
    vocab_inp_size = len(inp_lang.word_index)+1
    vocab_tar_size = len(targ_lang.word_index)+1

    ### saving values in numpy input file 

    np.save('max_value_file',np.array([max_length_inp,max_length_targ])) 



    print(f'steps per epochs {steps_per_epoch}')
    dataset  = p.get_tf_dataset(input_tensor_train, target_tensor_train,BATCH_SIZE)



    print("tf dataset created....")

    ####
    #
    #training phase:
    #
    ####
     
    # checkpoint_dir = './training_checkpoints'
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")   

    # print(f'check point dir : {checkpoint_dir}') 


    print("training started ...")
    print(f'number of samples {sent_limit}')


    encoder = model.create_model("encoder",vocab_inp_size,embedding_dim,units,BATCH_SIZE)
    decoder = model.create_model("decoder",vocab_tar_size,embedding_dim,units,BATCH_SIZE)


    train(dataset,BATCH_SIZE,steps_per_epoch,inp_lang,targ_lang,encoder,decoder)


    print("training ended...")

    ### saving section 
    # saving the tokenizers 

    pk.dump(inp_lang,open("input_tokenizer","wb"))
    pk.dump(targ_lang,open("output_tokenizer","wb"))
 






def evalModel(text = " ",sent = True):


    global sent_limit,BATCH_SIZE,embedding_dim,units    

    ##################
    #
    # evaluation phase 
    # loading the data
    #
    #################

    print("eval phase")


    #loading tokenizers 

    inp_lang = pk.load(open("input_tokenizer","rb"))
    targ_lang = pk.load(open("output_tokenizer","rb"))
    

    vocab_inp_size = len(inp_lang.word_index)+1
    vocab_tar_size = len(targ_lang.word_index)+1

    max_length_inp,max_length_targ = np.load('max_value_file.npy')

    print(max_length_inp,max_length_targ)


    encoder = model.load_model("encoder",vocab_inp_size,embedding_dim,units,BATCH_SIZE)
    decoder = model.load_model("decoder",vocab_tar_size,embedding_dim,units,BATCH_SIZE)

    if sent:

        return eval.translate(text,max_length_targ, max_length_inp,units,
                    inp_lang,targ_lang,encoder,decoder)   


    else:    

        ans = text


        
        while ans != "quit":

            try:   

                ans = input("Enter Text to translate:")
                eval.translate(ans,max_length_targ, max_length_inp,units,
                    inp_lang,targ_lang,encoder,decoder)


            except Exception:
                print("no response please enter again.")


if __name__ == "__main__":



    # arg = argparse.ArgumentParser()
    # arg.add_argument("model_name")


        op = int(input("Enter::\n1 to train only\n2 to eval only\n3 train and eval\n"))

        print(op)

        if op == 1:
            trainModel(path)
        elif op == 2:
            evalModel(sent = False)
        else:
            trainModel(path)
            evalModel(sent = False)        




   

   