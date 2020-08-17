# All-Lang-Translator

## The goal of this project is to learn about seq-2seq models using attention,experiment with different attention mechanisms and model atchitectures.

## Finally Create an all language translator. The Idea is to provided users with Gui based web application where users can just download data eg.
spanish to engish translation click on the train button and there you have a nmt model. If a model is already trained users can inference it. 

About the problem: 
NMT or neural machine translation is a problem where we try to translate a date from one langauge to another. For this we make use of the seq-2-seq model architecture
Here we have an encoder which takes the input language and outputs an encoded vector for each input word. This encoded output is then fed into the decoder alon with 
the decoders previous hidden state and the previous word. 
 
Note: 1. that the encoder takes in a sentence but the decoder doesn't output an sentence but the decoder output one word at a time until it reaches the </end> tag.
          The input to the encoder is the padded input langauge sentences. The decoder takes 3 inputs the previous predicted word vector , the prevoius hidden state 
	  and the encoders output.		

      2. The input and output language can be interchanged. If you have a english - 2 - spanish dataset , we can make to spanish - 2 - english no need to find new 
	 data.			
