# All-Lang-Translator

#### The goal of this project is to learn about seq-2seq models using attention,experiment with different attention mechanisms and model atchitectures.

#### Finally Create an all language translator. The Idea is to provided users with Gui based web application where users can just download data eg.
spanish to engish translation click on the train button and there you have a nmt model. If a model is already trained users can inference it. 

## About the problem: 

NMT or neural machine translation is a problem where we try to translate a date from one langauge to another. For this we make use of the seq-2-seq model architecture
Here we have an encoder which takes the input language and outputs an encoded vector for each input word. This encoded output is then fed into the `decoder along with 
the decoders previous hidden state and the previous word.` 
 
**Note:  1. that the encoder takes in a sentence but the decoder doesn't output an sentence but the decoder output one word at a time until it reaches the </end> tag.
     
        2. The input to the encoder is the padded input langauge sentences. The decoder takes 3 inputs the previous predicted word vector , the prevoius hidden state 
	  and the encoders output.		

      3. The input and output language can be interchanged. If you have a english - 2 - spanish dataset , we can make to spanish - 2 - english no need to find new 
	 data.			

## More about the dataset:

The data set comes from http://www.manythings.org/anki/ site it containes abt. 100  of foriegn language - english translation datasets. `The datasets are tab seperated.`
In this repo. I have used **2 datasets** 
	1.  spanish to english 
	2.  hindi to english 
		for hindi to english dataset I have not applied preprocessing to the hindi text as steps used for english dont work on hindi.
                **the constants.py file has two boolean parameters PREPROCESS_INPUT and PREPROCESS_TARGET  if set to true preprocessing is applied if set to false   preprocessing is not applied.
		One could choose to apply preprocessing to one language keeping the other unchanged as the case with hindi-english translation just set 
		PREPROCESS_INPUT = False (input is hindi) and PREPROCESS_TARGET = True.
		For spanish to english other PREPROCESS_INPUT and PREPROCESS_TARGET to True
		**
		
## Preprocessing: 
For preprocessing part I have followed from  google's seq-2-seq attention mechanism.

Problem: They have remove all punctuations except ".", "?", "!", "," the problem is with words like he's which now becomes [he, s]
when split. 

A quick fix would be to change the regex from r"[^a-zA-Z?.!,¿]+" to r"[^a-zA-Z?.!,\'¿]+", by doing this he's , it's will be considered 
as a single word.

Another way is doing contraction correction where will relace it's to it is or he's to he is 

# Training on other datasets.

## Configurations:

It's quite simple to train on a different dataset just follow the following steps.
1. move the dataset (.txt) file to the data folder 
2. All model parameters are stored in the constants.py file.
3. Remember to change the `PATH` parameter. It should point to your dataset file name eg. change from spa.txt (which spanish2english) to hin.txt if you are doing 
hindi-english translations.

## Training the models 

There are 2 options provided you can use the main.py file by running 
```python  python main.py```

Or 

You can use the streamlit web app  ```streamlit run app.py```


# Experiments 

I like performing experiments when I am learning new things few of the experiments I have performed are 

1. Experiments on different Attention Mechanisms
2. Experiment is changing the Units (output dim of encoder)
3. Experiment is changing the Embedding Dimension
4. Changing the number of Layers 
5. Changing the type of layers ef [gru ,lstm]

In this experiments while changing one parameter eg. number of layers all other parameter are kept constant


### 1.  Experiments on different Attention Mechanisms trained on only 30000 samples
   	
<table>
	<tr>
		<th>Attention Type</th>
		<th> Bleu - 1 Score</th>
		<th> Loss SCE </th>
		<th> Training Time</th>
		
	</tr>
	<tr>     
	        <td>Simple Attention</td>
		<td>64%</td>
		<td>0.009</td>
		<td>270s</td>
		
        </tr>	
	
	<tr>
		<td>Attention With Context</td>
		<td>66%</td>
		<td>0.006</td>
		<td>300s</td>
        </tr>
	
	<tr>
		<td>Additive Attention</td>
		<td>70%</td>
		<td>0.003</td>
		<td>300s</td>
        </tr>
</table>	

observation : Additive attention is better than other attention types. 


### 2. Experiment on different  Units


		
