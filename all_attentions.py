#the aim of this file is to hold all the attentions 

import tensorflow as tf
import  keras.backend as K
from tensorflow.keras.layers import Layer

import numpy as np

'''
1. attention
2. attentionContext
3. AdditiveAttention
4. ScaledDotProductAttention
5. MultiHeadAttention
'''



class Attention(Layer):
    
    
    def __init__(self):
        super(Attention, self).__init__()
        
        
        
    def build(self,input_shape):
        
        self.W = self.add_weight(
                                shape=(input_shape[-1],1),
                                initializer="he_normal",
                                dtype = "float32",    
                                trainable = True,
                                name = "W"
                                )
        self.B = self.add_weight(
                                shape=(input_shape[1],1),
                                initializer="ones",
                                trainable = True,
                                name = "B"
                                )           
        
        super(Attention, self).build(input_shape)    
        

    def call(self, x,bias = False):

    
        energies = tf.matmul(x,self.W)
        
        
        
        if bias:
            energies = tf.add(energies,self.B)
        
        
        energies = tf.tanh(energies)
    
        attention_vector = tf.math.softmax(energies)
        
        context_vecotor = x * attention_vector
        
        return tf.reduce_sum(context_vecotor,axis = 1)
    
    
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(Attention,self).get_config()
    
class AttentionContext(Layer):
    
    
    def _init__(self):
        super(AttentionContext,self).__init__()
        
       
        
        
    def build(self,input_shape):
        
        '''
        we need to initialize 2 matrices and 1 bias vector.
        The extra  matrix is the context which is multiplied to the energies 
        
        Here the matrix W will have shape (dim x dim) where dim is the feature dim.
        
        The final output need to remain same as the simple attention to achieve this 
        the vector W is converted into matrix 
        
        The bias will also have different dimensions from the original attention mechanism
        but will will still be a vector 
        
        
        The U vector which is the main change here is called the context vector 
        responsiable for  finding out important information from the energies 
        
        
        '''
        
        self.W = self.add_weight(
                                shape = (input_shape[-1],input_shape[-1]),
                                initializer = "he_normal",
                                dtype = 'float32',
                                trainable = True,
                                name = "W"
                                )
        
        self.B = self.add_weight(
                                shape = (input_shape[-1],),
                                initializer = "ones",
                                dtype = 'float32',
                                trainable = True,
                                name = "B"
                                )
        
        self.U = self.add_weight(
                                shape = (input_shape[-1],),
                                initializer = "he_normal",
                                dtype = 'float32',
                                trainable = True,
                                name = "B"
                                )
        
        
    # I am not implementing mask 
    def call(self,x,bias = True):
            
            
            assert len(x.shape) == 3
            
            energies = K.squeeze(K.dot(x, K.expand_dims(self.W)),axis = -1)  # output (batch_size,steps,dims)
            
 
            
            if bias:
                
                energies += self.B     # output (batch_size,steps,dims)       
            
            energies = K.tanh(energies)
            
            print(energies.shape) 
            
            context_vector = K.dot(energies,K.expand_dims(self.U))# output (batch_size,steps,1)
            
            
            attention_weights = K.softmax(context_vector,axis = 1)
            
            
            
            
            context = x * attention_weights 

            
            return   K.sum(context,axis = 1)
                    
class AdditiveAttention(tf.keras.layers.Layer):
    
    
      def __init__(self, units):
            super(AdditiveAttention, self).__init__()
            self.units = units
    
    
      def build(self,input_shape):   
            
            print('input_shape',input_shape)
    
            self.W1 = self.add_weight(
                                     shape = (input_shape[-1],self.units),
                                     initializer = 'he_normal',
                                     dtype = 'float32',
                                     trainable = True,
                                     name = "W1"
                                    )
        
            self.W2 = self.add_weight(
                                      shape = (input_shape[-1],self.units),
                                      initializer = 'he_normal',
                                      dtype = 'float32',
                                      trainable = True,
                                      name = "W2"
                                     )
            self.V1 =  self.add_weight(shape = (self.units,1),
                                     initializer = 'he_normal',
                                     dtype = 'float32',
                                     trainable = True,
                                     name = "V"
                                    )
            
            
                                     

      def call(self, query, values):
    
    
            '''
            value  - (batch_size,steps,dim)
            query  - (batch_size,1,dim) - > expanded query

            dense layer only changes the last layer 

            op1 =  W1 * query  =>  (batch_size,1,units)

            op2 = W2 * values =>  (batch_size,steps,units)  

            ADD =  op1 + op2 during this op1 will be broadcasted to match shape of op2 => (batch_size,steps,units)

            op3 = tanh(ADD)

            energies = V * op3 => (batch_size,steps,1)

            normalizerd_energies/ attention_weights = softmax(energises) => (batch_size,steps,1)

            The reason we normalzie is to maked the dot product of long sequences have similar magnited of dot of small vectors 

            context_vectors = x * attention_weights  =>(batch_size,steps,dims)

            return context_vectors 


            '''
    
    
            assert len(query.shape) == 2
            assert len(values.shape) == 3
    

            #expanded_query = tf.expand_dims(query, axsi = 1)
            
            Q = K.dot(query,self.W1) # output: (batch_size,units)
            
            # print('Q:',Q.shape)
            
            V = K.dot(values,self.W2) # output: (batch_size,steps,units)
            
            # print('V:',V.shape)
            
               
            ADD = K.expand_dims(Q,axis = 1) + V # output : (batch_size,steps,units)
            
            # print('ADD:',ADD.shape)
                
            score = K.dot(ADD,self.V1)
            
            # print('Score:',score.shape)

    
            # attention_weights shape == (batch_size, max_length, 1)
            attention_weights = K.softmax(score, axis=1)

            # context_vector shape after sum == (batch_size, hidden_size)
            context_vector = attention_weights * values
            context_vector = K.sum(context_vector, axis=1)

            
            # print('context_vector:',context_vector.shape)
            
            return context_vector

class ScaledDotProductAttention(Layer):
    
    
    def __init__(self,dim):
        
        super(ScaledDotProductAttention,self).__init__()
        
        self.dim = dim
        
    def build(self,input_shape):
        
        '''
        
        input_shape = (batch_size,steps,dims)
        
        steps = dmodel
        
        dims = dk
        
        matirx Wq,Wk,Wv
        
        shape of all is same dmodel x dk for Wq and Wk
        and dmodel x dv for Wv
        
        we take dk == dv but they can be different.
        
        '''
        
        self.Wq = self.add_weight( 
                                  shape = (input_shape[-1],self.dim),
                                  initializer = "he_normal",
                                  dtype = "float32",
                                  trainable = True,
                                  name = "Wq"  
                                 ) 
        
        self.Wk = self.add_weight( 
                                  shape = (input_shape[-1],self.dim),
                                  initializer = "he_normal",
                                  dtype = "float32",
                                  trainable = True,
                                  name = "Wk"  
                                 ) 
        
        
        self.Wv = self.add_weight( 
                                  shape = (input_shape[-1],self.dim),
                                  initializer = "he_normal",
                                  dtype = "float32",
                                  trainable = True,
                                  name = "Wv"  
                                 ) 
        
        
        
    
        
    def call(self,query,key,value):
        
        '''
        For classification tast query == value == key 
        key and value can be the same input.
        
        query shape = (batch_size,steps,dims)
        '''
        
        input_shape = query.shape
        
        Q = tf.matmul(query,self.Wq) #output: (batch_size,steps,step)
        
        K = tf.matmul(query,self.Wk) #output: (batch_size,steps,step)
        
        V = tf.matmul(query,self.Wv) #output: (batch_size,steps,step)
        
        t1 =tf.divide(tf.matmul(Q,K,transpose_b = True),np.sqrt(input_shape[-1]))
        
        t1 = tf.math.softmax(t1,axis = 1)
        
        attention  = tf.matmul(t1,V)
        
        return attention
              
class MultiHeadAttention(Layer):
    
    
    def __init__(self,head,dim):
        
        super(MultiHeadAttention,self).__init__()
        
        self.dim = dim
        self.head = head
        
    def build(self,input_shape):
        
        '''
        
        input_shape = (batch_size,steps,dims)
        
        dims = dmodel
        
        dk = units 
        
        matirx Wq,Wk,Wv
        
        shape of all is same dmodel x dk for Wq and Wk
        and dmodel x dv for Wv
        
        we take dk == dv but they can be different.
        
        
        output of the embedding 100,75,300
        
        '''
        
        self.Wq = self.add_weight( 
                                  shape = (input_shape[0][-1],self.dim * self.head),
                                  initializer = "he_normal",
                                  dtype = "float32",
                                  trainable = True,
                                  name = "Wq"  
                                 ) 
        
        self.Wk = self.add_weight( 
                                  shape = (input_shape[1][-1],self.dim * self.head),
                                  initializer = "he_normal",
                                  dtype = "float32",
                                  trainable = True,
                                  name = "Wk"  
                                 ) 
        
        
        self.Wv = self.add_weight( 
                                  shape = (input_shape[2][-1],self.dim * self.head),
                                  initializer = "he_normal",
                                  dtype = "float32",
                                  trainable = True,
                                  name = "Wv"  
                                 ) 
        
        self.Wo = self.add_weight( 
                                  shape = (self.head* self.dim,input_shape[0][-1]),
                                  initializer = "he_normal",
                                  dtype = "float32",
                                  trainable = True,
                                  name = "Wo"  
                                 )  
        


        super(MultiHeadAttention,self).build(input_shape) 
    
        
    def call(self,x):
        
        '''
        For classification tast query == value == key 
        key and value can be the same input.
        
        query shape = (batch_size,steps,dims)
        '''
        
        query,key,value = x
        
        Q = tf.matmul(query,self.Wq) #output: (batch_size,steps,step)
        
        K = tf.matmul(key,self.Wk) #output: (batch_size,steps,step)
        
        V = tf.matmul(value,self.Wv) #output: (batch_size,steps,step)
        
        t1 =tf.divide(tf.matmul(Q,K,transpose_b = True),np.sqrt(self.dim))
        
        t1 = tf.math.softmax(t1,axis = 1)
        
        head  = tf.matmul(t1,V)
        
        context_vector = tf.matmul(head,self.Wo)


        return context_vector
                     
            
         