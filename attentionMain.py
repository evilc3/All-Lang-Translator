from all_attentions import * 


class AllAttention():
    
    '''
    1. attention
    2. attentionContext
    3. AdditiveAttention
    4. ScaledDotProductAttention
    5. MultiHeadAttention
    '''

    def __init__(self):

        self.id_to_layer = {
                            'sa':Attention(),
                            'awc': AttentionContext(),
                            'aa':AdditiveAttention(11),
                            'sdpa':ScaledDotProductAttention(11),
                            'mha':MultiHeadAttention(1,1)
                            }

        self.names = {
                    'sa' : 'Simple Attention',
                    'awc':'Attention With Context',
                    'aa' :'Additive Attention',
                    'sdpa': 'Scalef Dot Product Attention',
                    'mha' : 'MultiHeadAttention'
                    }         


    def get_names(self):
        
        return self.names


    def get_layer(self,id):

        layer = self.id_to_layer.get(id,None)

        if layer == None:
            print(' error layer not found ')

        return layer      


if __name__ == "__main__":

    id = 'sa' 

    print(AllAttention().get_names())

    print(AllAttention().get_layer(id))

    import numpy as np

    input = np.random.random((10,11,11)).astype('float32')

    attention = AllAttention().get_layer(id)

    print(attention(input).shape)

