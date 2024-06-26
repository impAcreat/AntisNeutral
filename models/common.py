import torch
from torch.nn import Linear, ReLU, Dropout

'''
EmbeddingBlock:
    - A block of linear layers with ReLU and Dropout.
    - generate embeddings for each layer
    params:
    - linear layer dim
    - dropout rate
    - usage of ReLU and Dropout(defined by index)
    process:
    - append outputs rather than update one output
'''
class EmbeddingBlock(torch.nn.Module):
    def __init__(self, layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[]):
        super(EmbeddingBlock, self).__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(len(layers_dim) - 1):
            layer = Linear(layers_dim[i], layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        output = x
        embeddings = [x]
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(output) 
        return embeddings

'''
AttnAggregation:
    - Attention Aggregation
    - generate attention weights and aggregate embeddings
    params:
    - summary layer dim
    - dropout rate
    - usage of ReLU and Dropout(defined by index)
    process:
    - update output(attn weights)
    - aggregate embeddings by attention weights
'''
class AttnAggregation(torch.nn.Module):
    def __init__(self, layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[]):
        super(AttnAggregation, self).__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(len(layers_dim) - 1):
            layer = Linear(layers_dim[i], layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        output = x
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
                
        m = torch.tanh(torch.squeeze(output))
        m = torch.exp(m) / (torch.exp(m)).sum() 

        x = torch.matmul(m, x)
        return x
