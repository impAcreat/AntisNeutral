import torch
from torch.nn import ReLU, Dropout
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_adj
from common import EmbeddingBlock

class DenseGATBlock(torch.nn.Module):
    def __init__(self, gat_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[]):
        super(DenseGATBlock, self).__init__()
        self.conv_layers = torch.nn.ModuleList()
        for i in range(len(gat_layers_dim) - 1):
            conv_layer = GATConv(gat_layers_dim[i], gat_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)
        
        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_indexs_dropout):
        output = x
        embeddings = [output]
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, edge_indexs_dropout)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(torch.squeeze(output, dim=0))
        return embeddings


class DenseGATModel(torch.nn.Module):
    def __init__(self, layers_dim, 
                 feature_mlp_dims, dr_mlp_dims, pr_mlp_dims,
                 edge_dropout_rate=0):
        super(DenseGATModel, self).__init__()
        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = DenseGATBlock(layers_dim, 0.1, relu_layers_index=range(self.num_layers), dropout_layers_index=range(self.num_layers))

        self.feature_mlp = EmbeddingBlock(feature_mlp_dims, 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])
        self.mlp_dr = EmbeddingBlock(dr_mlp_dims, 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])
        self.mlp_pr = EmbeddingBlock(pr_mlp_dims, 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])
        
    def forward(self, graph):
        feature, dr_EMB, pr_EMB, adj, num_node1s, num_node2s =  graph.features, graph.drEMBED, graph.prEMBED, graph.adj, graph.num_node1s, graph.num_node2s
        
        feature_mlp = self.feature_mlp(feature)[-1]
        dr_EMB_mlp = self.mlp_dr(dr_EMB)[-1]
        pr_EMB_mlp = self.mlp_pr(pr_EMB)[-1]
        
        both_EMB = torch.cat((dr_EMB_mlp, pr_EMB_mlp), 0)
        xs = torch.cat((both_EMB, feature_mlp), 1)

        indexs = torch.where(adj != 0)
        edge_indexs = torch.cat((torch.unsqueeze(indexs[0], 0), torch.unsqueeze(indexs[1], 0)), 0)

        edge_indexs_dropout, edge_weights_dropout = dropout_adj(edge_index=edge_indexs, edge_attr=adj[indexs], p=self.edge_dropout_rate, force_undirected=True, num_nodes=num_node1s + num_node2s, training=self.training)
        adj_dropout = torch.zeros_like(adj)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout
        
        embeddings = self.graph_conv(xs, edge_indexs_dropout)
        return embeddings
