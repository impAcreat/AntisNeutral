import torch
from torch.nn import Parameter
from common import AttnAggregation
from GCNs import DenseGCNModel
from GATs import DenseGATModel

from config import CLConfig

'''
CL(Contrastive Learning)
'''
class CLModel(torch.nn.Module):
    def __init__(self, CLConfig):
        super(CLModel, self).__init__()

        # TODO: why adding?
        self.weight = Parameter(torch.Tensor(256, 256))

        self.gcn = DenseGCNModel(CLConfig.graph_dims,
                                 CLConfig.feature_mlp_dims,
                                 CLConfig.dr_mlp_dims,
                                 CLConfig.pr_mlp_dims,
                                 CLConfig.gcnConfig.dropout_rate)
        self.gat = DenseGATModel(CLConfig.graph_dims,
                                 CLConfig.feature_mlp_dims,
                                 CLConfig.dr_mlp_dims,
                                 CLConfig.pr_mlp_dims,
                                 CLConfig.gatConfig.dropout_rate)
        
        self.AttnAgg = AttnAggregation(CLConfig.AttnAggConfig.layer_dims,
                                       dropout_rate=CLConfig.AttnAggConfig.dropout_rate,
                                       relu_layers_index=[0],
                                       dropout_layers_index=[0, 1])


    def forward(self, graph):
        # TODO
        num_node1s, num_node2s = graph.num_node1s, graph.num_node2s
        
        ## embedding based on graph
        gcn_embedding = self.gcn(graph, supplement_x = None)[-1]
        gat_embedding = self.gat(graph, supplement_x = None)[-1]
        ## attention aggregation
        AttnAgg_gcn = self.AttnAgg(gcn_embedding)
        AttnAgg_gat = self.AttnAgg(gat_embedding)
        ## 
        antibody_embedding = torch.cat([gcn_embedding[:num_node1s], gat_embedding[:num_node1s]], 1)
        antigen_embedding = torch.cat([gcn_embedding[num_node1s:], gat_embedding[num_node1s:]], 1)
        
        ## loss
        dgi_gcn = self.loss(gcn_embedding, gat_embedding, AttnAgg_gcn)
        dgi_gat = self.loss(gat_embedding, gcn_embedding, AttnAgg_gat)
        
        return antibody_embedding, antigen_embedding, dgi_gcn, dgi_gat

    ## discriminate: get the similarity of two embeddings
    def discriminate(self, z, AttnAgg, sigmoid=True):
        value = torch.matmul(z, torch.matmul(self.weight, AttnAgg))
        return torch.sigmoid(value) if sigmoid else value

    ## get loss: total_loss = pos_loss + neg_loss
    def loss(self, z1, z2, AttnAgg):
        EPS = 1e-15 # avoid log(0)
        
        pos_loss = -torch.log(
            self.discriminate(z1, AttnAgg, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            self.discriminate(z2, AttnAgg, sigmoid=True) + EPS).mean()
        
        return pos_loss + neg_loss
        ## TODO: is there better way to define loss?

