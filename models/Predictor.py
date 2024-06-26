import torch
from common import EmbeddingBlock

'''
Predictor:
    - concat antibody feature and antigen feature
    - pass through MLP to get prediction
    params:
    - mlp_layers_dim
    - dropout rate
'''
class Predictor(torch.nn.Module):
    def __init__(self, mlp_layers_dim, dropout_rate=0):
        super(Predictor, self).__init__()

        self.mlp_layers_dim = mlp_layers_dim
        
        self.mlp = EmbeddingBlock(mlp_layers_dim, dropout_rate, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])

    def forward(self, data, ab_embedding, ag_embedding):
        ab_id, ag_id = data.drug_id, data.target_id
        
        ab_feature = ab_embedding[ab_id.int().cpu().numpy()]
        ag_feature = ag_embedding[ag_id.int().cpu().numpy()]
        concat_feature = torch.cat([ab_feature, ag_feature], dim = 1)

        mlp_embeddings = self.mlp(concat_feature)
        
        pred = mlp_embeddings[-1]
        return pred